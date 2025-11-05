# closures.py
import os
import pickle
import numpy as np
import torch
import Metrics  # uses your base metric functions

# ---------- Core label utilities ----------

def binarize_closure_labels(C, mode: str = "binary") -> torch.Tensor:
    """
    C: torch/np, shape [B,T,N] or [B,T,N,1], values:
       - binary mode: 0/1 (closed if >0.5)
       - time mode  : integers (minutes left), closed if > 0
    Returns: torch.bool mask [B,T,N]
    """
    C = torch.as_tensor(C, dtype=torch.float32)
    if C.dim() == 4 and C.size(-1) == 1:
        C = C[..., 0]
    if mode == "time":
        return (C > 0)
    # default: binary
    return (C > 0.5)


# ---------- Effect matrix loading/slicing ----------

def load_and_slice_effect(effect_matrix_path: str,
                          sensor_indices,
                          device: torch.device) -> torch.Tensor | None:
    """
    Load effect matrix payload (either raw NxN array or dict with 'matrix') and
    slice to THIS split using sensor_indices (positions in data order).
    Returns: torch.float32 [N_split, N_split] on device, or None on failure.
    """
    if not effect_matrix_path:
        return None
    try:
        with open(effect_matrix_path, 'rb') as f:
            obj = pickle.load(f)

        if isinstance(obj, dict) and 'matrix' in obj:
            A_full = np.array(obj['matrix'])
        else:
            A_full = np.array(obj)

        if A_full.ndim != 2 or A_full.shape[0] != A_full.shape[1]:
            return None

        idx = np.asarray(sensor_indices, dtype=int)
        A = torch.as_tensor(A_full, dtype=torch.float32, device=device)
        A = A.index_select(0, torch.as_tensor(idx, dtype=torch.long, device=device))
        A = A.index_select(1, torch.as_tensor(idx, dtype=torch.long, device=device))
        return A
    except Exception as e:
        return None


# ---------- Partition masks (all_open / closure / most / other) ----------

def build_partition_masks(C_bin: torch.Tensor,
                          A_split: torch.Tensor | None = None,
                          top_frac: float = 0.05,
                          include_closed_in_most: bool = True):
    """
    C_bin : torch.bool [B,T,N] (closed or not)
    A_split: torch.float [N,N] effect matrix (attenuation, ~0 is strong effect, ~1 no effect)
    top_frac: proportion for "most affected" using score = |1 - A| aggregated by max across closed cols
    include_closed_in_most: if False, force closed sensors to be excluded from 'most' (still in 'closure')

    Returns dict of masks (torch.bool [B,T,N]):
      - 'all_open'
      - 'closure'
      - optionally 'closure_most', 'closure_other' if A_split provided
    """
    B, T, N = C_bin.shape
    any_closed = C_bin.any(dim=2)                          # [B,T]
    closure = any_closed.unsqueeze(-1).expand(B, T, N)     # [B,T,N]
    all_open = ~closure

    out = {'all_open': all_open, 'closure': closure}

    if A_split is None:
        return out

    # Build MOST via top p% of |1 - A| aggregated across closed sensors at (b,t)
    # Score_i(b,t) = max_j_closed |1 - A[i,j]|
    A_cpu = A_split.detach().cpu()
    most_mask = torch.zeros((B, T, N), dtype=torch.bool)

    cb = C_bin.detach().cpu().numpy()  # [B,T,N]
    for b in range(B):
        for t in range(T):
            closed_j = np.where(cb[b, t])[0]
            if closed_j.size == 0:
                continue
            # aggregate by max over closed columns
            if closed_j.size == 1:
                scores = torch.abs(1.0 - A_cpu[:, closed_j[0]])  # [N]
            else:
                scores = torch.abs(1.0 - A_cpu[:, closed_j]).max(dim=1).values  # [N]

            # top p% indices
            k = max(1, int(np.ceil(N * top_frac)))
            top_idx = torch.topk(scores, k=k, largest=True).indices.cpu().numpy()
            most_mask[b, t, top_idx] = True

            if not include_closed_in_most and closed_j.size > 0:
                most_mask[b, t, closed_j] = False

    out['closure_most']  = closure & most_mask
    out['closure_other'] = closure & (~most_mask)
    return out


# ---------- Closure-aware metrics wrapper ----------

def evaluate_closure_metrics(y_true: torch.Tensor,
                             y_pred: torch.Tensor,
                             CYS_split,
                             effect_matrix_path: str | None = None,
                             sensor_indices=None,
                             closure_mode: str = 'binary',
                             top_frac: float = 0.05,
                             include_closed_in_most: bool = True):
    """
    Orchestrates: binarize labels -> slice effect -> build masks -> compute masked metrics.
    Returns dict { 'all_open': tuple|None, 'closure': tuple|None, 'closure_most':..., 'closure_other':... }
    """
    # to [B,T,N] and bool
    C_bin = binarize_closure_labels(CYS_split, mode=closure_mode).to(y_true.device)
    # effect slice if provided
    A_split = None
    if effect_matrix_path and sensor_indices is not None:
        A_split = load_and_slice_effect(effect_matrix_path, sensor_indices, y_true.device)

    masks = build_partition_masks(
        C_bin=C_bin,
        A_split=A_split,
        top_frac=top_frac,
        include_closed_in_most=include_closed_in_most
    )

    out = {}
    for key, m in masks.items():
        res = Metrics.evaluate_masked(y_true, y_pred, m)  # uses your base Metrics
        out[key] = res


    return out


# ---------- Test-time attenuation (A*) for prediction blending ----------

def build_effect_vector(CYS_split,
                        effect_matrix_path: str,
                        sensor_indices,
                        closure_mode: str,
                        device: torch.device,
                        include_closed_in_most: bool = True) -> torch.Tensor:
    """
    Returns A_star: [B, T, N, 1] in [0,1] per (b,t,i), aggregated from effect matrix
    using max across closed sensors at (b,t).
    If missing inputs, returns ones (no-op).
    """
    C_bin = binarize_closure_labels(CYS_split, mode=closure_mode)
    if C_bin.dim() == 4 and C_bin.size(-1) == 1:
        C_bin = C_bin[..., 0]
    B, T, N = C_bin.shape

    A_split = load_and_slice_effect(effect_matrix_path, sensor_indices, device)
    if A_split is None:
        return torch.ones((B, T, N, 1), device=device, dtype=torch.float32)

    A_cpu = A_split.detach().cpu()
    A_star = torch.ones((B, T, N), dtype=torch.float32)
    cb = C_bin.detach().cpu().numpy()

    for b in range(B):
        for t in range(T):
            closed_j = np.where(cb[b, t])[0]
            if closed_j.size == 0:
                continue
            if closed_j.size == 1:
                scores = A_cpu[:, closed_j[0]]
            else:
                scores = A_cpu[:, closed_j].max(dim=1).values
            A_star[b, t] = scores
            if not include_closed_in_most:
                A_star[b, t, closed_j] = 1.0

    return A_star.to(device)[..., None]  # [B,T,N,1]


# closures.py (minimal; alignment only)

import torch
import numpy as np

def _as_int(x):
    if isinstance(x, int):
        return x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError("Tensor must be scalar for int conversion")
        return int(x.item())
    return int(x)

def align_cys_for_eval(CYS, spatialsplit, sensor_idx_start, n_eval):
    """
    Align closure labels to the same sensor slice used in evaluateModel().
    Inputs:
      CYS: torch/np, shape [B,T,N,1] or [B,T,N]
      spatialsplit: object with .i_tst (list/array of test sensor indices in data order)
      sensor_idx_start: int (0 or len(spatialsplit.i_val) depending on is_testunseen)
      n_eval: int, number of sensors actually used in eval (YS.shape[2])
    Returns:
      CYS_aligned: torch.Tensor [B,T,N_eval,1]
      split_indices_aligned: np.ndarray of sensor indices for THIS slice (length N_eval)
    """
    # to tensor
    C = torch.as_tensor(CYS)
    if C.dim() == 3:     # [B,T,N]
        C = C.unsqueeze(-1)  # [B,T,N,1]
    elif C.dim() == 4 and C.size(-1) == 1:
        pass
    else:
        raise ValueError(f'Unexpected CYS shape {tuple(C.shape)}')

    start = _as_int(sensor_idx_start)
    N_eval = _as_int(n_eval)

    # Slice sensor dimension to match YS/YS_pred eval slice
    CYS_aligned = C[:, :, start:start+N_eval, :]

    # Build the corresponding subset of test sensor indices
    split_indices_full = np.array(spatialsplit.i_tst)
    split_indices_aligned = split_indices_full[start:start+N_eval]

    return CYS_aligned, split_indices_aligned
