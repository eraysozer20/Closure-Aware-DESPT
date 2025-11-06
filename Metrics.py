import os
import pickle
import numpy as np
import torch

def evaluate(y_true, y_pred):
    return MSE(y_true, y_pred), RMSE(y_true, y_pred), MAE(y_true, y_pred), MAPE(y_true, y_pred)

def evaluate_all(y_true, y_pred):
    return MSE_all(y_true, y_pred), RMSE_all(y_true, y_pred), MAE_all(y_true, y_pred), MAPE_all(y_true, y_pred)

# def MSE(y_true, y_pred):
#     # y_true[y_true < 1e-5] = 0
#     # y_pred[y_pred < 1e-5] = 0
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         mask = np.not_equal(y_true, 0)
#         mask = mask.astype(np.float32)
#         mask /= np.mean(mask)
#         mse = np.square(y_pred - y_true)
#         mse = np.nan_to_num(mse * mask)
#         mse = np.mean(mse)
#         return mse

# def MSE_all(y_true, y_pred):
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         mse = np.square(y_pred - y_true)
#         mse = np.mean(mse)
#         return mse
    
# def RMSE(y_true, y_pred):
#     # y_true[y_true < 1e-5] = 0
#     # y_pred[y_pred < 1e-5] = 0
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         mask = np.not_equal(y_true, 0)
#         mask = mask.astype(np.float32)
#         mask /= np.mean(mask)
#         rmse = np.square(np.abs(y_pred - y_true))
#         rmse = np.nan_to_num(rmse * mask)
#         rmse = np.sqrt(np.mean(rmse))
#         return rmse

# def RMSE_all(y_true, y_pred):
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         rmse = np.square(np.abs(y_pred - y_true))
#         rmse = np.sqrt(np.mean(rmse))
#         return rmse
        
# def MAE(y_true, y_pred):
#     # y_true[y_true < 1e-5] = 0
#     # y_pred[y_pred < 1e-5] = 0
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         mask = np.not_equal(y_true, 0)
#         mask = mask.astype(np.float32)
#         mask /= np.mean(mask)
#         mae = np.abs(y_pred - y_true)
#         mae = np.nan_to_num(mae * mask)
#         mae = np.mean(mae)
#         return mae
    
# def MAE_all(y_true, y_pred):
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         mae = np.abs(y_pred - y_true)
#         mae = np.mean(mae)
#         return mae

# def MAPE(y_true, y_pred):
#     # y_true[y_true < 1e-5] = 0
#     # y_pred[y_pred < 1e-5] = 0
#     with np.errstate(divide='ignore', invalid='ignore'):
#         mask = np.not_equal(y_true, 0)
#         mask = mask.astype('float32')
#         mask /= np.mean(mask)
#         mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
#         mape = np.nan_to_num(mask * mape)
#         return np.mean(mape) * 100

# def MAPE_all(y_true, y_pred):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
#         return np.mean(mape) * 100
    
def MSE(y_true, y_pred):
    mask = y_true != 0
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mse = torch.square(y_pred - y_true)
    mse = mse * mask
    mse = torch.where(torch.isnan(mse), torch.zeros_like(mse), mse)
    return torch.mean(mse)

def MSE_all(y_true, y_pred):
    mse = torch.square(y_pred - y_true)
    mse = torch.mean(mse)
    return mse

def RMSE(y_true, y_pred):
    mask = y_true != 0
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    rmse = torch.square(torch.abs(y_pred - y_true))
    rmse = rmse * mask
    rmse = torch.where(torch.isnan(rmse), torch.zeros_like(rmse), rmse)
    return torch.sqrt(torch.mean(rmse))

def RMSE_all(y_true, y_pred):
    rmse = torch.square(torch.abs(y_pred - y_true))
    rmse = torch.mean(rmse)
    return torch.sqrt(rmse)

def MAE(y_true, y_pred):
    mask = y_true != 0
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mae = torch.abs(y_pred - y_true)
    mae = mae * mask
    mae = torch.where(torch.isnan(mae), torch.zeros_like(mae), mae)
    return torch.mean(mae)

def MAE_all(y_true, y_pred):
    mae = torch.abs(y_pred - y_true)
    return torch.mean(mae)

def MAPE(y_true, y_pred):
    mask = y_true != 0
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mape = torch.abs(torch.divide((y_pred - y_true).float(), y_true))
    mape = mape * mask
    mape = torch.where(torch.isnan(mape), torch.zeros_like(mape), mape)
    return torch.mean(mape) * 100

def MAPE_all(y_true, y_pred):
    mape = torch.abs(torch.divide((y_pred - y_true).float(), y_true))
    return torch.mean(mape) * 100


# =========================
# Masked metrics (reuse *_all)
# =========================
def _to_3d_nochan(y):
    # Accept [B,T,N,1] or [B,T,N]; return [B,T,N]
    if y.dim() == 4 and y.size(-1) == 1:
        return y[..., 0]
    return y

def evaluate_masked(y_true, y_pred, mask_bool, use_masked_base: bool = True):
    """
    Reuse base metrics on a masked subset.
    If use_masked_base=True (default), uses MSE/RMSE/MAE/MAPE (which ignore zeros in y_true),
    matching your 'all pred steps' computation.
    If set to False, uses *_all variants.
    """
    y_true = _to_3d_nochan(y_true)
    y_pred = _to_3d_nochan(y_pred)

    mask = mask_bool.bool()
    finite = torch.isfinite(y_true) & torch.isfinite(y_pred)
    mask = mask & finite
    if mask.sum().item() == 0:
        return None

    # collapse to 1D
    yt = y_true[mask]
    yp = y_pred[mask]

    if use_masked_base:
        # match Metrics.evaluate() behavior
        return MSE(yt, yp), RMSE(yt, yp), MAE(yt, yp), MAPE(yt, yp)
    else:
        return MSE_all(yt, yp), RMSE_all(yt, yp), MAE_all(yt, yp), MAPE_all(yt, yp)


def closure_weighted_MAE(y_true, y_pred, cys=None, beta=2.0, eps=1e-6):
    """
    y_true, y_pred, cys: [B, T_out, N, 1]
    cys: time-valued (<=0 open/approach, >0 closed) or binary (0/1). If None -> plain MAE.
    beta: extra weight applied on closed steps (cys > 0).
    """
    if cys is None:
        return MAE(y_true, y_pred)

    # weights: 1 for open, 1+beta for closed
    is_closed = (cys > 0).to(y_true.dtype)
    w = 1.0 + beta * is_closed

    # normalize to keep average weight ~1 (so loss scale stays comparable)
    w = w / w.mean().clamp_min(eps)

    return (w * (y_pred - y_true).abs()).mean()

# Optional: a “soft” version that also upweights approaching closures (negative times)
def closure_soft_MAE(y_true, y_pred, cys=None, beta=2.0, tau=10.0, eps=1e-6):
    """
    - cys > 0 (closed): weight = 1 + beta
    - cys < 0 (approaching): weight rises smoothly as it gets closer (exp(cys/tau))
    - cys == 0: weight = 1
    """
    if cys is None:
        return MAE(y_true, y_pred)

    cys = cys.to(y_true.dtype)
    w = torch.ones_like(cys)

    closed = (cys > 0)
    approaching = (cys < 0)

    w[closed] = 1.0 + beta
    # closer to 0 (e.g., -2 minutes) → larger exp(-2/tau) than far (-30)
    w[approaching] = 1.0 + beta * torch.exp(cys[approaching] / tau)

    w = w / w.mean().clamp_min(eps)
    return (w * (y_pred - y_true).abs()).mean()


def effect_weighted_MAE(
    y_true, y_pred, cys,
    effect_sub,                 # [N_src, N_tgt] effect submatrix matching the split
    lambda_effect=1.0,
    use_soft=True,              # if True: timed labels; False: binary
    tau=10.0,                   # softness for timed labels
    beta=None,                  # scalar emphasis per epoch (e.g., from warm-up)
    combine_with_beta='mul',    # 'mul' or 'sum' (we’ll use 'mul' semantics below)
    w_min=1e-3,                 # optional caps to keep training stable
    w_max=10.0
):
    """
    y_true: [B, T_out, N_tgt, 1]
    y_pred: [B, T_out, N_tgt, 1]
    cys:    [B, T_out, N_src, 1]   # closure labels PER SOURCE sensor (same indexing as effect_sub rows)
    effect_sub: [N_src, N_tgt]     # from P.effect_slices['train'/'val_a'/...]
    """
    B, T, N_tgt, _ = y_true.shape
    _, _, N_src, _ = cys.shape

    # 1) Center around 1.0 per your construction (0=big drop, >1=reroute)
    #    We want magnitude-of-change from "normal".
    deltaE = torch.abs(effect_sub - 1.0)  # [N_src, N_tgt]

    # 2) Source activations from labels
    if use_soft:
        # cys = minutes-until-opening (nonnegative). Soften with exp(-t/tau).
        a = torch.exp(-torch.clamp(cys, min=0.0) / max(tau, 1e-8))   # [B, T, N_src, 1]
    else:
        # binary close/open
        a = (cys > 0).float()

    a = a.squeeze(-1)                       # [B, T, N_src]

    # 3) Influence per target: sum_i a_i * |E_ij - 1|
    #    (einsum: batch/time over i, times [i,j] -> [batch,time,j])
    # NOTE: deltaE must be on same device/dtype as y_true
    deltaE = deltaE.to(y_true.device, y_true.dtype)
    influence = torch.einsum('bti,ij->btj', a, deltaE)   # [B, T, N_tgt]

    # 4) Build weights
    #    w = 1 + lambda * influence  (then apply beta)
    base = 1.0 + lambda_effect * influence                # [B, T, N_tgt]

    if beta is not None:
        if combine_with_beta == 'mul':
            # Equivalent to w = 1 + (beta*lambda) * influence
            base = 1.0 + beta * (base - 1.0)
        elif combine_with_beta == 'sum':
            base = base + beta
        # else: ignore / already handled

    w = torch.clamp(base, min=w_min, max=w_max)           # [B, T, N_tgt]

    # 5) Weighted MAE over targets/timesteps/batch
    err = torch.abs(y_true - y_pred).squeeze(-1)          # [B, T, N_tgt]
    loss = (w * err).mean()
    return loss