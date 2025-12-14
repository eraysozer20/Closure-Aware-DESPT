import torch
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import unseen_nodes
from utils import *
from encoder import *
from augmentation import *
from model import *
import Metrics
from closures import (
    evaluate_closure_metrics,
)
import json
from pymongo import MongoClient, UpdateOne
import pickle

class StandardScaler:
    def __init__(self):
        self.u = None
        self.z = None
    def fit_transform(self, x):
        self.u = x.mean()
        self.z = x.std()
        return (x-self.u)/self.z
    def inverse_transform(self, x):
        return x * self.z + self.u

def connect_mongo():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    username = config['username']
    password = config['password']
    cluster_url = config['cluster_url']
    db_name = config['db_name']
    connection_string = f"mongodb+srv://{username}:{password}@{cluster_url}/test?retryWrites=true&w=majority"
    client = MongoClient(connection_string)
    db = client[db_name]
    return db

def save_parameters(P, param_obj, filename, mongodb):
    data = {attr: [getattr(param_obj, attr)] for attr in dir(param_obj) if not attr.startswith("__") and not callable(getattr(param_obj, attr))}
    column_order = ['exe_id'] + [col for col in data if col != 'exe_id']
    new_row = pd.DataFrame(data, columns=column_order)
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        df = pd.DataFrame(columns=column_order)
    df = pd.concat([df, new_row], ignore_index=True)
    df = df[column_order]
    df.to_csv(filename, index=False)
    if P.is_mongo:
        data_db = {attr: getattr(param_obj, attr) for attr in dir(param_obj) if not attr.startswith("__") and not callable(getattr(param_obj, attr)) 
                   and attr != 'exe_id'
                   and attr != 'replication'
                   and attr != 'track_id'
                   and attr != 'fold_i'}
        doc = {
                "exe_id": param_obj.exe_id,  # Ensure that param_obj has an exe_id att{attr: getattr(param_obj, attr) for attr in dir(param_obj) if not attr.startswith("__") and not callable(getattr(param_obj, attr)) and attr != 'exe_id'}ribute
                "track_id": param_obj.track_id,
                "replication": param_obj.replication,
                "fold_i": param_obj.fold_i,
                "P": data_db
            }

        mongodb['performance'].insert_one(doc)

def getModel(P, name, device, support_len):
    if name == 'gwnet':
        model = gwnet(device, num_nodes=P.n_sensor, in_dim=P.n_channel, adp_adj=P.gwnet_is_adp_adj, sga=P.gwnet_is_SGA, support_len=support_len, is_concat=P.is_concat_encoder_model, is_layer_after_concat=P.is_layer_after_concat).to(device)
        # if P.gwnet_is_adp_adj == False:
        #     model = nn.DataParallel(model)
    elif name == 'LSTM':
        if P.is_pretrain == False:
            lstm_input_dim = 32
        elif P.is_layer_after_concat:
            lstm_input_dim = 32
        elif P.is_concat_encoder_model:
            lstm_input_dim = 64
        else:
            lstm_input_dim = 32
        model = LSTM_uni(input_dim=P.n_channel, lstm_input_dim=lstm_input_dim, hidden_dim=P.lstm_hidden_dim, output_dim=12, layer_dim = P.lstm_layers, dropout_prob = P.lstm_dropout, device=device, is_GCN_after_CL = P.is_GCN_after_CL, support_len = support_len, gcn_order=P.gcn_order, gcn_dropout=P.gcn_dropout).to(device)
        model = nn.DataParallel(model)
    return model

def getXSYS(P, data, mode):
    TRAIN_NUM = int(data.shape[0] * (P.t_train + P.t_val))
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - P.timestep_out - P.timestep_in + 1):
            x = data[i:i+P.timestep_in, :]
            y = data[i+P.timestep_in:i+P.timestep_in+P.timestep_out, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - P.timestep_in,  data.shape[0] - P.timestep_out - P.timestep_in + 1):
            x = data[i:i+P.timestep_in, :]
            y = data[i+P.timestep_in:i+P.timestep_in+P.timestep_out, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    XS = XS.transpose(0, 3, 2, 1)
    return XS, YS

def setups(P, device):
    '''
    If the save folder does not exist, create it.
    '''
    if not os.path.exists(P.save_path):
        os.makedirs(P.save_path)
    torch.manual_seed(P.seed)
    torch.cuda.manual_seed(P.seed)
    
    '''
    Split the data in temporal dimension into training and testing. 
    At this step, training set contains both training and validation data.
    So that each each dimension correspond to [instance, input, sensor, output]
    '''
    trainXS, trainYS = getXSYS(P, data, 'TRAIN')
    testXS, testYS = getXSYS(P, data, 'TEST')

    print('\ntrainXS.shape', trainXS.shape)
    print('trainYS.shape', trainYS.shape)
    print('testXS.shape', testXS.shape)
    print('testYS.shape', testYS.shape)


    if getattr(P, 'has_closure_labels', False):
        # closure_mask: [T_total, N_sensor] (already loaded in main)
        closure_mask = P.closure_mask

        # Same temporal split as getXSYS: train + val part
        TRAIN_NUM = int(closure_mask.shape[0] * (P.t_train + P.t_val))
        train_clos = closure_mask[:TRAIN_NUM, :]          # [T_trainval, N]

        # has_train_closure[j] = True if sensor j ever closed in train+val
        has_train_closure = (train_clos > 0).any(axis=0)  # numpy bool array [N]

        # unseen_closure_flag[j] = 1 if sensor j NEVER closed in train+val
        P.unseen_closure_flag = (~has_train_closure).astype(np.int8)

        print("\n[Unseen-closure flag]")
        print("  sensors with NO closure in train+val:",
                int(P.unseen_closure_flag.sum()), "/", P.n_sensor)

    if P.example_verbose:
        print('\nFirst instance trainXS for first sensor:', trainXS[0,0,0,:])
        print('\nFirst instance trainYS for first sensor:', trainYS[0,:,0,0])
        print('\nLast instance trainXS for first sensor:', trainXS[-1,0,0,:])
        print('\nFirst instance testXS for first sensor:', testXS[0,0,0,:])
        print('\nFirst instance trainXS for second sensor:', trainXS[0,0,1,:])
        print('\nFirst instance trainYS for second sensor:', trainYS[0,:,1,0])


    if P.has_closure_labels:
        trainCXS, trainCYS = getXSYS(P, P.closure_mask, 'TRAIN')
        testCXS,  testCYS  = getXSYS(P, P.closure_mask, 'TEST')
        print('\ntrainCXS.shape', trainCXS.shape)
        print('trainCYS.shape', trainCYS.shape)
        print('testCXS.shape',  testCXS.shape)
        print('testCYS.shape',  testCYS.shape)
    else:
        trainCXS = trainCYS = testCXS = testCYS = None

    '''
    Split the training set further into training and validation sets.
    '''
    P.trainval_size = len(trainXS)
    P.train_size = int(P.trainval_size * (P.t_train / (P.t_train + P.t_val)))
    XS_torch_trn = trainXS[:P.train_size,:,:,:]
    YS_torch_trn = trainYS[:P.train_size,:,:,:]
    XS_torch_val = trainXS[P.train_size:P.trainval_size,:,:,:]
    YS_torch_val = trainYS[P.train_size:P.trainval_size,:,:,:]

    '''
    Get the sensor indexes for each spatial splited set.
    '''
    spatialSplit_unseen = unseen_nodes.SpatialSplit(data.shape[1], P.fold_i, r_trn=P.s_train, r_val=P.s_val, r_tst=(1-P.s_train-P.s_val), seed=P.seed)
    spatialSplit_allNod = unseen_nodes.SpatialSplit(data.shape[1], P.fold_i, r_trn=P.s_train, r_val=min(1.0,P.s_val+P.s_train), r_tst=1.0, seed=P.seed)
    print('\nspatialSplit_unseen', spatialSplit_unseen)
    print('spatialSplit_allNod', spatialSplit_allNod)
    print('spatialSplit_unseen.i_tst', spatialSplit_unseen.i_tst)
    # save the spatial split
    with open(P.save_path + '/' + 'i_tst.txt', 'w') as f:
        np.savetxt(f, spatialSplit_unseen.i_tst, fmt='%d')
    with open(P.save_path + '/' + 'i_val.txt', 'w') as f:
        np.savetxt(f, spatialSplit_unseen.i_val, fmt='%d')
    with open(P.save_path + '/' + 'i_trn.txt', 'w') as f:
        np.savetxt(f, spatialSplit_unseen.i_trn, fmt='%d')
    if P.example_verbose:
        print('\nspatialSplit_unseen.i_trn', spatialSplit_unseen.i_trn)
        print('spatialSplit_unseen.i_val', spatialSplit_unseen.i_val)
        print('spatialSplit_unseen.i_tst', spatialSplit_unseen.i_tst)
        print('spatialSplit_allNod.i_trn', spatialSplit_allNod.i_trn)
        print('spatialSplit_allNod.i_val', spatialSplit_allNod.i_val)
        print('spatialSplit_allNod.i_tst', spatialSplit_allNod.i_tst)

    if getattr(P, 'has_closure_labels', False) and hasattr(P, 'unseen_closure_flag'):
        flag_full = P.unseen_closure_flag  # [N_sensor] in original order

        # same ordering as adj / XS_torch_* built with spatialSplit_allNod
        P.unseen_closure_flag_trn = flag_full[spatialSplit_allNod.i_trn]
        P.unseen_closure_flag_val = flag_full[spatialSplit_allNod.i_val]
        P.unseen_closure_flag_tst = flag_full[spatialSplit_allNod.i_tst]

        print("\n[Unseen-closure flag per split]")
        print("  train:", int(P.unseen_closure_flag_trn.sum()),
                " val:",  int(P.unseen_closure_flag_val.sum()),
                " test:", int(P.unseen_closure_flag_tst.sum()))

    P.effect_slices = None
    try:
        if getattr(P, 'has_closure_labels', False) and hasattr(P, 'effect_matrix_path'):
            with open(P.effect_matrix_path, 'rb') as fh:
                eff_obj = pickle.load(fh)
            if isinstance(eff_obj, dict) and 'matrix' in eff_obj:
                E_full = eff_obj['matrix']
            else:
                E_full = eff_obj
            E_full = torch.tensor(E_full).float()

            def _submat(rows, cols):
                return E_full[torch.as_tensor(rows, dtype=torch.long)][:, torch.as_tensor(cols, dtype=torch.long)]

            i_trn = spatialSplit_allNod.i_trn
            i_val_u = spatialSplit_unseen.i_val
            i_val_a = spatialSplit_allNod.i_val
            i_tst_u = spatialSplit_unseen.i_tst
            i_tst_a = spatialSplit_allNod.i_tst

            P.effect_slices = {
                'train':  _submat(i_trn,  i_trn),
                'val_u':  _submat(i_val_u, i_val_u),
                'val_a':  _submat(i_val_a, i_val_a),
                'tst_u':  _submat(i_tst_u, i_tst_u),
                'tst_a':  _submat(i_tst_a, i_tst_a),
            }

            # ---- Debug prints ----
            print("\n[Effect matrix slicing results:]")
            for name, M in P.effect_slices.items():
                print(f"  {name:<7} shape={tuple(M.shape)}  mean={M.mean():.4f}  std={M.std():.4f}")

    except Exception as e:
        print("[effect] Could not load/slice effect matrix:", e)
        P.effect_slices = None

    XS_torch_train = torch.Tensor(XS_torch_trn[:,:,spatialSplit_allNod.i_trn,:])
    YS_torch_train = torch.Tensor(YS_torch_trn[:,:,spatialSplit_allNod.i_trn,:])
    XS_torch_train_model = torch.Tensor(XS_torch_val[:,:,spatialSplit_allNod.i_trn,:])
    YS_torch_train_model = torch.Tensor(YS_torch_val[:,:,spatialSplit_allNod.i_trn,:])
    XS_torch_val_u = torch.Tensor(XS_torch_val[:,:,spatialSplit_unseen.i_val,:])
    YS_torch_val_u = torch.Tensor(YS_torch_val[:,:,spatialSplit_unseen.i_val,:])
    XS_torch_val_a = torch.Tensor(XS_torch_val[:,:,spatialSplit_allNod.i_val,:])
    YS_torch_val_a = torch.Tensor(YS_torch_val[:,:,spatialSplit_allNod.i_val,:])
    XS_torch_tst_u = torch.Tensor(testXS[:,:,spatialSplit_unseen.i_tst,:])
    YS_torch_tst_u = torch.Tensor(testYS[:,:,spatialSplit_unseen.i_tst,:])
    XS_torch_tst_a = torch.Tensor(testXS[:,:,spatialSplit_allNod.i_tst,:])
    YS_torch_tst_a = torch.Tensor(testYS[:,:,spatialSplit_allNod.i_tst,:])
    print('\ntrain.shape', XS_torch_train.shape, YS_torch_train.shape)
    print('train_model.shape', XS_torch_train_model.shape, YS_torch_train_model.shape)
    print('val_u.shape', XS_torch_val_u.shape, YS_torch_val_u.shape)
    print('val_a.shape', XS_torch_val_a.shape, YS_torch_val_a.shape)
    print('tst_u.shape', XS_torch_tst_u.shape, YS_torch_tst_u.shape)
    print('tst_a.shape', XS_torch_tst_a.shape, YS_torch_tst_a.shape)
    if P.example_verbose:
        print('\nFor veryfication purposes')
        print('Corresponding first sensor in org order in test', XS_torch_tst_u[0,0,1,:])
        print('Corresponding second sensor in org order in train', XS_torch_train[0,0,1,:])
    


    
    # ---- Closure labels for ALL splits (train/val/test) ----
    CXS_trn = CYS_trn = CXS_val_u = CYS_val_u = CXS_val_a = CYS_val_a = None
    CXS_tst_u = CYS_tst_u = CXS_tst_a = CYS_tst_a = None
    CYS_torch_train = CYS_torch_train_model = None
    CYS_torch_val_u = CYS_torch_val_a = None
    CYS_torch_tst_u = CYS_torch_tst_a = None


    if P.has_closure_labels:
        # 1) Windowize closures exactly like data
        trainCXS, trainCYS = getXSYS(P, P.closure_mask, 'TRAIN')  # shapes ~ [B_trainval, 1, N, T_in] / [B_trainval, T_out, N, 1]
        testCXS,  testCYS  = getXSYS(P, P.closure_mask, 'TEST')

        # 2) Spatial slicing (mirror node splits)
        # TRAIN/VAL (time) on ALL-NODES TRAIN split
        CXS_trn = trainCXS[:, :, spatialSplit_allNod.i_trn, :]     # [B_trainval, 1, N_tr, T_in]
        CYS_trn = trainCYS[:, :, spatialSplit_allNod.i_trn, :]     # [B_trainval, T_out, N_tr, 1]

        # VAL_U / VAL_A (these mirror your XS_torch_val_* node sets)
        CXS_val_u = trainCXS[:, :, spatialSplit_unseen.i_val, :]   # [B_trainval, 1, N_val_u, T_in]
        CYS_val_u = trainCYS[:, :, spatialSplit_unseen.i_val, :]
        CXS_val_a = trainCXS[:, :, spatialSplit_allNod.i_val, :]   # [B_trainval, 1, N_val_a, T_in]
        CYS_val_a = trainCYS[:, :, spatialSplit_allNod.i_val, :]

        # TEST splits
        if testCXS is not None:
            CXS_tst_u = testCXS[:, :, spatialSplit_unseen.i_tst, :]  # [B_test, 1, N_tst_u, T_in]
            CYS_tst_u = testCYS[:, :, spatialSplit_unseen.i_tst, :]
            CXS_tst_a = testCXS[:, :, spatialSplit_allNod.i_tst, :]  # [B_test, 1, N_tst_a, T_in]
            CYS_tst_a = testCYS[:, :, spatialSplit_allNod.i_tst, :]
            if P.example_verbose:
                print('\nCXS/CYS TEST split shapes:')
                print('  CXS_tst_u:', CXS_tst_u.shape, '  CYS_tst_u:', CYS_tst_u.shape)
                print('  CXS_tst_a:', CXS_tst_a.shape, '  CYS_tst_a:', CYS_tst_a.shape)

        # 3) (IMPORTANT) Temporal sub-splits to align with XS_torch_* tensors
        # XS_torch_train       corresponds to  trainXS[:P.train_size,  ..., i_trn, :]
        # XS_torch_train_model corresponds to  trainXS[P.train_size:..., i_trn, :]   (your "B" split in time)
        # XS_torch_val_*       corresponds to  trainXS[P.train_size:..., i_val/val_a, :]
        # So we mirror this for closures:

        # a) For TRAIN nodes (i_trn):
        CXS_torch_trn        = torch.tensor(CXS_trn[:P.train_size, ...])                     # align XS_torch_train
        CXS_torch_trn_model  = torch.tensor(CXS_trn[P.train_size:P.trainval_size, ...])      # align XS_torch_train_model

        # b) For VAL_U / VAL_A (node sets): use the same temporal cut as XS_torch_val_*
        CXS_torch_val_u      = torch.tensor(CXS_val_u[P.train_size:P.trainval_size, ...])
        CXS_torch_val_a      = torch.tensor(CXS_val_a[P.train_size:P.trainval_size, ...])

        # c) TEST splits are already built on TEST windows:
        CXS_torch_tst_u      = torch.tensor(CXS_tst_u) if CXS_tst_u is not None else None
        CXS_torch_tst_a      = torch.tensor(CXS_tst_a) if CXS_tst_a is not None else None

        # 4) Keep TIME labels as-is. DO NOT binarize if closure_mode == 'time'.
        # If you still want binary labels for metrics, create *_bin copies without overwriting originals:
        want_bin_for_metrics = getattr(P, 'closure_bin_for_metrics', False)
        if want_bin_for_metrics:
            def _mk_bin(arr): return None if arr is None else (arr > 0).astype(np.float32)
            CYS_trn_bin   = _mk_bin(CYS_trn);   CYS_val_u_bin = _mk_bin(CYS_val_u)
            CYS_val_a_bin = _mk_bin(CYS_val_a); CYS_tst_u_bin = _mk_bin(CYS_tst_u)
            CYS_tst_a_bin = _mk_bin(CYS_tst_a)
            # store them on P if you want access later:
            P.CYS_bin = {
                'trn': CYS_trn_bin, 'val_u': CYS_val_u_bin, 'val_a': CYS_val_a_bin,
                'tst_u': CYS_tst_u_bin, 'tst_a': CYS_tst_a_bin
            }

        # TRAIN nodes (i_trn) with temporal train slice
        CYS_torch_train       = torch.tensor(CYS_trn[:P.train_size, ...])                 # matches YS_torch_train
        # TRAIN_MODEL = temporal val slice over TRAIN nodes
        CYS_torch_train_model = torch.tensor(CYS_trn[P.train_size:P.trainval_size, ...])  # matches YS_torch_train_model

        # VAL splits (temporal val slice)
        CYS_torch_val_u = torch.tensor(CYS_val_u[P.train_size:P.trainval_size, ...])      # matches YS_torch_val_u
        CYS_torch_val_a = torch.tensor(CYS_val_a[P.train_size:P.trainval_size, ...])      # matches YS_torch_val_a

        # TEST splits
        CYS_torch_tst_u = torch.tensor(CYS_tst_u) if CYS_tst_u is not None else None      # matches YS_torch_tst_u
        CYS_torch_tst_a = torch.tensor(CYS_tst_a) if CYS_tst_a is not None else None      # matches YS_torch_tst_a

        if P.example_verbose:
            print('\n[Closure labels / all splits] (time labels preserved)')
            for name, obj in [
                ('CXS_trn', CXS_trn), ('CYS_trn', CYS_trn),
                ('CXS_val_u', CXS_val_u), ('CYS_val_u', CYS_val_u),
                ('CXS_val_a', CXS_val_a), ('CYS_val_a', CYS_val_a),
                ('CXS_tst_u', CXS_tst_u), ('CYS_tst_u', CYS_tst_u),
                ('CXS_tst_a', CXS_tst_a), ('CYS_tst_a', CYS_tst_a),
            ]:
                if obj is not None: print(f'  {name}: {obj.shape}')
            print('[Closure labels ready]')
    # ---------------------------------------

    # ---- Concatenate closure channel to all XS_* tensors (aligning temporal and spatial splits) ----

    if P.has_closure_labels and P.add_closure_labels:
        print("\n[Adding closure channel to inputs]")

        # TRAIN (temporal train slice, train nodes)
        XS_torch_train       = torch.cat([XS_torch_train,       CXS_torch_trn.to(XS_torch_train.dtype)], dim=1)

        # TRAIN_MODEL (temporal val slice over TRAIN nodes)
        XS_torch_train_model = torch.cat([XS_torch_train_model, CXS_torch_trn_model.to(XS_torch_train_model.dtype)], dim=1)

        # VAL splits (temporal val slice)
        XS_torch_val_u       = torch.cat([XS_torch_val_u,       CXS_torch_val_u.to(XS_torch_val_u.dtype)], dim=1)
        XS_torch_val_a       = torch.cat([XS_torch_val_a,       CXS_torch_val_a.to(XS_torch_val_a.dtype)], dim=1)

        # TEST splits
        if XS_torch_tst_u is not None and CXS_torch_tst_u is not None:
            XS_torch_tst_u   = torch.cat([XS_torch_tst_u,       CXS_torch_tst_u.to(XS_torch_tst_u.dtype)], dim=1)
        if XS_torch_tst_a is not None and CXS_torch_tst_a is not None:
            XS_torch_tst_a   = torch.cat([XS_torch_tst_a,       CXS_torch_tst_a.to(XS_torch_tst_a.dtype)], dim=1)

        # Increase input channel count
        P.n_channel += 1
        print(f"Input channels increased to {P.n_channel}")

    # Helper so datasets become (X,Y,CYS) when closures exist, else (X,Y)
    def _make_ds(X, Y, C):
        return torch.utils.data.TensorDataset(X, Y, C) if C is not None \
            else torch.utils.data.TensorDataset(X, Y)

    # ⬇️ Build datasets AFTER you've concatenated the closure X-channel and bumped P.n_channel
    train_data        = _make_ds(XS_torch_train,       YS_torch_train,       CYS_torch_train)
    train_model_data  = _make_ds(XS_torch_train_model, YS_torch_train_model, CYS_torch_train_model)
    val_u_data        = _make_ds(XS_torch_val_u,       YS_torch_val_u,       CYS_torch_val_u)
    val_a_data        = _make_ds(XS_torch_val_a,       YS_torch_val_a,       CYS_torch_val_a)
    tst_u_data        = _make_ds(XS_torch_tst_u,       YS_torch_tst_u,       CYS_torch_tst_u)
    tst_a_data        = _make_ds(XS_torch_tst_a,       YS_torch_tst_a,       CYS_torch_tst_a)

    # DataLoaders
    train_iter        = torch.utils.data.DataLoader(train_data,        P.batch_size, shuffle=False)
    train_model_iter  = torch.utils.data.DataLoader(train_model_data,  P.batch_size, shuffle=False)
    val_u_iter        = torch.utils.data.DataLoader(val_u_data,        P.batch_size, shuffle=False)
    val_a_iter        = torch.utils.data.DataLoader(val_a_data,        P.batch_size, shuffle=False)
    tst_u_iter        = torch.utils.data.DataLoader(tst_u_data,        P.batch_size, shuffle=False)
    tst_a_iter        = torch.utils.data.DataLoader(tst_a_data,        P.batch_size, shuffle=False)




    # Load the adjacency matrix
    adj_mx = load_adj(P.adj_path, P.adj_type, P.dataname, P.adj_diag)
    if P.example_verbose:
        print('\nadjacency matrix after (or not) normalization')
        print('frist row of the first adjacency matrix', adj_mx[0][0])
        print('Entry (18,1):', adj_mx[0][18][1])
        print('Entry (7,11):', adj_mx[0][7][11])
        print('Entry (19,10):', adj_mx[0][19][10])

    # Split the adjacency matrix by spatial and temporal splits
    adj_train = [torch.tensor(i[spatialSplit_unseen.i_trn,:][:,spatialSplit_unseen.i_trn]).to(device) for i in adj_mx]
    adj_val_u = [torch.tensor(i[spatialSplit_unseen.i_val,:][:,spatialSplit_unseen.i_val]).to(device) for i in adj_mx]
    adj_val_a = [torch.tensor(i[spatialSplit_allNod.i_val,:][:,spatialSplit_allNod.i_val]).to(device) for i in adj_mx]
    adj_tst_u = [torch.tensor(i[spatialSplit_unseen.i_tst,:][:,spatialSplit_unseen.i_tst]).to(device) for i in adj_mx]
    adj_tst_a = [torch.tensor(i[spatialSplit_allNod.i_tst,:][:,spatialSplit_allNod.i_tst]).to(device) for i in adj_mx]
    print('\nadj_train', 'length of', len(adj_train), adj_train[0].shape)
    print('adj_val_u', 'length of', len(adj_val_u), adj_val_u[0].shape)
    print('adj_val_a', 'length of', len(adj_val_a), adj_val_a[0].shape)
    print('adj_tst_u', 'length of', len(adj_tst_u), adj_tst_u[0].shape)
    print('adj_tst_a', 'length of', len(adj_tst_a), adj_tst_a[0].shape)
    
    if P.example_verbose:
        print('\nFor veryfication purposes')
        #print('Corresponding Entry (18,1)', adj_train[0][0][1])
        #print('Corresponding Entry (7,11)', adj_val_u[0][1][3])
        #print('Corresponding Entry (19,10)', adj_train[0][2][4])
        print('\nadjacency matrix after normalization')
        print('train adj', adj_train[0])
        print('val_u adj', adj_val_u[0])
        print('val_a adj', adj_val_a[0])
        print('tst_u adj', adj_tst_u[0])
        print('tst_a adj', adj_tst_a[0])
    
    '''
    PRETRAIN data loader.
    No need to wrap 12 timestamps into a single instance.
    Therefore, destruct them to one dimension.
    Both of them should have dimension [number_of_sensor, timestamp]
    '''
    pretrn_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            XS_torch_train[:, 0, :, 0].T  # [N_tr, B_tr]
        ),
        batch_size=1,
        shuffle=True
    )

    preval_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(
                XS_torch_trn[:, 0, spatialSplit_allNod.i_val, 0]  # [B_tr, N_val]
            ).T.float()  # [N_val, B_tr]
        ),
        batch_size=1,
        shuffle=False
    )

    print('\npretrn_iter.dataset.tensors[0].shape', pretrn_iter.dataset.tensors[0].shape)
    print('preval_iter.dataset.tensors[0].shape', preval_iter.dataset.tensors[0].shape)
    if P.example_verbose:
        print('\nFor veryfication purposes')
        a = pretrn_iter.dataset.tensors
        b = preval_iter.dataset.tensors
        print('Corresponding second instance pretrn_iter', a[0][1,:])
        print('Corresponding second instance preval_iter', b[0][1,:])

    return pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod, \
        train_iter, train_model_iter, val_u_iter, val_a_iter, tst_u_iter, tst_a_iter, \
        adj_train, adj_val_u, adj_val_a, adj_tst_u, adj_tst_a, \
        CXS_tst_u, CYS_tst_u, CXS_tst_a, CYS_tst_a, \
        CXS_trn, CYS_trn, CXS_val_u, CYS_val_u, CXS_val_a, CYS_val_a


def pretrainModel(P, name, pretrain_iter, preval_iter, adj_train, adj_val, device, spatialSplit_allNod, mongodb):
    print('\npretrainModel Started ...')
    adj_train = [tensor.to(device) for tensor in adj_train]
    adj_val = [tensor.to(device) for tensor in adj_val]
    if P.example_verbose:
        print('\nFor veryfication purposes')
        print('adj_train for second sensor in original order', adj_train[0][1])
        print('adj_val for second sensor in original order', adj_val[0][1])
        print('there are ', len(adj_train), 'adjacency matrices in adj_train')
    if P.augmentation == 'sampler':
        is_sampler = True
    else:
        is_sampler = False
    # Get the encoder model
    if P.pre_model == 'COST':
        model = CoSTEncoder(1, 32, P.cost_kernals, 201, 64, 10, P.cost_alpha, P.cl_temperature, P.is_GCN_encoder, is_sampler, len(adj_train), P.gcn_order, P.gcn_dropout).to(device)
        model = torch.nn.DataParallel(model).to(device)
    elif P.pre_model == 'TCN':
        model = Contrastive_FeatureExtractor_conv(P.cl_temperature, P.is_GCN_encoder, is_sampler, len(adj_train), P.gcn_order, P.gcn_dropout).to(device)
        model = torch.nn.DataParallel(model).to(device)
    # Start pretraining
    min_val_loss = np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=P.learn_rate, weight_decay=P.weight_decay)
    s_time = datetime.now()
    encoder_log = []
    for epoch in range(P.pretrain_epoch):
        starttime = datetime.now()
        model.train()
        x = pretrain_iter.dataset.tensors
        if P.example_verbose:
            print('\nFor pretraining, training dataset:')
            print('x[0].shape', x[0].shape)
            print('x[0] for second sensor in original order', x[0][1,:])
        # Get the loss
        if P.augmentation == 'edge_masking':
            loss = model.module.contrast(x[0].to(device), x[0].to(device), edge_masking(adj_train, 0.02, device), edge_masking(adj_train, 0.02, device), 0, P.example_verbose)
        elif P.augmentation == 'sampler':
            loss = model.module.contrast(x[0].to(device), x[0].to(device), adj_train, adj_train, 0, P.example_verbose)
        elif P.augmentation == 'temporal_shifting':
            x1 = temporal_shifting(x[0], P.temporal_shifting_r).to(device)
            x2 = temporal_shifting(x[0], P.temporal_shifting_r).to(device)
            if P.example_verbose:
                print('\nthe first augmentated input', x1[1,:])
                print('the second augmentated input', x2[1,:])
            loss = model.module.contrast(x1,x2, adj_train, adj_train, 0, P.example_verbose)
        elif P.augmentation == 'temporal_shifting_new':
            x1 = temporal_shifting_new(x[0], P.temporal_shifting_r).to(device)
            x2 = temporal_shifting_new(x[0], P.temporal_shifting_r).to(device)
            if P.example_verbose:
                print('\nthe first augmentated input, for fixed temporal shifting', x1[1,:])
                print('the second augmentated input, for fixed temporal shifting', x2[1,:])
            loss = model.module.contrast(x1,x2, adj_train, adj_train, 0, P.example_verbose)
        elif P.augmentation == 'input_smoothing':
            x1 = input_smoothing(x[0], P.input_smoothing_r, P.input_smoothing_e).to(device)
            x2 = input_smoothing(x[0], P.input_smoothing_r, P.input_smoothing_e).to(device)
            if P.example_verbose:
                print('\nAugmented input for input smoothing')
            loss = model.module.contrast(x1,x2, adj_train, adj_train, 0, P.example_verbose)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss

        # Set the start index for the validation where the loss will be calculated only for the sensors after this index
        if P.is_testunseen:
            sensor_idx_start = len(spatialSplit_allNod.i_trn)
        else:
            sensor_idx_start = 0
        if P.example_verbose:
            print('\nThe validation for encoder starts at index', sensor_idx_start)

        # Validation
        val_loss = pre_evaluateModel(P, model, preval_iter, adj_val, sensor_idx_start, device)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), P.save_path + '/' + name + '.pt')

        # Time
        endtime = datetime.now()
        epoch_time = (endtime - starttime).total_seconds()

        # save epoch results
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(P.save_path + '/' + name + '_log.txt', 'a') as f:
            f.write("%s %d, %s %d %s, %s %.10f, %s %.10f\n" % ("epoch:", epoch, "time used:", epoch_time, "seconds", "train loss:", train_loss, "validation loss:", val_loss))
        if P.is_mongo:
            train_loss_float = float(f"{train_loss:.10f}")
            val_loss_float = float(f"{val_loss:.10f}")
            doc = {
                "exe_id": P.exe_id,
                'track_id': P.track_id,
                'replication': P.replication,
                "epoch": epoch,
                "time_used": epoch_time,
                "train_loss": train_loss_float,
                "val_loss": val_loss_float
            }
            encoder_log.append(doc)
    e_time = datetime.now()
    print('PRETIME DURATION:', (e_time-s_time).total_seconds())

    # Write the final results to the log file
    try:
        df = pd.read_csv('save/results.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['exe_id', 'min_pretrain_val_loss', 'pretrain_time'])
    row_exists = 'exe_id' in df.columns and any(df['exe_id'] == P.exe_id)
    if row_exists:
        df.loc[df['exe_id'] == P.exe_id, ['min_pretrain_val_loss', 'pretrain_time']] = [min_val_loss, (e_time-s_time).total_seconds()]
    else:
        new_row = pd.DataFrame({'exe_id': [P.exe_id], 'min_pretrain_val_loss': [min_val_loss.cpu().numpy()], 'pretrain_time': [(e_time-s_time).total_seconds()]})
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv('save/results.csv', index=False)
    
    if P.is_mongo:
        doc = {
            "exe_id": P.exe_id,
            "min_pretrain_val_loss": float(f"{min_val_loss:.10f}"),
            "pretrain_time": (e_time-s_time).total_seconds(),
        }
        update_operation = UpdateOne(
            {"exe_id": doc["exe_id"]},  # Search filter
            {"$set": {
                "min_pretrain_val_loss": doc["min_pretrain_val_loss"],
                "pretrain_time": doc["pretrain_time"]
            }},
            upsert=True  # Insert the document if it does not exist
        )
        mongodb['performance'].bulk_write([update_operation])

    print('pretrainModel Ended ...\n')

    return encoder_log

def pre_evaluateModel(P, model, data_iter, adj, sensor_idx_start, device):
    '''
    Calculate the loss for the validation set
    Execute the encoder for the full length of the data
    But only the sensors after sensor_idx_start are used for the loss calculation
    '''
    model.eval()
    with torch.no_grad():
        x = data_iter.dataset.tensors
        if P.augmentation == 'edge_masking':
            l = model.module.contrast(x[0].to(device), x[0].to(device), edge_masking(adj, 0.02, device), edge_masking(adj, 0.02, device), sensor_idx_start, P.example_verbose)
        elif P.augmentation == 'sampler':
            l = model.module.contrast(x[0].to(device), x[0].to(device), adj, adj, sensor_idx_start, P.example_verbose)
        elif P.augmentation == 'temporal_shifting':
            l = model.module.contrast(temporal_shifting(x[0], P.temporal_shifting_r).to(device),temporal_shifting(x[0], P.temporal_shifting_r).to(device), adj, adj, sensor_idx_start, P.example_verbose)
        elif P.augmentation == 'temporal_shifting_new':
            l = model.module.contrast(temporal_shifting_new(x[0], P.temporal_shifting_r).to(device),temporal_shifting_new(x[0], P.temporal_shifting_r).to(device), adj, adj, sensor_idx_start, P.example_verbose)
        elif P.augmentation == 'input_smoothing':
            l = model.module.contrast(input_smoothing(x[0], P.input_smoothing_r, P.input_smoothing_e).to(device),input_smoothing(x[0], P.input_smoothing_r, P.input_smoothing_e).to(device), adj, adj, sensor_idx_start, P.example_verbose)
        return l

def trainModel(P, name, mode, 
        train_iter, train_model_iter, val_u_iter, val_a_iter,
        adj_train, adj_val_u, adj_val_a,
        spatialSplit_unseen, spatialSplit_allNod, device_cpu, device_gpu, mongodb):
    print('\ntrainModel Started ...')

    # Set the device for the encoder
    if P.train_encoder_on == 'cpu':
        device_encoder = device_cpu
    else:
        device_encoder = device_gpu

    if P.augmentation == 'sampler':
        is_sampler = True
    else:
        is_sampler = False

    model = getModel(P, name, device_gpu, len(adj_train)) # Prediction model

    # ===== Training settings (with closure-loss warm-up) =====
    min_val_u_loss = np.inf
    min_val_a_loss = np.inf
    final_train_loss = np.inf
    tolerance = 0

    # ---- Hyperparameters (read from P with safe defaults)
    use_closure_loss   = getattr(P, "use_closure_loss", False)           # turn closure-weighted loss on/off
    use_soft           = getattr(P, "closure_weight_soft", False)        # soft weighting using time-to-closure
    tau                = getattr(P, "closure_tau", 10.0)                 # softness for soft weighting
    beta_start         = getattr(P, "closure_weight_beta_start", 1.0)    # warm-up start
    beta_end           = getattr(P, "closure_weight_beta_end",   3.0)    # warm-up end
    warmup_epochs      = getattr(P, "closure_warmup_epochs",    10)      # epochs to reach beta_end
    beta_schedule      = getattr(P, "closure_beta_schedule",  "linear")  # 'linear' | 'exp' | 'step'
    beta_step_epoch    = getattr(P, "closure_beta_step_epoch",  5)       # if 'step', when to jump to beta_end
    val_use_closure    = getattr(P, "val_use_closure_loss", False)       # use closure loss for val or plain MAE
    lambda_effect = getattr(P, 'lambda_effect', 1.0)
    combine_with_beta = getattr(P, 'combine_with_beta', 'mul')  # 'mul' | 'sum' | 'max'
    effect_train = None
    effect_val_a = None
    if getattr(P, 'use_closure_loss', False) and getattr(P, 'effect_slices', None) is not None:
        # put on the GPU that runs the predictor
        effect_train = P.effect_slices.get('train')
        effect_val_a = P.effect_slices.get('val_a')
        if effect_train is not None: effect_train = effect_train.to(device_gpu)
        if effect_val_a is not None: effect_val_a = effect_val_a.to(device_gpu)
    
    # Helper: compute beta for the current epoch (0-based)
    def compute_beta(e):
        if warmup_epochs <= 0:
            return beta_end
        if beta_schedule == "linear":
            t = min(1.0, e / float(warmup_epochs))
            return beta_start + t * (beta_end - beta_start)
        elif beta_schedule == "exp":
            # smooth exponential ramp-up
            t = min(1.0, e / float(warmup_epochs))
            return beta_start * ((beta_end / max(1e-8, beta_start)) ** t)
        elif beta_schedule == "step":
            return beta_start if e < beta_step_epoch else beta_end
        else:
            return beta_end  # fallback
        
    def make_criterion(effect_sub, use_for_val: bool):
        """
        Returns a callable(y_true,y_pred,cys) that applies:
          - effect-weighted MAE if enabled and tensors provided
          - else your existing closure_weighted_MAE
          - else plain MAE
        """
        def _crit(y_true, y_pred, cys):
            if (not use_closure_loss) or (cys is None):
                return Metrics.MAE(y_true, y_pred)

            b = compute_beta(epoch)  # tie to current epoch
            # Prefer effect-weighted when we have a submatrix
            if effect_sub is not None:
                return Metrics.effect_weighted_MAE(
                    y_true, y_pred, cys,
                    effect_sub=effect_sub,
                    lambda_effect=lambda_effect,
                    use_soft=use_soft,
                    tau=tau,
                    beta=b if combine_with_beta else None,
                    combine_with_beta=combine_with_beta,
                )
            # Fallback to your earlier closure-only weighting
            if use_soft:
                return Metrics.closure_soft_MAE(y_true, y_pred, cys, beta=b, tau=tau)
            else:
                return Metrics.closure_weighted_MAE(y_true, y_pred, cys, beta=b)
        return _crit

    # Optimizer (unchanged)
    optimizer = torch.optim.Adam(model.parameters(), lr=P.learn_rate, weight_decay=P.weight_decay)
    s_time = datetime.now()

    # Adjacency Matrix used for encoder
    adj_train = [tensor.to(device_encoder) for tensor in adj_train]
    adj_val_u = [tensor.to(device_encoder) for tensor in adj_val_u]
    adj_val_a = [tensor.to(device_encoder) for tensor in adj_val_a]
    if P.example_verbose:
        print('\nFor veryfication purposes')
        print('The shape of adj_train', adj_train[0].shape)
        print('The shape of adj_val_u', adj_val_u[0].shape)
        print('The shape of adj_val_a', adj_val_a[0].shape)

    # Get the enbeddings from the encoder
    if P.is_pretrain:
        if P.pre_model == 'COST':
            encoder = CoSTEncoder(1, 32, P.cost_kernals, 201, 64, 10, P.cost_alpha, P.cl_temperature, P.is_GCN_encoder, is_sampler, len(adj_train), P.gcn_order, P.gcn_dropout).to(device_encoder)
            encoder = nn.DataParallel(encoder)
        elif P.pre_model == 'TCN':
            encoder = Contrastive_FeatureExtractor_conv(P.cl_temperature, P.is_GCN_encoder, is_sampler, len(adj_train), P.gcn_order, P.gcn_dropout).to(device_encoder)
            encoder = nn.DataParallel(encoder)
        encoder.eval()
        with torch.no_grad():
            encoder.load_state_dict(torch.load(P.save_path+ '/' + 'encoder' + '.pt'))
            train_encoder_input = train_iter.dataset.tensors[0][:,0,:,0].T
            val_u_encoder_input = torch.Tensor(data[:P.train_size,spatialSplit_unseen.i_val]).to(device_encoder).float().T
            val_a_encoder_input = torch.Tensor(data[:P.train_size,spatialSplit_allNod.i_val]).to(device_encoder).float().T
            if P.is_always_augmentation:
                if P.augmentation == 'edge_masking':
                    adj_train = edge_masking(adj_train, 0.02, device_encoder)
                    adj_val_u = edge_masking(adj_val_u, 0.02, device_encoder)
                    adj_val_a = edge_masking(adj_val_a, 0.02, device_encoder)
                elif P.augmentation == 'temporal_shifting':
                    train_encoder_input = temporal_shifting(train_encoder_input, P.temporal_shifting_r).to(device_encoder)
                    val_u_encoder_input = temporal_shifting(val_u_encoder_input, P.temporal_shifting_r).to(device_encoder)
                    val_a_encoder_input = temporal_shifting(val_a_encoder_input, P.temporal_shifting_r).to(device_encoder)
                elif P.augmentation == 'temporal_shifting_new':
                    train_encoder_input = temporal_shifting_new(train_encoder_input, P.temporal_shifting_r).to(device_encoder)
                    val_u_encoder_input = temporal_shifting_new(val_u_encoder_input, P.temporal_shifting_r).to(device_encoder)
                    val_a_encoder_input = temporal_shifting_new(val_a_encoder_input, P.temporal_shifting_r).to(device_encoder)
                elif P.augmentation == 'input_smoothing':
                    train_encoder_input = input_smoothing(train_encoder_input, P.input_smoothing_r, P.input_smoothing_e).to(device_encoder)
                    val_u_encoder_input = input_smoothing(val_u_encoder_input, P.input_smoothing_r, P.input_smoothing_e).to(device_encoder)
                    val_a_encoder_input = input_smoothing(val_a_encoder_input, P.input_smoothing_r, P.input_smoothing_e).to(device_encoder)
            if P.example_verbose:
                print('\nThe shape of train_encoder_input', train_encoder_input.shape)
                print('The shape of val_u_encoder_input', val_u_encoder_input.shape)
                print('The shape of val_a_encoder_input', val_a_encoder_input.shape)
            train_embed = encoder(train_encoder_input, adj_train, P.example_verbose).T.detach().to(device_gpu)
            val_u_embed = encoder(val_u_encoder_input, adj_val_u, P.example_verbose).T.detach().to(device_gpu)
            val_a_embed = encoder(val_a_encoder_input, adj_val_a, P.example_verbose).T.detach().to(device_gpu)
    else:
        train_embed = torch.zeros(32, train_iter.dataset.tensors[0].shape[2]).to(device_gpu).detach()
        val_u_embed = torch.zeros(32, val_u_iter.dataset.tensors[0].shape[2]).to(device_gpu).detach()
        val_a_embed = torch.zeros(32, val_a_iter.dataset.tensors[0].shape[2]).to(device_gpu).detach()

    adj_train = [tensor.to(device_gpu) for tensor in adj_train]
    adj_val_u = [tensor.to(device_gpu) for tensor in adj_val_u]
    adj_val_a = [tensor.to(device_gpu) for tensor in adj_val_a]
    m_time = datetime.now()

    print('\nENCODER INFER DURATION IN MODEL TRAINING:', m_time-s_time)
    print('train_embed', train_embed.shape, train_embed.mean(), train_embed.std())
    print('val_u_embed', val_u_embed.shape, val_u_embed.mean(), val_u_embed.std())
    print('val_a_embed', val_a_embed.shape, val_a_embed.mean(), val_a_embed.std())

    '''
    Decide which dataset to use for training the model
    A: the same dataset used for training the encoder
    B: the same sensors as A, but different timestamps: validation split in temporal dimension
    '''
    if P.train_model_datasplit == 'A':
        model_input = train_iter
    elif P.train_model_datasplit == 'B':
        model_input = train_model_iter
    if P.example_verbose:
        print('\nFor verification purposes')
        print('The shape of model_input', model_input.dataset.tensors[0].shape)
    
    model_log = []
    for epoch in range(P.train_epoch):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        train_criterion = make_criterion(effect_train, use_for_val=False)
        for batch in model_input:
            # Support both (x,y) and (x,y,cys)
            if len(batch) == 3:
                x, y, cys = batch
                cys = cys.to(device_gpu)
            else:
                x, y = batch
                cys = None

            # Apply prediction
            if P.model == 'gwnet':
                y_pred = model(x.to(device_gpu), adj_train, train_embed, P.is_concat_encoder_model, P.is_layer_after_concat)
            elif P.model == 'LSTM':
                y_pred = model(x.to(device_gpu), train_embed, P.encoder_to_model_ratio, P.is_concat_encoder_model, support = adj_train, is_example = P.example_verbose, is_layer_after_concat = P.is_layer_after_concat)
            '''
            The output of y_pred and y should have the same shape
            The shape of y_pred should be [batch_size, timestep_out, number_of_sensors, number_of_channels]
            '''
            if P.example_verbose:
                print('\nFor veryfication purposes')
                print('The shape of y_pred', y_pred.shape)
                print('The shape of y', y.shape)
                print('The corresponding second instance of y_pred', y_pred[0,:,1,0])
                print('The corresponding second instance of y', y[0,:,1,0])
            y_pred = scaler.inverse_transform(y_pred)
            y = scaler.inverse_transform(y.to(device_gpu))
            loss = train_criterion(y, y_pred, cys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n # Final average loss for the epoch
        train_end_time = datetime.now()
        print('TRAINING DURATION FOR EPOCH:', train_end_time-starttime)
        if P.is_testunseen:
            sensor_idx_start = len(spatialSplit_allNod.i_trn)
        else:
            sensor_idx_start = 0
        if P.example_verbose:
            print('\nThe validation for model training starts at index', sensor_idx_start)
        # Calculate the loss for the validation set, and save the optimal model
        val_criterion = make_criterion(effect_val_a, use_for_val=val_use_closure)
        val_a_loss, Y_pred, Y = evaluateModel(P, model, val_criterion, val_a_iter, adj_val_a, val_a_embed, device_gpu, sensor_idx_start)
        if val_a_loss < min_val_a_loss:
            min_val_a_loss = val_a_loss
            final_train_loss = train_loss
            tolerance = 0
            torch.save(model.state_dict(), P.save_path + '/' + name + '_a.pt')
        else:
            tolerance += 1
        endtime = datetime.now()
        epoch_time = (endtime - starttime).total_seconds()
        print("epoch", epoch,
            "time used:",epoch_time," seconds ",
            "train loss:", train_loss,
            "validation all nodes loss:", val_a_loss)
        with open(P.save_path + '/' + name + '_log.txt', 'a') as f:
            f.write("%s %d, %s %d %s, %s %.10f, %s %.10f\n" % \
                ("epoch", epoch,
                 "time used:",epoch_time," seconds ",
                 "train loss:", train_loss,
                 "validation all nodes loss:", val_a_loss))
        if P.is_mongo:
            train_loss_float = float(f"{train_loss:.10f}")
            val_loss_float = float(f"{val_a_loss:.10f}")
            doc = {
                "exe_id": P.exe_id,
                'track_id': P.track_id,
                'replication': P.replication,
                "epoch": epoch,
                "time_used": epoch_time,
                "train_loss": train_loss_float,
                "val_loss": val_loss_float
            }
            model_log.append(doc)
        if tolerance >= P.tolerance:
            break
    e_time = datetime.now()
    print('MODEL TRAINING DURATION:', (e_time-m_time).total_seconds())
    # Write the final results to the log file
    try:
        df = pd.read_csv('save/results.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['exe_id', 'min_pretrain_val_loss', 'pretrain_time'])
    row_exists = 'exe_id' in df.columns and any(df['exe_id'] == P.exe_id)
    if row_exists:
        df.loc[df['exe_id'] == P.exe_id, ['train_loss', 'min_val_a_loss', 'train_time']] = [final_train_loss, min_val_a_loss, (e_time-m_time).total_seconds()]
    else:
        new_row = pd.DataFrame({'exe_id': [P.exe_id], 'train_loss': [final_train_loss], 'min_val_a_loss': [min_val_a_loss], 'train_time': [(e_time-m_time).total_seconds()]})
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv('save/results.csv', index=False)
    if P.is_mongo:
        doc = {
            "exe_id": P.exe_id,
            "train_loss": float(f"{final_train_loss:.10f}"),
            "min_val_a_loss": float(f"{min_val_a_loss:.10f}"),
            "train_time": (e_time-m_time).total_seconds(),
        }
        update_operation = UpdateOne(
            {"exe_id": doc["exe_id"]},  # Search filter
            {"$set": {
                "train_loss": doc["train_loss"],
                "min_val_a_loss": doc["min_val_a_loss"],
                "train_time": doc["train_time"]
            }},
            upsert=True  # Insert the document if it does not exist
        )
        mongodb['performance'].bulk_write([update_operation])
    with open(P.save_path + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, %s, %.10f\n" % (name, mode, 'MAE on train', final_train_loss))
    print('trainModel Ended ...\n')
    return model_log, min_val_a_loss

def evaluateModel(P, model, criterion_fn, data_iter, adj, embed, device, sensor_idx_start, test=False):
    """
    criterion_fn: callable(y_true, y_pred, cys_or_None) -> scalar loss
    data_iter may yield (x,y) or (x,y,cys)
    """
    YS_pred = []
    Y = []
    model.eval()
    torch.cuda.empty_cache()
    l_sum, n = 0.0, 0

    # For LSTM branch you used embed_after_index
    embed_after_index = embed[:, sensor_idx_start:]

    with torch.no_grad():
        for batch in data_iter:
            if len(batch) == 3:
                x, y, cys = batch
                cys = cys.to(device)
            else:
                x, y = batch
                cys = None

            # Forward
            if P.model == 'gwnet':
                y_pred = model(x.to(device), adj, embed, P.is_concat_encoder_model, P.is_layer_after_concat).to(device)
                # slice sensors AFTER prediction
                y_pred = y_pred[:, :, sensor_idx_start:, :]
                # slice labels (and cys) to match
                y = y[:, :, sensor_idx_start:, :].to(device)
                if cys is not None:
                    cys = cys[:, :, sensor_idx_start:, :]

            elif P.model == 'LSTM':
                # slice inputs before prediction; use sliced embed
                x = x[:, :, sensor_idx_start:, :].to(device)
                y_pred = model(
                    x, embed_after_index, P.encoder_to_model_ratio, P.is_concat_encoder_model,
                    support=adj, is_example=P.example_verbose, is_layer_after_concat=P.is_layer_after_concat
                ).to(device)
                # slice labels (and cys) to match
                y = y[:, :, sensor_idx_start:, :].to(device)
                if cys is not None:
                    cys = cys[:, :, sensor_idx_start:, :]

            if P.example_verbose:
                print('\nIn model evaluation process:')
                print('The shape of y_pred', y_pred.shape)
                print('The shape of y', y.shape)
                if cys is not None:
                    print('The shape of cys', cys.shape)

            # Inverse-scale in real units **before** loss
            y_pred = scaler.inverse_transform(y_pred)
            y      = scaler.inverse_transform(y)

            if not test:
                l = criterion_fn(y, y_pred, cys)
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]

            Y.append(y)
            YS_pred.append(y_pred)

        YS_pred = torch.vstack(YS_pred)
        Y = torch.vstack(Y)

    if not test:
        return l_sum / max(n, 1), YS_pred, Y
    else:
        return YS_pred, Y


def testModel(P, name, mode, test_iter, adj_tst, spatialsplit, device_cpu, device_gpu, mongodb, CXS_tst=None, CYS_tst=None):

    criterion = Metrics.MAE
    print('Model Testing', mode, 'Started ...')
    if P.train_encoder_on == 'cpu':
        device_encoder = device_cpu
    else:
        device_encoder = device_gpu

    if P.augmentation == 'sampler':
        is_sampler = True
    else:
        is_sampler = False
    
    if P.is_pretrain:
        if P.pre_model == 'COST':
            encoder = CoSTEncoder(1, 32, P.cost_kernals, 201, 64, 10, P.cost_alpha, P.cl_temperature, P.is_GCN_encoder, is_sampler, len(adj_tst), P.gcn_order, P.gcn_dropout).to(device_encoder)
            encoder = nn.DataParallel(encoder)
        elif P.pre_model == 'TCN':
            encoder = Contrastive_FeatureExtractor_conv(P.cl_temperature, P.is_GCN_encoder, is_sampler, len(adj_tst), P.gcn_order, P.gcn_dropout).to(device_encoder)
            encoder = nn.DataParallel(encoder)
        encoder.load_state_dict(torch.load(P.save_path+ '/' + 'encoder' + '.pt', map_location=device_encoder))
        encoder.eval()
    model = getModel(P, name, device_gpu, len(adj_tst))
    model.load_state_dict(torch.load(P.save_path+ '/' + name +mode[-2:]+ '.pt', map_location=device_gpu))
    s_time = datetime.now()
    
    '''
    Encoder inference
    '''
    print('Model Infer Start ...')
    tst_embed = torch.zeros(32, test_iter.dataset.tensors[0].shape[2]).to(device_gpu).detach()
    adj_tst = [tensor.to(device_encoder) for tensor in adj_tst]
    if P.is_pretrain:
        with torch.no_grad():
            tst_encoder_input = torch.Tensor(data[:P.train_size,spatialsplit.i_tst]).to(device_encoder).float().T
            if P.is_always_augmentation:
                if P.augmentation == 'edge_masking':
                    adj_tst = edge_masking(adj_tst, 0.02, device_encoder)
                elif P.augmentation == 'temporal_shifting':
                    tst_encoder_input = temporal_shifting(tst_encoder_input, P.temporal_shifting_r).to(device_encoder)
                elif P.augmentation == 'temporal_shifting_new':
                    tst_encoder_input = temporal_shifting_new(tst_encoder_input, P.temporal_shifting_r).to(device_encoder) 
                elif P.augmentation == 'input_smoothing':
                    tst_encoder_input = input_smoothing(tst_encoder_input, P.input_smoothing_r, P.input_smoothing_e).to(device_encoder)
            if P.example_verbose:
                print('\nThe shape of tst_encoder_input', tst_encoder_input.shape)
            tst_embed = encoder(tst_encoder_input, adj_tst, P.example_verbose).T.detach().to(device_gpu)
    adj_tst = [tensor.to(device_gpu) for tensor in adj_tst]
    m_time = datetime.now()
    print('ENCODER INFER DURATION:', m_time-s_time)

    if P.is_testunseen:
        sensor_idx_start = len(spatialsplit.i_val)
    else:
        sensor_idx_start = 0
    if P.example_verbose:
        print('\nThe inference for model starts at index', sensor_idx_start)

    print('\nMODEL INFER START ...')
    '''
    Prediction is on all sensors
    But the loss is calculated only for the sensors after sensor_idx_start
    '''
    YS_pred, YS = evaluateModel(P, model, criterion, test_iter, adj_tst, tst_embed, device_gpu, sensor_idx_start, test = True)



    
    e_time = datetime.now()
    print('Model Infer End ...', e_time)
    
    print('MODEL INFER DURATION:', (e_time-m_time).total_seconds())
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    #  Save the results
    torch.save(YS, P.save_path + '/' + 'YS.pt')
    torch.save(YS_pred, P.save_path + '/' + 'YS_pred.pt')

    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    MSE, RMSE, MAE, MAPE = MSE.cpu().numpy(), RMSE.cpu().numpy(), MAE.cpu().numpy(), MAPE.cpu().numpy()
    # Write the final results to the log file
    df = pd.read_csv('save/results.csv')
    columns_to_update = [mode+'_MAE', mode+'_MSE', mode+'_RMSE', mode+'_MAPE', 'encoder_infer_time', 'model_infer_time']
    values_to_assign = [MAE, MSE, RMSE, MAPE, (m_time-s_time).total_seconds(), (e_time-m_time).total_seconds()]
    df.loc[df['exe_id'] == P.exe_id, columns_to_update] = values_to_assign
    df.to_csv('save/results.csv', index=False) 
    if P.is_mongo:
        doc = {
            "exe_id": P.exe_id,
            "finished": True,
            mode+"_MAE": float(f"{MAE:.10f}"),
            mode+"_MSE": float(f"{MSE:.10f}"),
            mode+"_RMSE": float(f"{RMSE:.10f}"),
            mode+"_MAPE": float(f"{MAPE:.10f}"),
            "encoder_infer_time": (m_time-s_time).total_seconds(),
            "model_infer_time": (e_time-m_time).total_seconds(),
        }

    print('*' * 40)
    f = open(P.save_path + '/' + name + '_prediction_scores.txt', 'a')
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(P.timestep_out):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        MSE, RMSE, MAE, MAPE = MSE.cpu().numpy(), RMSE.cpu().numpy(), MAE.cpu().numpy(), MAPE.cpu().numpy()
        df = pd.read_csv('save/results.csv')
        columns_to_update = [mode+'_step_'+str(i+1)+'_MAE', mode+'_step_'+str(i+1)+'_MSE', mode+'_step_'+str(i+1)+'_RMSE', mode+'_step_'+str(i+1)+'_MAPE']
        values_to_assign = [MAE, MSE, RMSE, MAPE]
        df.loc[df['exe_id'] == P.exe_id, columns_to_update] = values_to_assign
        df.to_csv('save/results.csv', index=False)
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        if P.is_mongo:
            nested_obj = {
                "MAE": float(f"{MAE:.10f}"),
                "MSE": float(f"{MSE:.10f}"),
                "RMSE": float(f"{RMSE:.10f}"),
                "MAPE": float(f"{MAPE:.10f}")
            }
            doc['step_'+str(i+1)] = nested_obj
    if P.is_mongo:
        update_operation = UpdateOne(
            {"exe_id": doc["exe_id"]},  # Search filter
            {"$set": doc},
            upsert=True  # Insert the document if it does not exist
        )
        mongodb['performance'].bulk_write([update_operation])
    f.close()
    print('Model Testing Ended ...', time.ctime())

    # ---- Closure-aware metrics (NO BLENDING; use raw predictions) ----
    if P.has_closure_labels:
        # Align closure labels to the same sensor slice used in loss/eval
        from closures import align_cys_for_eval  # keep only this import from closures.py
        CYS_aligned, split_indices_aligned = align_cys_for_eval(
            CYS_tst, spatialsplit, sensor_idx_start, YS.shape[2]
        )

        res = evaluate_closure_metrics(
            y_true=YS,
            y_pred=YS_pred, 
            CYS_split=CYS_aligned,
            effect_matrix_path=getattr(P, 'effect_matrix_path', None),
            sensor_indices=split_indices_aligned,
            closure_mode=getattr(P, 'closure_mode', 'time'),   # 'binary' or 'time'
            top_frac=getattr(P, 'most_top_frac', 0.05)           # if you use top-% most logic
        )

        # ==========================
        # Unseen-vs-seen closure sensors
        # ==========================
        unseen_flag_tst = getattr(P, 'unseen_closure_flag_tst', None)
        if unseen_flag_tst is not None:
            # unseen_flag_tst is in full test sensor order; we only evaluate from sensor_idx_start
            unseen_flag_eval = unseen_flag_tst[sensor_idx_start:]        # [N_eff]
            unseen_flag_eval = torch.from_numpy(unseen_flag_eval).to(YS.device).bool()

            # CYS_aligned has shape [B, T, N_eff, 1]; >0 means closure
            cys_eval = CYS_aligned  # [B, T, N_eff, 1]
            closed_mask = (cys_eval > 0).squeeze(-1)                     # [B, T, N_eff]

            # Broadcast sensor flags to [B, T, N_eff]
            flag_broadcast = unseen_flag_eval.view(1, 1, -1)            # [1,1,N_eff]

            mask_unseen = closed_mask & flag_broadcast                   # closures on unseen-closure sensors
            mask_seen   = closed_mask & (~flag_broadcast)                # closures on seen-closure sensors

            unseen_metrics = Metrics.evaluate_masked(YS, YS_pred, mask_unseen)
            seen_metrics   = Metrics.evaluate_masked(YS, YS_pred, mask_seen)

            print('\n[Unseen/seen closure-sensor metrics]')
            if unseen_metrics is None:
                print('  closure-unseenSens  N=0')
            else:
                umse, urmse, umae, umape = unseen_metrics
                print(f'  closure-unseenSens MSE={umse:.6f} RMSE={urmse:.6f} MAE={umae:.6f} MAPE={umape:.6f}')
            if seen_metrics is None:
                print('  closure-seenSens   N=0')
            else:
                smse, srmse, smae, smape = seen_metrics
                print(f'  closure-seenSens  MSE={smse:.6f} RMSE={srmse:.6f} MAE={smae:.6f} MAPE={smape:.6f}')

            # Optionally write to results.csv
            try:
                df = pd.read_csv('save/results.csv')
                rowmask = (df['exe_id'] == P.exe_id)
                if unseen_metrics is not None:
                    umse, urmse, umae, umape = unseen_metrics
                    df.loc[rowmask, mode+'_closure_unseenSens_MSE']  = float(umse)
                    df.loc[rowmask, mode+'_closure_unseenSens_RMSE'] = float(urmse)
                    df.loc[rowmask, mode+'_closure_unseenSens_MAE']  = float(umae)
                    df.loc[rowmask, mode+'_closure_unseenSens_MAPE'] = float(umape)
                if seen_metrics is not None:
                    smse, srmse, smae, smape = seen_metrics
                    df.loc[rowmask, mode+'_closure_seenSens_MSE']  = float(smse)
                    df.loc[rowmask, mode+'_closure_seenSens_RMSE'] = float(srmse)
                    df.loc[rowmask, mode+'_closure_seenSens_MAE']  = float(smae)
                    df.loc[rowmask, mode+'_closure_seenSens_MAPE'] = float(smape)
                df.to_csv('save/results.csv', index=False)
            except Exception:
                pass


        # Pretty-print (unchanged)
        def _p(name, tup):
            if tup is None:
                print(f'  {name:<14} N=0')
            else:
                mse, rmse, mae, mape = tup
                print(f'  {name:<14} MSE={mse:.6f} RMSE={rmse:.6f} MAE={mae:.6f} MAPE={mape:.6f}')

        print('\n[Closure-aware metrics]')
        _p('all-open',      res.get('all_open'))
        _p('closure-glob',  res.get('closure'))
        _p('closure-most',  res.get('closure_most'))
        _p('closure-other', res.get('closure_other'))

        # Optional: persist to results.csv (same as before)
        try:
            df = pd.read_csv('save/results.csv')
            rowmask = (df['exe_id'] == P.exe_id)
            def _set(prefix, tup):
                if tup is None: return
                mse, rmse, mae, mape = tup
                df.loc[rowmask, prefix+'_MSE']  = float(mse)
                df.loc[rowmask, prefix+'_RMSE'] = float(rmse)
                df.loc[rowmask, prefix+'_MAE']  = float(mae)
                df.loc[rowmask, prefix+'_MAPE'] = float(mape)
            _set(mode+'_all_open',      res.get('all_open'))
            _set(mode+'_closure',       res.get('closure'))
            _set(mode+'_closure_most',  res.get('closure_most'))
            _set(mode+'_closure_other', res.get('closure_other'))
            df.to_csv('save/results.csv', index=False)
        except Exception:
            pass




'''
System parameters
P = type('Parameters', (object,), {})()
P.dataname = 'METRLA'
P.model = 'LSTM'
P.pre_model = 'TCN'
P.track_id = 0
P.replication = 1
P.seed = 0

P.t_train = 0.7
P.t_val = 0.1
P.s_train = 0.7
P.s_val = 0.1
P.fold = 2

P.timestep_in = 12
P.timestep_out = 12
P.n_channel = 1
P.batch_size = 64

P.lstm_hidden_dim = 128
P.lstm_layers = 2
P.lstm_dropout = 0.2
P.gwnet_is_adp_adj = True
P.gwnet_is_SGA = False

P.adj_type = 'doubletransition'
P.adj_method = 1
P.adj_diag = 0
P.cost_kernals = [1, 2, 4, 8, 16, 32, 64, 128]
P.cost_alpha = 0.5
P.cl_temperature = 1
P.is_pretrain = True
P.is_GCN_encoder = True
P.is_GCN_after_CL = True
P.gcn_order = 1
P.gcn_dropout = 0

P.augmentation = 'sampler'
P.temporal_shifting_r = 0.8
P.input_smoothing_r = 0.9
P.input_smoothing_e = 20
P.encoder_to_model_ratio = 1
P.is_concat_encoder_model = True
P.is_layer_after_concat = True
P.is_always_augmentation = True

P.tolerance = 10
P.learn_rate = 0.001
P.pretrain_epoch = 2
P.train_epoch = 1
P.weight_decay = 0
P.is_testunseen = True
P.train_model_datasplit = 'B'
P.train_encoder_on = 'cpu'

P.is_mongo = True
P.example_verbose = True
P.is_tune = False
'''

def main(P):
    global data
    global scaler

    '''
    Check the parameter settings.
    '''
    if P.is_pretrain == False and P.is_concat_encoder_model == True:
        raise ValueError('Pretraining should be enabled for concatenation')
    if P.augmentation == 'edge_masking' and P.is_GCN_encoder == False:
        raise ValueError('edge_masking augmentation requires GCN encoder')
    if P.is_GCN_encoder == True and P.is_GCN_after_CL == True:
        raise ValueError('GCN should be used only in one place')
    if P.is_layer_after_concat == True and P.is_concat_encoder_model == False:
        raise ValueError('Layer after concatenation requires concatenation')
    if P.fold * (1 - P.s_train - P.s_val) > 1:
        raise ValueError('The number of sensors cannot meet this fold requirement')
    if P.is_layer_after_concat and P.gwnet_is_SGA:
        raise ValueError('Layer after concatenation requires no SGA')

    '''
    Set backend devices. 
    Check the type of GPU, either mps or cuda, is available.
    Also, another options is to use the CPU.
    '''
    if torch.backends.mps.is_available():
        device_gpu = torch.device('mps') 
    elif torch.cuda.is_available():
        device_gpu = torch.device('cuda')
    else:
        device_gpu = torch.device('cpu')
    device_cpu = torch.device('cpu')

    # load data from file
    # P.exe_id = P.dataname + '_' + P.model + '_' + P.pre_model + '_' + datetime.now().strftime("%y%m%d-%H%M")
    # P.save_path = 'save/' + P.exe_id
    if P.dataname == 'METRLA':
        print('P.dataname == METRLA')
        data_path = './data/METRLA/metr-la.h5'
        if P.adj_method == 0:
            P.adj_path = './data/METRLA/adj_mx.pkl'
        elif P.adj_method == 1:
            P.adj_path = './data/METRLA/adj_mx_new1.pkl'
        P.n_sensor = 207
        data = pd.read_hdf(data_path).values
    elif P.dataname == 'PEMSBAY':
        print('P.dataname == PEMSBAY')
        data_path = './data/PEMSBAY/pems-bay.h5'
        if P.adj_method == 0:
            P.adj_path = './data/PEMSBAY/adj_mx_bay.pkl'
        elif P.adj_method == 1:
            P.adj_path = './data/PEMSBAY/adj_mx_new1.pkl'
        P.n_sensor = 325
        data = pd.read_hdf(data_path).values
    elif P.dataname == 'PEMSD7':
        print('P.dataname == PEMSD7')
        data_path = './data/PEMSD7/pemsd7.h5'
        if P.adj_method == 0:
            P.adj_path = './data/PEMSD7/adj_mx.pkl'
        elif P.adj_method == 1:
            P.adj_path = './data/PEMSD7/adj_mx_new.pkl'
        P.n_sensor = 228
        data = pd.read_hdf(data_path).values
    elif P.dataname == 'SIM_GRID_NOC':
        print('P.dataname == SIM_GRID_NOC')
        data_path = './data/SIM_GRID/NOC/sensor_readings.h5'
        if P.adj_method == 0:
            P.adj_path = './data/SIM_GRID/NOC/adj_mx_diag0.pkl'
        elif P.adj_method == 1:
            P.adj_path = './data/SIM_GRID/NOC/adj_mx_diag1.pkl'
        P.n_sensor = 120
        data = pd.read_hdf(data_path).values 
    elif P.dataname == 'SIM_GRID_C':
        print('P.dataname == SIM_GRID_C')
        data_path = './data/SIM_GRID/C/sensor_readings.h5'
        if P.adj_method == 0:
            P.adj_path = './data/SIM_GRID/C/adj_mx_diag0.pkl'
        elif P.adj_method == 1:
            P.adj_path = './data/SIM_GRID/C/adj_mx_diag1.pkl'
        P.n_sensor = 120
        data = pd.read_hdf(data_path).values 
    elif P.dataname == 'SIM_GRID_COT':
        print('P.dataname == SIM_GRID_COT')
        data_path = './data/SIM_GRID/COT/sensor_readings.h5'
        if P.adj_method == 0:
            P.adj_path = './data/SIM_GRID/COT/adj_mx_diag0.pkl'
        elif P.adj_method == 1:
            P.adj_path = './data/SIM_GRID/COT/adj_mx_diag1.pkl'
        P.n_sensor = 120
        data = pd.read_hdf(data_path).values 
    elif P.dataname == 'SIM_GRID_5DAY_NOC':
        print('P.dataname == SIM_GRID_5DAY_NOC')
        data_path = './data/SIM_GRID/5DAY_NOC/sensor_readings.h5'
        if P.adj_method == 0:
            P.adj_path = './data/SIM_GRID/5DAY_NOC/adj_mx_diag0.pkl'
        elif P.adj_method == 1:
            P.adj_path = './data/SIM_GRID/5DAY_NOC/adj_mx_diag1.pkl'
        P.n_sensor = 120
        data = pd.read_hdf(data_path).values
        P.has_closure_labels = True
        P.closure_mask_path = './data/SIM_GRID/5DAY_NOC/closure_labels.h5' 
        closure_mask = pd.read_hdf(P.closure_mask_path).values
        P.closure_mask = closure_mask     
        P.effect_matrix_path = './data/SIM_GRID/5DAY_NOC/closure_effect.pkl' 
        P.closure_mode = 'binary'   # 'time' or 'binary' 
    elif P.dataname == 'SIM_GRID_5DAY_C':
        print('P.dataname == SIM_GRID_5DAY_C')
        data_path = './data/SIM_GRID/5DAY_C/sensor_readings.h5'
        if P.adj_method == 0:
            P.adj_path = './data/SIM_GRID/5DAY_C/adj_mx_diag0.pkl'
        elif P.adj_method == 1:
            P.adj_path = './data/SIM_GRID/5DAY_C/adj_mx_diag1.pkl'
        P.n_sensor = 120
        data = pd.read_hdf(data_path).values
        P.has_closure_labels = True
        P.closure_mask_path = './data/SIM_GRID/5DAY_C/closure_labels.h5' 
        closure_mask = pd.read_hdf(P.closure_mask_path).values
        P.closure_mask = closure_mask     
        P.effect_matrix_path = './data/SIM_GRID/5DAY_C/closure_effect.pkl' 
        P.closure_mode = 'binary'   # 'time' or 'binary' 
    elif P.dataname == 'SIM_GRID_2MON_C':
        print('P.dataname == SIM_GRID_2MON_C')
        data_path = './data/SIM_GRID/2MON_C/sensor_readings.h5'
        if P.adj_method == 0:
            P.adj_path = './data/SIM_GRID/2MON_C/adj_mx_diag0.pkl'
        elif P.adj_method == 1:
            P.adj_path = './data/SIM_GRID/2MON_C/adj_mx_diag1.pkl'
        P.n_sensor = 120
        data = pd.read_hdf(data_path).values
        P.has_closure_labels = True
        P.closure_mask_path = './data/SIM_GRID/2MON_C/closure_labels.h5' 
        closure_mask = pd.read_hdf(P.closure_mask_path).values
        P.closure_mask = closure_mask     
        P.effect_matrix_path = './data/SIM_GRID/2MON_C/closure_effect.pkl' 
        P.closure_mode = 'time'   # 'time' or 'binary' 
    elif P.dataname == 'SIM_GRID_5DAY_COT':
        print('P.dataname == OT')
        data_path = './data/SIM_GRID/5DAY_COT/sensor_readings.h5'
        if P.adj_method == 0:
            P.adj_path = './data/SIM_GRID/5DAY_COT/adj_mx_diag0.pkl'
        elif P.adj_method == 1:
            P.adj_path = './data/SIM_GRID/5DAY_COT/adj_mx_diag1.pkl'
        P.n_sensor = 120
        data = pd.read_hdf(data_path).values
        P.has_closure_labels = True
        P.closure_mask_path = './data/SIM_GRID/5DAY_COT/closure_labels.h5' 
        closure_mask = pd.read_hdf(P.closure_mask_path).values
        P.closure_mask = closure_mask     
        P.effect_matrix_path = './data/SIM_GRID/5DAY_COT/closure_effect.pkl'
        P.closure_mode = 'binary'   # 'time' or 'binary' 
    elif P.dataname == '2MON_COTR0TE20D12_36':
        print('P.dataname == 2MON_COTR0TE20D12_36')
        data_path = './data/SIM_GRID/2MON_COTR0TE20D12_36/sensor_readings.h5'
        if P.adj_method == 0:
            P.adj_path = './data/SIM_GRID/2MON_COTR0TE20D12_36/adj_mx_diag0.pkl'
        elif P.adj_method == 1:
            P.adj_path = './data/SIM_GRID/2MON_COTR0TE20D12_36/adj_mx_diag1.pkl'
        P.n_sensor = 120
        data = pd.read_hdf(data_path).values
        P.has_closure_labels = True
        P.closure_mask_path = './data/SIM_GRID/2MON_COTR0TE20D12_36/closure_labels.h5' 
        closure_mask = pd.read_hdf(P.closure_mask_path).values
        P.closure_mask = closure_mask     
        P.effect_matrix_path = './data/SIM_GRID/2MON_COTR0TE20D12_36/closure_effect.pkl'
        P.closure_mode = 'time'   # 'time' or 'binary' 
    elif P.dataname == '2MON_COTR10TE20D12_36':
        print('P.dataname == 2MON_COTR10TE20D12_36')
        data_path = './data/SIM_GRID/2MON_COTR10TE20D12_36/sensor_readings.h5'
        if P.adj_method == 0:
            P.adj_path = './data/SIM_GRID/2MON_COTR10TE20D12_36/adj_mx_diag0.pkl'
        elif P.adj_method == 1:
            P.adj_path = './data/SIM_GRID/2MON_COTR10TE20D12_36/adj_mx_diag1.pkl'
        P.n_sensor = 120
        data = pd.read_hdf(data_path).values
        P.has_closure_labels = True
        P.closure_mask_path = './data/SIM_GRID/2MON_COTR10TE20D12_36/closure_labels.h5' 
        closure_mask = pd.read_hdf(P.closure_mask_path).values
        P.closure_mask = closure_mask     
        P.effect_matrix_path = './data/SIM_GRID/2MON_COTR10TE20D12_36/closure_effect.pkl'
        P.closure_mode = 'time'   # 'time' or 'binary' 
    elif P.dataname == '2MON_COTR20TE20D12_24':
        print('P.dataname == 2MON_COTR20TE20D12_24')
        data_path = './data/SIM_GRID/2MON_COTR20TE20D12_24/sensor_readings.h5'
        if P.adj_method == 0:
            P.adj_path = './data/SIM_GRID/2MON_COTR20TE20D12_24/adj_mx_diag0.pkl'
        elif P.adj_method == 1:
            P.adj_path = './data/SIM_GRID/2MON_COTR20TE20D12_24/adj_mx_diag1.pkl'
        P.n_sensor = 120
        data = pd.read_hdf(data_path).values
        P.has_closure_labels = True
        P.closure_mask_path = './data/SIM_GRID/2MON_COTR20TE20D12_24/closure_labels.h5' 
        closure_mask = pd.read_hdf(P.closure_mask_path).values
        P.closure_mask = closure_mask     
        P.effect_matrix_path = './data/SIM_GRID/2MON_COTR20TE20D12_24/closure_effect.pkl'
        P.closure_mode = 'time'   # 'time' or 'binary' 
    elif P.dataname == 'HAGUE_FULL':
        print('P.dataname == HAGUE_FULL')
        data_path = './data/Hauge/hague_filled.h5'
        if P.adj_method == 1:
            P.adj_path = './data/Hauge/adj_mx1.pkl'
        P.n_sensor = 144
        data = pd.read_hdf(data_path).values
    elif P.dataname == 'HAGUE':
        print('P.dataname == HAGUE')
        data_path = './data/Hauge/hague_comp_filled.h5'
        if P.adj_method == 1:
            P.adj_path = './data/Hauge/adj_mx_comp1.pkl'
        if P.adj_method == 2:
            P.adj_path = './data/Hauge/adj_mx_comp2.pkl'
        P.n_sensor = 89
        data = pd.read_hdf(data_path).values
    elif P.dataname == 'HAGUE_25':
        print('P.dataname == HAGUE_25')
        data_path = './data/Hauge/hague_comp_filled_25.h5'
        if P.adj_method == 1:
            P.adj_path = './data/Hauge/adj_mx_comp1.pkl'
        if P.adj_method == 2:
            P.adj_path = './data/Hauge/adj_mx_comp2.pkl'
        P.n_sensor = 89
        data = pd.read_hdf(data_path).values
    elif P.dataname == 'HAGUE_20_2':
        print('P.dataname == HAGUE_20_2')
        data_path = './data/Hauge/hague_comp_filled_20_2.h5'
        if P.adj_method == 1:
            P.adj_path = './data/Hauge/adj_mx_comp1.pkl'
        if P.adj_method == 2:
            P.adj_path = './data/Hauge/adj_mx_comp2.pkl'
        P.n_sensor = 89
        data = pd.read_hdf(data_path).values
    elif P.dataname == 'HAGUE_20_3':
        print('P.dataname == HAGUE_20_3')
        data_path = './data/Hauge/hague_comp_filled_20_3.h5'
        if P.adj_method == 1:
            P.adj_path = './data/Hauge/adj_mx_comp1.pkl'
        if P.adj_method == 2:
            P.adj_path = './data/Hauge/adj_mx_comp2.pkl'
        P.n_sensor = 89
        data = pd.read_hdf(data_path).values
    elif P.dataname == 'HAGUE_FULL_75':
        print('P.dataname == HAGUE_FULL_75')
        data_path = './data/Hauge/hague_filled_75.h5'
        if P.adj_method == 1:
            P.adj_path = './data/Hauge/adj_mx1.pkl'
        P.n_sensor = 144
        data = pd.read_hdf(data_path).values
    elif P.dataname == 'EXAMPLE':
        print('P.dataname == EXAMPLE')
        data_path = './data/example.h5'
        if P.adj_method == 1:
            P.adj_path = './data/adj_mx_ex1.pkl'
        P.n_sensor = 20
        data = pd.read_hdf(data_path).values
        if P.example_verbose:
            print('\nFirst row at 10 am:', data[0])
            print('\nSecond row at 10:05:', data[1])
    
    '''
    Apply the scaler to the sensor readings. 
    The data is scaled based on the mean and standard deviation of the data.
    '''
    if P.is_mongo:
        mongodb = connect_mongo()
    else:
        mongodb = None

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    P.data_mean = scaler.u
    P.data_std = scaler.z
    if P.example_verbose:
        print('data.shape:', data.shape)
        print('data mean:', P.data_mean)
        print('data stadard deviation:', P.data_std)
        if P.dataname == 'EXAMPLE':
            print('\nFirst row at 10 am after scaling:', data[0])
            print('\nSecond row at 10:05 after scaling:', data[1])
            print('\nFirst sensor data:', data[:,0])
            print('\nSecond sensor data:', data[:,1])

    P.fold_i = 0
    encoder_logs = []
    model_logs = []
    val_losses = []
    for i in range(P.fold):
        P.exe_id = str(P.track_id) + '_r' + str(P.replication) + '_f' + str(P.fold_i) + '_' + P.dataname + '_' + P.model + '_' + P.pre_model + '_' + datetime.now().strftime("%y%m%d-%H%M")
        P.save_path = 'save/' + P.exe_id
        # setup
        (pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod,
        train_iter, train_model_iter, val_u_iter, val_a_iter, tst_u_iter, tst_a_iter,
        adj_train, adj_val_u, adj_val_a, adj_tst_u, adj_tst_a,
        CXS_tst_u, CYS_tst_u, CXS_tst_a, CYS_tst_a,
        CXS_trn, CYS_trn, CXS_val_u, CYS_val_u, CXS_val_a, CYS_val_a) = setups(P, device_gpu)


        
        # save parameters
        save_parameters(P, P, 'save/parameters.csv', mongodb)
        
        if P.is_pretrain:
            if P.train_encoder_on == 'cpu':
                encoder_log = pretrainModel(P, 'encoder', pretrn_iter, preval_iter, adj_train, adj_val_a, device_cpu, spatialSplit_allNod, mongodb)
            else:
                encoder_log = pretrainModel(P, 'encoder', pretrn_iter, preval_iter, adj_train, adj_val_a, device_gpu, spatialSplit_allNod, mongodb)

        model_log, min_val_loss = trainModel(P, P.model, 'train',
            train_iter, train_model_iter, val_u_iter, val_a_iter,
            adj_train, adj_val_u, adj_val_a,
            spatialSplit_unseen, spatialSplit_allNod, device_cpu, device_gpu, mongodb)
        
        val_losses.append(min_val_loss)

        if P.is_tune == False:
            testModel(P, P.model, 'test_a', tst_a_iter, adj_tst_a, spatialSplit_allNod, device_cpu, device_gpu, mongodb, CXS_tst=CXS_tst_a, CYS_tst=CYS_tst_a)

        if P.is_mongo:
            if P.is_pretrain:
                encoder_logs.extend(encoder_log)
            model_logs.extend(model_log)
        
        P.fold_i += 1
    
    if P.is_mongo:
        if P.is_pretrain:
            mongodb['encoder_log'].insert_many(encoder_logs)
        mongodb['model_log'].insert_many(model_logs)
        mongodb.client.close()
    
    return sum(val_losses) / len(val_losses)

if __name__ == '__main__':
    main(P)

    