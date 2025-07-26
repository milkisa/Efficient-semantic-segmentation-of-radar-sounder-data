from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
seed = 42
np.random.seed(seed)


def sharad_data_model():
    print('start')
    import pandas as pd
    p= 453
    rs_image= pd.read_csv('dataset/sharad_rs.csv', header = None).to_numpy().reshape(p,-1,64)
    
    rs_label= pd.read_csv('dataset/sharad_gt.csv', header = None).to_numpy().reshape(p,-1,64)
   # rs_label[rs_label==5]=3



    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    
    
    
    folds_1 = [
    (np.arange(151, 453), np.arange(0, 151)),          # First 151 as test set
    (np.concatenate((np.arange(0, 151), np.arange(302, 453))), np.arange(151, 302)),  # Middle 151 as test set
    ( np.arange(0, 302), np.arange(302, 453))           # Last 151 as test set
    ]

    
    
    folds_2 = [
    (np.arange(0, 136), np.arange(136,453)), # First 136 as training, remaining 317 as test
    (np.arange(159, 295), np.concatenate((np.arange(0, 159), np.arange(295, 453)))),  # Middle 136 as training
    (np.arange(317, 453), np.arange(0, 317))  # Last 136 as training, first 317 as test
    ]
    
    
  
    folds_3 = [
    (np.arange(0, 43), np.arange(43, 427)),                 # First 43 as training, remaining as test
    (np.arange(192, 235), np.concatenate((np.arange(0, 192), np.arange(235, 427)))),  # Middle 43 as training
    (np.arange(384, 427), np.arange(0, 384))                # Last 43 as training, first 384 as test
    ]
    fold= [folds_1, folds_2, folds_3]
    model_dir = ['/home/milkisayebasse/spatial/saved_model/mc10/horizontal_10_90/efficent_u2net_newdata_fold_0_fold__after_bce_itr_10000_train_0.153570_tar_0.010154_ time_2702.474025.pth',
        '/home/milkisayebasse/spatial/saved_model/mc10/horizontal_10_90/efficent_u2net_newdata_fold_1_fold__after_bce_itr_10000_train_0.008921_tar_0.000485_ time_2679.684262.pth',
        '/home/milkisayebasse/spatial/saved_model/mc10/horizontal_10_90/efficent_u2net_newdata_fold_2_fold__after_bce_itr_10000_train_0.075389_tar_0.004418_ time_2686.198569.pth']
    return rs_image,rs_label,fold[2], model_dir

def mc10_data_model(partition_idx=0):
    import pandas as pd
    import torch
    p=427
    #rs_image= pd.read_csv('dataset/data_mc10.csv', header = None).to_numpy().reshape(p,-1,64)
    #rs_label= pd.read_csv('dataset/gt_mc10.csv', header = None).to_numpy().reshape(p,-1,64)
    rs_image=  torch.load('dataset/rs_data.pt',weights_only=False)
    rs_label=  torch.load('dataset/rs_gt.pt', weights_only=False)
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    
    
    partition_1 = [
    (np.arange(140, 427), np.arange(0, 140)),          # First 140 as test set
    (np.concatenate((np.arange(0, 140), np.arange(280, 427))), np.arange(140, 280)),  # Middle 140 as test set
    (np.arange(0, 287), np.arange(287, 427))           # Last 140 as test set
    ]
    
    partition_2 = [
    (np.arange(0, 128), np.arange(128, 427)),           # First 128 as training, remaining 299 as test
    (np.arange(150, 278), np.concatenate((np.arange(0, 150), np.arange(278, 427)))),  # Middle 128 as training
    (np.arange(299, 427), np.arange(0, 299))            # Last 128 as training, first 299 as test
    ]
    
    partition_3 = [
    (np.arange(0, 43), np.arange(43, 427)),                 # First 43 as training, remaining as test
    (np.arange(192, 235), np.concatenate((np.arange(0, 192), np.arange(235, 427)))),  # Middle 43 as training
    (np.arange(384, 427), np.arange(0, 384))                # Last 43 as training, first 384 as test
    ]
    partition= [partition_1, partition_2, partition_3]
    model_dir_1 = ['saved_models/fold_model/70_30/efficent_u2net_newdata_fold_0.000000_go_bce_itr_30000_train_0.003910_tar_0.000100_ time_4615.792476.pth',
    'saved_models/fold_model/70_30/efficent_u2net_newdata_fold_1.000000_go_bce_itr_30000_train_0.001474_tar_0.000049_ time_4620.946236.pth',
    'saved_models/fold_model/70_30/efficent_u2net_newdata_fold_2.000000_go_bce_itr_30000_train_0.003245_tar_0.000052_ time_4621.841755.pth']
    model_dir_2 = ['saved_models/fold_model/30_70/efficent_u2net_newdata_fold_0_fold__before_bce_itr_30000_train_0.001298_tar_0.000036_ time_4122.264396.pth',
    'saved_models/fold_model/30_70/efficent_u2net_newdata_fold_1_fold__before_bce_itr_30000_train_0.006022_tar_0.000048_ time_4133.394689.pth',
    'saved_models/fold_model/30_70/efficent_u2net_newdata_fold_2_fold__before_bce_itr_30000_train_0.001884_tar_0.000020_ time_4136.441822.pth']

    model_dir_3 = ['saved_models/fold_model/10_90/efficent_u2net_newdata_fold_0_fold__before_bce_itr_10000_train_0.076936_tar_0.013119_ time_1476.315381.pth',
    'saved_models/fold_model/10_90/efficent_u2net_newdata_fold_1_fold__before_bce_itr_10000_train_0.092701_tar_0.015821_ time_1474.470069.pth',
    'saved_models/fold_model/10_90/efficent_u2net_newdata_fold_2_fold__before_bce_itr_10000_train_0.091719_tar_0.022336_ time_1476.603739.pth']
    model_dir =[model_dir_1, model_dir_2, model_dir_3]
    return rs_image,rs_label,partition[partition_idx], model_dir[partition_idx]