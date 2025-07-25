import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
from PIL import Image
import glob

import cv2

from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from skimage.transform import rotate

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import scipy.io as sio


from implementation.output import save_output
from implementation.metrics import calculate_recall_precision
from implementation.metrics import average_recall_precision
from implementation.metrics import calc_metrics
from implementation.metrics import cv_calc
from implementation.test import ite_test
from implementation.dataset import mc10_data_model
seed = 42
np.random.seed(seed)


p= 427

all_fold_recalls = []
all_fold_precisions = []
all_fold_accuracies = []
rs_pred = []
rs_lab = []
def main():
    rs_image, rs_label, folds, model_dir = mc10_data_model()
    
    #kf.split(rs_image)
    for fold, (train_index, test_index) in enumerate(folds):
        torch.cuda.empty_cache()
        print(fold)
        print(type(fold))
        print(f"\nFold {fold + 1}")
        
        # Split images and labels into train/test for the current fold
        train_images, test_images = rs_image[train_index], rs_image[test_index]
        train_labels, test_labels = rs_label[train_index], rs_label[test_index]
        
        # Display the shapes of the training and testing data
        print("Train Images shape:", train_images.shape)
        print("Test Images shape:", test_images.shape)
        print("Train Labels shape:", train_labels.shape)
        print("Test Labels shape:", test_labels.shape)

        rs_image_fold= np.expand_dims(test_images, axis=-1)
        rs_label_fold= np.expand_dims(test_labels, axis=-1)
        print(rs_image_fold.shape,'testing image size')

        # --------- 1. get image path and name ---------
        model_name='u2netp'#u2netp


        test_salobj_dataset = SalObjDataset(img_name_list = rs_image_fold,
                                            lbl_name_list= rs_label_fold,
                                            # lbl_name_list = [],
                                            transform=transforms.Compose([
                                                                        ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)

        # --------- 3. model define ---------
        if(model_name=='u2net'):
            print("...load U2NET---173.6 MB")
            net = U2NET(1,5)
        elif(model_name=='u2netp'):
            print("...load U2NEP---4.7 MB")
            net = U2NETP(1,5)

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_dir[fold]))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_dir[fold], map_location='cpu'))
    


        #flops, params = get_model_complexity_info(net, (1, 410, 64), as_strings=True, print_per_layer_stat=True)
        #||||||||||||||||||||||||||||||||||||||||||||test |||||||||||||||||||||||||||||||||||||||||||||||||
        rs_pred, rs_lab= ite_test(test_salobj_dataloader,net,fold)
        #||||||||||||||||||||||||||||||||||||||||||||recall_precision |||||||||||||||||||||||||||||||||||||||||||||||||
       
        avg_recall, avg_precision , avg_accuracy  = calc_metrics(rs_pred, rs_lab)
        print(model_dir[fold])
        all_fold_recalls.append(avg_recall)
        all_fold_precisions.append(avg_precision)
        all_fold_accuracies.append(avg_accuracy)
     #||||||||||||||||||||||||||||||||||||||||||||overalll |||||||||||||||||||||||||||||||||||||||||||||||||
    cv_calc(all_fold_recalls,all_fold_precisions,all_fold_accuracies)
    print("number of test sample is ", rs_image_fold.shape)
if __name__ == "__main__":
    main()
