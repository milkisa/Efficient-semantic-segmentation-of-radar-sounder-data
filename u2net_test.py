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
import h5py

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset



from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from skimage.transform import rotate

import rasterio
from rasterio.plot import reshape_as_image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import scipy.io as sio

#raster= rasterio.open("Data_20101104_06_extended_aligned_dB_target").read()
#raster_label= rasterio.open("Data_20101104_06_extended_aligned_dB_target_rois_for_classification").read()
raster= rasterio.open("Data_20101104_06_extended_aligned_dB_target").read()
raster_label= rasterio.open("Data_20101104_06_extended_aligned_dB_target_rois_for_classification").read()





rs_image= np.empty((427, 1,410,64))
rs_label= np.empty((427,1,410,64))
i=0
s=0

for x in range (64, 27328, 64):
        rs_image[i,:,:,:]=raster[:,:, s:x]
        rs_label[i,:,:,:]= raster_label[:,:,s:x]
        s=x
        i= i+1

    #for x in range (200, 27151, 200):
     #   rs_image[i,:,:,:]=raster[:,:, s:x]
      #  rs_label[i,:,:,:]= raster_label[:,:,s:x]
       # s=x
        #i= i+1
print(rs_image.shape)
rs_image= rs_image.reshape(427,410,64,1)
rs_label= rs_label.reshape(427,410,64,1)
rs_image= rs_image[298:426]
rs_label= rs_label[298:426]


# df= pd.read_csv('power.csv', header= None)
# a= df.to_numpy()
# f = h5py.File('Data_img_01_20101104_06_023.mat','r')
# data = f.get('Data')
# data = np.array(data) # For converting to a NumPy array
# data =data.reshape(902, 3537)
# power= mw_to_dbm(data)
# i=0
# s=0
# for x in range (200, 3537, 200):
#     rs_images[i,:,:,:]=power[:, s:x]
# #     rs_label[i,:,:,:]= raster_label[:,:,s:x]
#     s=x
#     i= i+1
# print(rs_images.shape)
# rs_images= rs_images.reshape(17,902,200,1)
print(rs_image.shape)


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(pred,labels, d_dir, num):
    print("images")
    predict = pred
 
    predict = predict.squeeze()
    label= labels.squeeze()
    print(label.shape, "labels")
    # label= rotate(label, 180)
    print(predict.shape)
    predict_np = predict.cpu().data.numpy()
     # print(predict_np.shape)
    predict_np= np.argmax(predict_np, axis=0)
    print(predict_np.shape, "prediction++++")
    # print(predict_np.shape)
    tp1= 0
    fp1=0
    fn1=0
    tp2= 0
    fp2=0
    fn2=0
    tp3= 0
    fp3=0
    fn3=0
    tn1=0
    tn2=0
    tn3=0
    
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            pred= predict_np[i,j]
            # print()
            # print(label.shape)
            lab= label[i,j]
            if ((lab==1 and pred==1)):
                tp1= tp1+1
            elif (pred==1 and (lab==3 or lab== 2)):
                fp1=fp1+1
            elif ((pred==2 or pred==3 or pred==0) and lab==1):
                fn1= fn1+1

            elif ((lab==2 and pred==2)):
                tp2= tp2+1
            elif (pred==2 and (lab==1 or lab== 3)):
                fp2=fp2+1
            elif ((pred==1 or pred==3 or pred==0) and lab==2):
                fn2= fn2+1

            elif ((lab==3 and pred==3)):
                tp3= tp3+1
            elif (pred==3 and (lab==1 or lab== 2)):
                fp3=fp3+1
            elif ((pred==2 or pred==0 or pred==1) and lab==3):
                fn3= fn3+1
          
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            pred= predict_np[i,j]
            # print()
            # print(label.shape)
            lab= label[i,j]
            if(pred!=1 and lab!=1 and lab!=0):
                tn1= tn1+1
            elif(pred!=2 and lab!=2 and lab!=0):
                tn2= tn2+1
            elif(pred!=3 and lab!=3 and lab!=0):
                tn3= tn3+1
    Recall1= tp1/ (tp1+fn1+0.0000000000000001)
    precision1= tp1/(tp1+fp1+0.00000000000001)
    print('tp2',tp2,'fn2', fn2, 'fp2', fp2, 'num',num)
    Recall2= (tp2+0.000000000000000001)/(tp2+fn2+0.000000000000000001)
    precision2= (tp2+0.000000000000000001)/(tp2+fp2+0.000000000000000001)
    print('tp2',tp2,'fn2', fn2, 'fp2', fp2, 'num',num, 'Recall',Recall2, 'precision', precision2)
    Recall3= tp3/ (tp3+fn3+0.0000001)
    precision3= tp3/(tp3+fp3+0.0000000001)
    accuracy1= (tp1+tn1)/(tp1+fn1+tn1+fp1+0.00000000000001)
    accuracy2= (tp2+tn2)/(tp2+fn2+tn2+fp2+0.0000000000001)
    accuracy3= (tp3+tn3)/(tp3+fn3+tn3+fp3+0.000000000000000001)
          



   
    # predict_np= predict_np.reshape(320,320,3)
   
    # plt.imshow(rgb)
    # plt.show()
    # image= rs_labels[num]
    # print(image)
    # print(np.max(image))
    # print(np.max(predict_np))
    # max= np.max(np.abs(image))
    # print(max)
    # predict_np= predict_np*max
    # # predict= predict_np.astype(np.uint8)
    # print(predict)
    # print(np.max(predict))

   # print(predict_np)
    rgb= color_mapping(predict_np)
    #print(rgb.shape)
    rgb= rgb.astype(np.uint8)

    label_rgb= color_mapping(label)
   # print(label_rgb.shape,'labellllll')
    label_rgb= label_rgb.astype(np.uint8)
 
   # plt.imshow(rgb)
   # plt.show()
   # plt.imshow(label_rgb)
   # plt.show()


    name= "result/" + str(num) + '.png'
    # names= str(num) + "rtoated.png"
    label_names= "result/" +str(num) + "_labels.png"
    plt.imsave(name,rgb)
    #plt.imsave(name, rgb)
    plt.imsave(label_names, label_rgb)
    # return Recall1, Recall2,Recall3, precision1, precision2, precision3,accuracy1,accuracy2,accuracy3
    # print(predict_np.shape)
    # plt.imshow(predict)
    # plt.show()

    # im = Image.fromarray(predict_np, 'RGB')
    
    # plt.imshow(im)
    # plt.show()*
  
    # img_name = image_name.split(os.sep)[-1]
    # matimage= mpimage.imread(image_name)
    
    # image = io.imread(image_name)
    # im= io.imread(r'1.jpg')
    # im = Image.fromarray(im, 'RGB')
    # imo = im.resize(((image.shape[1],image.shape[0])),resample=Image.BILINEAR)

    # # plt.imshow(imo)
    # # plt.show()

    # pb_np = np.array(im)

    # aaa = img_name.split(".")
    # bbb = aaa[0:-1]
    # imidx = bbb[0]
    # for i in range(1,len(bbb)):
    #     imidx = imidx + "." + bbb[i]

    # imo.save(d_dir+imidx+'.png')
    return Recall1,precision1,Recall2, precision2, Recall3,precision3, accuracy1,accuracy2,accuracy3
def color_mapping(predict_np):
    y = np.array([255, 255, 0]) 
    g = np.array([60,179,113])
    bl = np.array([75,0, 130])
    r= np.array([153,0, 0])
    lb=np.array([153,255,255])


    b= np.array([70,130,180])
    # # print(predict_np.shape[0])
    rgb= np.zeros((predict_np.shape[0], predict_np.shape[1], 3), dtype=int)
    rgb[predict_np==0,:]= bl
    rgb[predict_np==3,:]= y
    rgb[predict_np==2,:]= g
    rgb[predict_np==1,:]= b
    rgb[predict_np==4,:]= lb
    rgb[predict_np==5,:]= r
    return rgb;  
def main():
    print('start')

    # --------- 1. get image path and name ---------
    model_name='u2netp'#u2netp



    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(),'new_experiment','more_time','test_298+','no_augmentation_400-64','saved_model', "u2net_bce_itr_600000_train_0.005324_tar_0.000002_ time_47083.832987" + '.pth')
    print(model_dir)

    img_name_list = glob.glob(image_dir + os.sep + '*')
    

    # print(rs_label.shape)

    # --------- 2. dataloader ---------
    #1. dataloader

    test_salobj_dataset = SalObjDataset(img_name_list = rs_image,
                                        lbl_name_list= rs_label,
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
        net = U2NET(1,4)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(1,4)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    num= 298
    rec1=[]
    rec2=[]
    rec3=[]
    pre1=[]
    pre2=[]
    pre3=[]
    acc1=[]
    acc2=[]
    acc3=[]
    # --------- 4. inference for each image ---------
    print('before for loop')
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print('for looop')

        # print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        labels= data_test['label']
        # print(type(inputs_test))

        inputs_test = inputs_test.type(torch.FloatTensor)
        # labels = labels.type(torch.FloatTensor)
        # print(inputs_test.shape,"include 0")
        # inputs_test= inputs_test[labels!=0]
        # print(inputs_test.shape,"exclude 0")
        # print(inputs_test)
        

        # inputs_test= torch.reshape(inputs_test, (1,1,410,200))

        # print("inputs======", inputs_test.shape, type(inputs_test))

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
       # summary(net, (1, 410,200))

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        print(d1.shape)
        pred = d1[:,:,:,:]
        print(pred.shape)
        # pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        r1,p1,r2,p2,r3,p3 ,a1,a2,a3 = save_output(pred,labels, prediction_dir,num )
        print(r1)
        rec1.append(r1)
        rec2.append(r2)
        rec3.append(r3)
        pre1.append(p1)
        pre2.append(p2)
        pre3.append(p3)
        acc1.append(a1)
        acc2.append(a2)
        acc3.append(a3)
        num= num+1

        del d1,d2,d3,d4,d5,d6,d7
    print(len(rec1))
    rs_recal1=sum(rec1)/len(rec1)
    rs_recall2= sum(rec2)/len(rec2)
    rs_recall3= sum(rec3)/len(rec3)
    rs_precision1= sum(pre1)/len(pre1)
    rs_precision2= sum(pre2)/len(pre2)
    rs_precision3= sum(pre3)/len(pre3)
    rs_recall_tot= (rs_recall2+rs_recal1+rs_recall3)/3
    rs_precision_tot= (rs_precision1+rs_precision2+rs_precision3)/3

    print("average recall of call 1 =", sum(rec1)/len(rec1), "-----all", rec1)
    print("average recall of call 2 =", sum(rec2)/len(rec2), "-----all", rec2)
    print("average recall of call 3 =", sum(rec3)/len(rec3), "-----all", rec3)
    print("average precisoin of call 1 =", sum(pre1)/len(pre1), "-----all", pre1)
    print("average precision of call 2 =", sum(pre2)/len(pre2), "-----all", pre2)
    print("average precision of call 3 =", sum(pre3)/len(pre3), "-----all", pre3)
    print("precision =", rs_precision_tot)
    print("recall =", rs_recall_tot)
    print("average accuracy of call 1 =", sum(acc1)/len(acc1), "-----all", acc1)
    print("average accuracy of call 2 =", sum(acc2)/len(acc2), "-----all", acc2)
    print("average accuracy of call 3 =", sum(acc3)/len(acc3), "-----all", acc3)


if __name__ == "__main__":
    main()
