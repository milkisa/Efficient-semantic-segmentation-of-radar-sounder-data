
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from implementation.output import save_output
import os
import numpy as np
# --------- 4. inference for each image ---------
def ite_test(test_salobj_dataloader,net,fold):
    rs_pred=[]
    rs_lab=[] 
    net.eval()
    num=0

    for i_test, data_test in enumerate(test_salobj_dataloader):


        # print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        labels= data_test['label']
        # print(type(inputs_test))

        inputs_test = inputs_test.type(torch.FloatTensor)




        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)


        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization

        pred = d1[:,:,:,:]
      
        

        # pred = normPRED(pred)



        #propagate= rs_propagate[num]

        p,l = save_output(inputs_test, pred,labels,num ,fold)
        
        rs_pred.append(p)

        rs_lab.append(l)

        num= num+1

        del d1,d2,d3,d4,d5,d6,d7,p,l
    return rs_pred, rs_lab