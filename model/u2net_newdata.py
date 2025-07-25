def run():
    # import parallelTestModule

    # if __name__ == '__main__':    
    #     extractor = parallelTestModule.ParallelExtractor()
    #     extractor.runInParallel(numProcesses=2, numThreads=4)
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    import os
    import torch
    import torchvision
    from torch.autograd import Variable
    import torch.nn as nn
    import torch.nn.functional as F

    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, utils
    import torch.optim as optim
    import torchvision.transforms as standard_transforms

    import numpy as np
    import glob
    import os
    import matplotlib.pyplot as plt

    from data_loader import Rescale
    from data_loader import RescaleT
    from data_loader import RandomCrop
    from data_loader import ToTensor
    from data_loader import ToTensorLab
    from data_loader import SalObjDataset

    from model import U2NET
    from model import U2NETP

    # ------- 1. define loss function --------

    

    def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
        # print(d0.shape)
        # print(labels_v.shape)
        labels_v= labels_v.squeeze(1)
        # print(labels_v.shape)
        # labels_v= torch.tensor(labels_v, dtype=torch.long)
        labels_v= labels_v.type(torch.long)
        # print(labels_v)
        # print(d0)
        

        loss0 = bce_loss(d0, labels_v)
        print("loss")
        print(d0.shape)
        print(labels_v.shape)
        print(loss0.shape)
        loss1 = bce_loss(d1, labels_v)
        loss2 = bce_loss(d2, labels_v)
        loss3 = bce_loss(d3, labels_v)
        loss4 = bce_loss(d4, labels_v)
        loss5 = bce_loss(d5, labels_v)
        loss6 = bce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
        loss5.data.item(), loss6.data.item()))

        return loss0, loss

    # ------- 2. set the directory of training dataset --------

    model_name = 'u2netp'  # 'u2netp'

    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = os.path.join('im_aug' + os.sep)
    tra_label_dir = os.path.join('gt_aug' + os.sep)

    image_ext = '.jpg'
    label_ext = '.jpg'

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
    print(model_dir)

    epoch_num = 100000
    batch_size_train = 8
    batch_size_val = 1
    train_num = 0
    val_num = 0

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]
        # print(img_name)


        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)
    # print(train_num)
    # print(tra_img_name_list[1])
    import rasterio
    from rasterio.plot import reshape_as_image
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader
    import torch
    import pandas as pd
    import scipy.io as sio
    data = sio.loadmat('dataset.mat')
    dataframe = pd.DataFrame(data.get('dataset2'))
    raster= dataframe.to_numpy()
    gt = pd.DataFrame(data.get('gt2'))
    raster_label= gt.to_numpy()
#raster= rasterio.open("Data_20101104_06_extended_aligned_dB_target").read()
#raster_label= rasterio.open("Data_20101104_06_extended_aligned_dB_target_rois_for_classification").read()

    raster= raster.reshape(1, 1536,20000)
    raster_label= raster_label.reshape(1,1536,20000)

# def mw_to_dbm(mw):
#     return 10*np.log10(np.abs(mw))

    rs_images= np.empty((100, 1,1536,200))
    rs_labels= np.empty((100,1,1536,200))
    i=0
    s=0
    for x in range (200, 20000, 200):
        rs_images[i,:,:,:]=raster[:,:, s:x]
        rs_labels[i,:,:,:]= raster_label[:,:,s:x]
        s=x
        i= i+1
    rs_images= rs_images.reshape(100,1536,200,1)
    rs_labels= rs_labels.reshape(100,1536,200,1)
    rs_image= rs_images[:75]
    rs_label= rs_labels[:75]

    print('rs_immaggges', rs_image.shape)
    print(rs_label.shape)
    print(rs_image.shape)
    print(rs_label.shape)
    label= rs_label.reshape(-1)
    from sklearn.utils import class_weight
    import numpy as np
    class_weights=class_weight.compute_class_weight(class_weight = 'balanced',classes=np.unique(label), y= label)
    class_weights=class_weights.astype(np.float32)
    class_weights= torch.from_numpy(class_weights)
    class_weights= class_weights.cuda()
    bce_loss = nn.CrossEntropyLoss(size_average=True, weight=class_weights, ignore_index=0)
    salobj_dataset = SalObjDataset(
        img_name_list=rs_image,
        lbl_name_list= rs_label,
        transform=transforms.Compose([
            # RescaleT(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

    # ------- 3. define model --------
    # define the net
    if (model_name == 'u2net'):
        net = U2NET(1, 6)
    elif (model_name == 'u2netp'):
        net = U2NETP(1, 6)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 50000  # save the model every 2000 iterations

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            # print(inputs[1])
            # print(inputs.shape)
            # plt.imshow(inputs[1].reshape(410,200,1))
            # plt.show()
            # plt.imshow(labels[1].reshape(410,200,1))
            # plt.show()
       
# 
            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:
                torch.save(net.state_dict(), model_dir + model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0


if __name__ == '__main__':
    print("start")
    run()


