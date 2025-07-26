import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
def save_output(s_inputs,pred,labels, num,fold):

# print(s_inputs.shape)
    
    predict = pred

    predict = predict.squeeze()
    label= labels.squeeze()

    predict= F.softmax(predict, dim=0)

    predict = predict.cpu().data.numpy()
    s_inputs = s_inputs.cpu().data.numpy()
    np.set_printoptions(suppress=True, precision=4)

    predict_np= np.argmax(predict, axis=0)
    rgb= color_mapping(predict_np)
    #print(rgb.shape)
    rgb= rgb.astype(np.uint8)

    label_rgb= color_mapping(label)

    label_rgb= label_rgb.astype(np.uint8)
    output_dir= "test_data/folds/" + str(fold) + "/"
    if not os.path.exists(output_dir):
         os.makedirs(output_dir)


    name= "test_data/folds/"+str(fold)+ "/" + str(num) + '.png'
    # names= str(num) + "rtoated.png"
    label_names= "test_data/folds/" +str(fold)+ "/" +str(num) + "_labels.png"
# dense_names= "D:/important/phd/project/scribble/efficent_u2net/result/scribble/26_aug/" +str(num) + "_dense.png"
    # # # plt.imsave(name,rgb)
    pred_names= "test_data/folds/" +str(fold)+ "/" +str(num) + "_pred.png"
    plt.imsave(pred_names, rgb)
    plt.imsave(label_names, label_rgb)
    plt.imsave(name,s_inputs.reshape(rgb.shape[0],64))


    return predict_np, label
def color_mapping(predict_np):
    green = [0, 255, 0]    # Green
    yellow = [255, 255, 0] # Yellow
    red = [255, 0, 0]      # Red
    blue = [0, 0, 255]     # Blue
    purple = [75,0, 130] # Purple
    orange = [255, 165, 0] # Orange
    # # print(predict_np.shape[0])
    rgb= np.zeros((predict_np.shape[0], predict_np.shape[1], 3), dtype=int)
    rgb[predict_np==0,:]= purple
    rgb[predict_np==3,:]= yellow
    rgb[predict_np==2,:]= green
    rgb[predict_np==1,:]= blue
    rgb[predict_np==4,:]= orange
    rgb[predict_np==5,:]= red
    return rgb;  
def dist(predict, scribble):
    scribble_label= scribble.squeeze()
    distance_threshold = 4
    predict=predict

# Get the indices of the scribble labels (non-zero entries)
# Get the indices of the scribble labels (non-zero entries)
    scribble_indices = np.argwhere(scribble_label != 0)

    # Create an output array initialized to zeros
    sorted_predict_indices = np.argsort(predict, axis=0)
    second_max_class = sorted_predict_indices[-2]  # Second-highest prediction class for each pixel
    print(np.unique(second_max_class),'yayyyyyyyyyyyyy')
    print(np.unique(np.argmax(predict, axis=0)))
    final_label = np.copy(second_max_class)
    zero_label= np.zeros_like(np.argmax(predict, axis=0))
    height, width = np.argmax(predict, axis=0).shape

    # Get the shape of the prediction array
   

    # Create a grid of coordinates
    y_coords, x_coords = np.indices((height, width))

    # Iterate through each scribble index to calculate distances
    for scribble in scribble_indices:
        scribble_y, scribble_x = scribble
        scribble_value = scribble_label[scribble_y, scribble_x]

        # Compute the Euclidean distance from all points to the scribble label
        distance = np.sqrt((y_coords - scribble_y) ** 2 + (x_coords - scribble_x) ** 2)

        # Set the distance threshold based on the scribble value
        distance_threshold = {3: 1150, 2: 6, 1: 40}.get(scribble_value, 0)

        # Create a mask where conditions are met
        mask = (np.argmax(predict, axis=0) == scribble_value) & (distance < distance_threshold)

        # Apply the mask to the final labels
        final_label[mask] = scribble_value
        zero_label[mask]= scribble_value
    return zero_label,final_label


