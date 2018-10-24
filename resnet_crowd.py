
# Estimating the crowd count and density map using shanghai dataset 

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.models import Model , Sequential
from keras import layers
from keras.layers import Conv2D,MaxPool2D,ReLU,Dense,AveragePooling2D,Flatten,Concatenate
from keras.activations import sigmoid
import cv2
from tensorflow import image

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
from glob import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
from matplotlib import cm as CM
from tqdm import tqdm
from numba import cuda
# from image import *
# %matplotlib inline


# input_layer = keras.Input((None,None,3)) 


def crowd(input_images):

    base_model = ResNet50(input_tensor = input_images,input_shape=(224,224,3),weights='imagenet',include_top=False)
    model = Model(inputs = base_model.input,outputs = base_model.get_layer("activation_22").output )
    for layer in model.layers:
        layer.trainable = False
    
    # For producing heatmap of the input
    model1 = Conv2D(1,(3,3),padding='same',activation='relu', name="heatmap_a")(model.output)
    # model1 = np.asarray(model1)


    # heatmap = sigmoid(model1)
    # heatmap = Flatten()(heatmap)

    # heatmap = cv2.resize(heatmap,(224,224))


    # For performing counting,violence prediction,denstiy classification
    model2 = AveragePooling2D(pool_size = 2,strides = None,padding = 'same' )(model.output)

    # For violence classification
    # model21 = Flatten()(model2)
    # model21 = Dense(units = 32,activation = 'relu',input_dim = 100352 )(model21)
    # model21 = Dense(units = 2,activation = 'softmax',input_dim = 32 )(model21)

    # For density classification into 5 classes 
    # model22 = Flatten()(model2)
    # model22 = Dense(units = 32,activation = 'relu',input_dim = 100352 )(model22)
    # model22 = Dense(units = 5,activation = 'softmax',input_dim = 32 )(model22)

    # For counting number of people 
    model23 = Flatten()(model2)
    model23 = Dense(units = 32,activation = 'relu',input_dim = 100352, name="counting_a" )(model23)
    model23 = Dense(units = 1,activation = 'softmax', name="counting_b")(model23)
    # model23 = Flatten()(model23)
    # count = model23

    model_final = Model(inputs = base_model.input, outputs = [model1, model23])
    # print (model_final.summary())
    
    return model_final  
    # return model_final,heatmap,count



def calculate_loss(model,ground_truth_heatmap,ground_truth_count):
    
    heatmap, count = model.output
    # print(count)
    tf.summary.image('predicted', heatmap)                          # predicted image of the predicted image 
    tf.summary.image('ground_truth', ground_truth_heatmap)          # ground truth image
    heatmap = tf.reshape(heatmap,[-1, 28*28])                       # making all the output images flat
    ground_truth_heatmap = tf.reshape(ground_truth_heatmap, [-1, 28*28])
    loss_heatmap = tf.keras.backend.binary_crossentropy(
                        ground_truth_heatmap,
                        heatmap,
                        from_logits=False
                    )
    loss_heatmap = tf.reduce_mean(loss_heatmap, axis=1)
    loss_counting = tf.losses.mean_squared_error(
                        ground_truth_count,
                        count,
                        weights=1.0,
                        scope=None,
                        loss_collection=tf.GraphKeys.LOSSES,
                        reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
                    )

    loss = loss_heatmap + loss_counting 
    print (loss_counting.get_shape())
    print (loss_heatmap.get_shape())
    print (loss.get_shape())

    return loss
    



'''
def non_trainable(model,layer1_name,layer2_name):
    model1 = Model(inputs = model.input,outputs = model.get_layer(layer1_name).output )
    for layer in model1.layers:
        layer.trainable =False

    model2 = Model(inputs = layer2 ,outputs = model.output  )

    return Model(inputs = model1.input,outputs = model2.output)

model = crowd(x)
'''

# if __name__ == '__main__':
#     heatmap,count = crowd() 