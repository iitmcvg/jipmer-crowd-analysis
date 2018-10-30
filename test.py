import matplotlib.pyplot as plt
import tensorflow as tf
import os
from input_data import _corrupt_brightness,_corrupt_contrast,_corrupt_saturation,_flip_left_right
import argparse
from resnet_crowd import crowd
# from model_train import loss,optimizer,summary
import cv2
import numpy as np


parser = argparse.ArgumentParser(description="Inputs to the code")
parser.add_argument("--load_ckpt",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/checkpoints/',help="path to load checkpoints from")
parser.add_argument("--image_path",type=str,default='/home/saivinay/Documents/jipmer-crowd-analysis/shanghai_dataset/part_A/test_data/images/IMG_1.jpg',help="Path for test images")
parser.add_argument("--heatmap_path",type=str,default='/home/saivinay/Documents/jipmer-crowd-analysis/shanghai_dataset/part_A/test_data/labels/LAB_1.npy',help="Path for actual heatmap")
parser.add_argument("--count_path",type=str,default='/home/saivinay/Documents/jipmer-crowd-analysis/shanghai_dataset/part_A/test_data/count/COUNT_1.npy',help="Path for actual count")
args = parser.parse_args()


image_path = args.image_path
ground_image_path = args.heatmap_path
ground_count_path = args.count_path

mask = []
ground_truth_count = np.load(ground_count_path)
print(ground_truth_count)
image = cv2.imread(image_path)


cv2.imshow("img",image)
cv2.waitKey(1000) # a gray scale image
cv2.destroyAllWindows()

x = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
model = crowd(x)
saver = tf.train.Saver()


with tf.Session() as sess:
    # summ_writer = tf.summary.FileWriter(PATH,sess.graph)
    # summary = tf.summary.merge_all()

    saver.restore(sess,tf.train.latest_checkpoint(args.load_ckpt))
    
    image = cv2.resize(image, (224,224))
    [heatmap_val , count_val] = sess.run(model.output, feed_dict={x: image[None, :, :, :]})

    # heatmap_val = tf.reshape(heatmap_val,[224,224,1])
    print(heatmap_val.shape)
    print(image.shape)

    plt.subplot(2,2,1)
    plt.plot(heatmap_val[0,:,:,0])
    plt.subplot(2,2,2)
    plt.plot(image[:,:,0])
    plt.show()
    
    # _,loss_val,summary_val  = sess.run([optimizer,loss,summary])
    # heatmap,count = sess.run([crowd(image_path).output])
    
    # summ_writer.add_summary(summary_val)
    # tf.summary.image("heatmap",heatmap)
    # tf.summary.scalar("count",count)


    # summ_writer.add_summary(summary_val)