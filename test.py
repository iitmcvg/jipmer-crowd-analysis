import matplotlib.pyplot as plt
import tensorflow as tf
import os
from input_data import _corrupt_brightness,_corrupt_contrast,_corrupt_saturation,_flip_left_right
import argparse
from resnet_crowd import crowd
# from model_train import loss,optimizer,summary
import cv2


parser = argparse.ArgumentParser(description="Inputs to the code")
parser.add_argument("--load_ckpt",type = str,default='/home/saivinay/Documents/crowd/checkpoints/',help="path to load checkpoints from")
args = parser.parse_args()


image_path = '/home/saivinay/Documents/crowd/shanghai_dataset/part_A/test_data/images/IMG_1.jpg'
heatmap_path = '/home/saivinay/Documents/crowd/shanghai_dataset/part_A/test_data/labels/LAB_1.npy'
count_path = '/home/saivinay/Documents/crowd/shanghai_dataset/part_B/test_data/count/COUNT_1.jpg'

mask = []
count = []

image = cv2.imread(image_path)
# print(image.get_shape())
# test_image = _corrupt_brightness(image,mask,count)
# test_image = _corrupt_contrast(image,mask,count)
# test_image = _corrupt_saturation(image,mask,count)
# test_image = _flip_left_right(image,mask,count)
x = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
model = crowd(x)
saver = tf.train.Saver()


PATH = "/home/saivinay/Documents/crowd/summary"

with tf.Session() as sess:
    # summ_writer = tf.summary.FileWriter(PATH,sess.graph)
    # summary = tf.summary.merge_all()

    saver.restore(sess,tf.train.latest_checkpoint(args.load_ckpt))
    image = cv2.resize(image, (224,224))
    [heatmap_val , count_val] = sess.run(model.output, feed_dict={x: image[None, :, :, :]})
    print(heatmap_val)
    print(heatmap_val.shape)
    plt.imshow(heatmap_val[0,:,:,0])
    plt.show()
    # _,loss_val,summary_val  = sess.run([optimizer,loss,summary])
    # heatmap,count = sess.run([crowd(image_path).output])
    
    # summ_writer.add_summary(summary_val)
    # tf.summary.image("heatmap",heatmap)
    # tf.summary.scalar("count",count)


    # summ_writer.add_summary(summary_val)