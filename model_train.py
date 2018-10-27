

import argparse
import tensorflow as tf
import numpy as np
from resnet_crowd import crowd,calculate_loss
from input_data import input_data
import os
from os.path import exists
import random
import cv2

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description="Inputs to the code")

    parser.add_argument("--input_record_file",type=str,default='/home/saivinay/Documents/jipmer-crowd-analysis/shanghai_dataset/train.tfrecords',help="path to TFRecord file with training examples")
    parser.add_argument("--batch_size",type=int,default=16,help="Batch Size")
    parser.add_argument("--log_directory",type = str,default='./log_dir',help="path to tensorboard log")
    parser.add_argument("--ckpt_savedir",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/checkpoints',help="path to save checkpoints")
    parser.add_argument("--load_ckpt",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/checkpoints',help="path to load checkpoints from")
    parser.add_argument("--save_freq",type = int,default=50,help="save frequency")
    parser.add_argument("--display_step",type = int,default=1,help="display frequency")
    parser.add_argument("--summary_freq",type = int,default=50,help="summary writer frequency")
    parser.add_argument("--no_iterations",type=int,default=50000,help="number of iterations for training")
    parser.add_argument("--learning_rate",type=float,default=0.001,help="learning rate for training")
    parser.add_argument("--summary_path",type=str,default='/home/saivinay/Documents/jipmer-crowd-analysis/summary',help="path to tensorboard summary")
    parser.add_argument("--validation_dir",type=str,default='/home/saivinay/Documents/jipmer-crowd-analysis/shanghai_dataset/part_A/test_data')

    args = parser.parse_args()

    
    
    learning_rate = args.learning_rate
    batch_size  = args.batch_size

    
    
    # For training on workstation
    gpu_options = tf.GPUOptions(allow_growth = True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options , allow_soft_placement = True , log_device_placement = False))


    # initializing placeholders
    # x = tf.placeholder(dtype=tf.float32,shape=[None,224,224,3])
    # y_count = tf.placeholder(dtype=tf.float32,shape=[None, 1])
    # y_heatmap = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3]) ## SHape?

    
    TFRecord_file = args.input_record_file
    iterator = input_data(TFRecord_file,batch_size=args.batch_size)           # defining an iterator to iterate our the dataset
    images,labels,count = iterator.get_next()                                 # gives the next batch of data

    
    global_step_tensor = tf.train.get_or_create_global_step()       

    
    model = crowd(images)
    loss,loss_heatmap,loss_count = calculate_loss(model, labels, count)       # calculating losses    
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss,global_step=global_step_tensor) 

    
    init = tf.global_variables_initializer()                                  # for initializing global variables


    tf.summary.scalar('Loss',tf.reduce_mean(loss))                            # summary for training_loss
    # tf.summary.scalar('predicted_count', tf.reduce_mean(count))             # summary for count
    # tf.summary.scalar('Loss_heatmap',tf.reduce_mean(loss_heatmap))          # summary for training_loss_heatmap
    # tf.summary.scalar('Loss_count',tf.reduce_mean(loss_count))              # summary for training_loss_count
    
    
    saver = tf.train.Saver()                                                  # creating a saver

    
    
    
    with tf.Session() as sess:

        sess.run(init)                                                        # initializing all global variables
        
        
        # If checkpoints exist loading them and training
        if args.load_ckpt is not None:
            if exists(args.load_ckpt):
                if tf.train.latest_checkpoint(args.load_ckpt) is not None:
                    tf.logging.info("Loading checkpoint from "+ tf.train.latest_checkpoint(args.load_ckpt))
                    saver.restore(sess,tf.train.latest_checkpoint(args.load_ckpt))
                else:
                    tf.logging.info("Checkpoints not found - Training from scratch")
        else:
            tf.logging.info('Training from scratch')

        
        
        
        # Training begins
        while True:                                 

            # Running optimizer every iteration    
            global_step,_  = sess.run([global_step_tensor,optimizer])
            
            print(global_step)
            # if global_step%(args.summary_freq) == 0:
                # print("Loss at iteration :", '%04d' %global_step, " is ",loss_val )
                # summ_writer.add_summary(summary_val,global_step)                      # adding summary
               
            
            if global_step%(args.save_freq/50)==0:   

                loss_val,loss_heatmap_val,loss_count_val = sess.run([loss,loss_heatmap,loss_count])
                f"Loss : {loss_val},Loss_heatmap : {loss_heatmap_val},Loss_count : {loss_count_val}, iteration : {str(global_step)} "     # These are training losses

                # saving the model
                saver.save(sess,args.ckpt_savedir,global_step=tf.train.get_global_step())



                # Evaluation and writing into tensorboard
                random_image_path = random.choice(os.listdir(os.path.join(args.validation_dir,"images")))
                
                image = cv2.imread(os.path.join(args.validation_dir,"images/",random_image_path))
                image = cv2.resize(image, (224,224))
                ground_truth_heatmap = random_image_path.replace('.jpg','.npy').replace('images','labels').replace('IMG_','LAB_')
                ground_truth_count =  random_image_path.replace('.jpg','.npy').replace('images','count').replace('IMG_','COUNT_')


                x = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
                model_val = crowd(x)
                
                heatmap,count = sess.run(model_val.output , feed_dict={x : image[None,:,:,:]})
                Loss,Loss_heatmap,Loss_count = sess.run(calculate_loss(model_val,ground_truth_heatmap,ground_truth_count) , feed_dict={x : image[None,:,:,:]})
                
                
                tf.summary.image('predicted_heatmap',heatmap)
                tf.summary.image('actual_heatmap',ground_truth_heatmap)
                tf.summary.scalar('predicted_count',count)
                tf.summary.scalar('actual_count',ground_truth_count)
                tf.summary.scalar('Loss',Loss)
                tf.summary.scalar('Loss_heatmap',Loss_heatmap)
                tf.summary.scalar('Loss_count',Loss_count)
        
                summ_writer = tf.summary.FileWriter(args.summary_path,sess.graph)    # path for summary
                summary = tf.summary.merge_all()                                     # merges all the summary collected in the default graph.    summary = tf.summary.merge_all()       
            
                summary_val = sess.run(summary)                               # no need of feeding as they are not placeholders

                summ_writer.add_summary(summary_val,global_step)                     # adding summary
            
            
            
            if np.floor(global_step) == args.no_iterations:
                break
    


'''
def perform_validation():

    # saver = tf.train.Saver() 

    x = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
    model = crowd(x)
    heatmap, count = model.output

    random_image_path = random.choice(x for x in os.listdir(os.path.join(args.validation_dir,"images"))
                                        if os.path.isfile(os.path.join("path", x)))
    ground_truth_heatmap = random_image_path.replace('.jpg','.npy').replace('images','labels').replace('IMG_','LAB_')
    ground_truth_count =  random_image_path.replace('.jpg','.npy').replace('images','count').replace('IMG_','COUNT_')
    

    loss = calculate_loss(model,ground_truth_heatmap,ground_truth_count)


    tf.summary.image('predicted_heatmap',heatmap)
    tf.summary.image('actual_heatmap',ground_truth_heatmap)
    tf.summary.scalar('predicted_count',count)
    tf.summary.scalar('actual_count',ground_truth_count)
    tf.summary.scalar('Loss',loss)


    with tf.Session() as sess:
        summ_writer = tf.summary.FileWriter(args.summary_path,sess.graph)    # path for summary
        summary = tf.summary.merge_all()                                     # merges all the summary collected in the default graph.    summary = tf.summary.merge_all()       

        # saver.restore(sess,tf.train.latest_checkpoint(args.load_ckpt))

        image = cv2.imread(random_image_path)
        image = cv2.resize(image, (224,224))

        # [heatmap_img , count_val] = sess.run(model.output, feed_dict={x: image[None, :, :, :]}) # is this required ?
        loss_val,summary_val = sess.run([loss,summary])  # no need of feeding as they are not placeholders

        summ_writer.add_summary(summary_val,global_step)                  # adding summary

'''