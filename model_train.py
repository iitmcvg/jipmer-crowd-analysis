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
    parser.add_argument("--validation_record_file",type=str,default='/home/saivinay/Documents/jipmer-crowd-analysis/shanghai_dataset/test.tfrecords',help="path to TFRecord file with test examples")
    parser.add_argument("--batch_size",type=int,default=8,help="Batch Size")
    parser.add_argument("--log_directory",type = str,default='./log_dir',help="path to tensorboard log")
    parser.add_argument("--ckpt_savedir",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/checkpoints/',help="path to save checkpoints")
    parser.add_argument("--load_ckpt",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/checkpoints/',help="path to load checkpoints from")
    parser.add_argument("--save_freq",type = int,default=50,help="save frequency")
    parser.add_argument("--display_step",type = int,default=1,help="display frequency")
    parser.add_argument("--summary_freq",type = int,default=50,help="summary writer frequency")
    parser.add_argument("--no_iterations",type=int,default=50000,help="number of iterations for training")
    parser.add_argument("--learning_rate",type=float,default=0.001,help="learning rate for training")
    parser.add_argument("--summary_path",type=str,default='/home/saivinay/Documents/jipmer-crowd-analysis/summary/',help="path to tensorboard summary")

    args = parser.parse_args()

    
        
    
    # For training on workstation
    gpu_options = tf.GPUOptions(allow_growth = True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options , allow_soft_placement = True , log_device_placement = False))

    global_step_tensor = tf.train.get_or_create_global_step()               
    # initializing placeholders
    x = tf.placeholder(dtype=tf.float32,shape=[None,224,224,3])
    y_count = tf.placeholder(dtype=tf.float32,shape=[None, 1])
    y_heatmap = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1]) 

    # defining model,loss,optimizer 
    model = crowd(x)
    loss,loss_heatmap,loss_count = calculate_loss(model, y_heatmap,y_count )       # calculating losses    
    optimizer = tf.train.AdagradOptimizer(args.learning_rate).minimize(loss,global_step=global_step_tensor) 


    # data for training
    TFRecord_file = args.input_record_file
    iterator = input_data(TFRecord_file,batch_size=args.batch_size)           # defining an iterator to iterate our the dataset
    images,labels,count = iterator.get_next()                                 # gives the next batch of data
    # data for validation
    test_TFrecord_file = args.validation_record_file
    val_iterator = input_data(test_TFrecord_file,batch_size=args.batch_size)
    val_images,val_labels,val_count = val_iterator.get_next()
    
    
    init = tf.global_variables_initializer()                                  # for initializing global variables


    # tf.summary.scalar('Loss',tf.reduce_mean(loss))                          # summary for training_loss
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
                    print(" Checkpoints exits loading them ")
                    saver.restore(sess,tf.train.latest_checkpoint(args.load_ckpt))
                else:
                    tf.logging.info("Checkpoints doesnt exist")
        else:
            tf.logging.info" Checkpoints not found, training from scratch ")
        
        
        
        ####### Training starts #######
        while True:                                 
            
            ## Running optimizer every iteration    
            # making data numpy arrays to feed 
            images_sess = sess.run(images)
            labels_sess = sess.run(labels)
            count_sess = sess.run(count)
            
            #*** feed dict requires numpy arrays
            global_step,_,predicted_count  = sess.run([global_step_tensor,optimizer,model.output[1]],feed_dict={x:images_sess,y_heatmap:labels_sess,y_count:count_sess})
            # No need to run as we are not printing 
            # loss_val,loss_heatmap_val,loss_count_val = sess.run([loss,loss_heatmap,loss_count],feed_dict={x:img_sess,y_heatmap:labels_sess,y_count:count_sess})
            '''
            print("loss_val ",sess.run(tf.reduce_mean(loss_val))  )  # reduced mean for averaging over batch size
            print("loss_count_val ",loss_count_val)
            print("predicted_count",predicted_count)
            print("actual count",sess.run(count))
            print("loss_heatmap_val ",sess.run(tf.reduce_mean(loss_heatmap_val)))
            # f"Loss : {loss_val},Loss_heatmap : {loss_heatmap_val},Loss_count : {loss_count_val}, iteration : {str(global_step)} "     # These are training losses
            print(global_step)
            '''
            # if global_step%(args.summary_freq) == 0:
                # print("Loss at iteration :", '%04d' %global_step, " is ",loss_val )
                # summ_writer.add_summary(summary_val,global_step)                      # adding summary
            




            ### Evaluation and writing summary to tensorboard
            if global_step%(args.save_freq//50)==0:   
                
                ## Calculating the training losses
                loss_val,loss_heatmap_val,loss_count_val = sess.run([loss,loss_heatmap,loss_count],feed_dict={x:images_sess,y_heatmap:labels_sess,y_count:count_sess})
                
                print("loss_val ",sess.run(tf.reduce_mean(loss_val))  )  # reduced mean for averaging over batch size
                print("loss_count_val ",sess.run(tf.reduce_mean(loss_count_val)))
                # print("predicted_count",predicted_count)
                # print("actual count",sess.run(count))
                print("loss_heatmap_val ",sess.run(tf.reduce_mean(loss_heatmap_val)))
                # f"Loss : {loss_val},Loss_heatmap : {loss_heatmap_val},Loss_count : {loss_count_val}, iteration : {str(global_step)} "     # These are training losses
            
                
                ## Evaluation and writing into tensorboard

                # making data numpy arrays to feed             
                val_images_sess = sess.run(val_images)
                val_labels_sess = sess.run(val_labels)
                val_count_sess = sess.run(val_count)

                # print(val_count_sess)
                # print(global_step)
                # print(val_img_sess)
                # print(val_labels_sess)
                # print(val_img_sess.get_shape())


                heatmap,pred_count = sess.run(model.output , feed_dict={x:val_images_sess,y_heatmap:val_labels_sess,y_count:val_count_sess})
                Loss,Loss_heatmap,Loss_count = sess.run([loss,loss_heatmap,loss_count],feed_dict={x:val_images_sess,y_heatmap:val_labels_sess,y_count:val_count_sess})
                # print(count)
                
                ##***  must  feed only tensors for tesnsorboard 
                # tf.summary.image('predicted_heatmap',heatmap)
                # tf.summary.image('actual_heatmap',val_labels_sess)

                # global_step = tf.convert_to_tensor(global_step, np.float32)
                tf.summary.scalar('predicted_count',tf.reduce_mean(pred_count))
                tf.summary.scalar('actual_count',tf.reduce_mean(val_count))
                tf.summary.scalar('Loss',tf.reduce_mean(Loss))
                tf.summary.scalar('Loss_heatmap',tf.reduce_mean(Loss_heatmap))
                tf.summary.scalar('Loss_count',tf.reduce_mean(Loss_count))
                tf.summary.scalar('Global_step',global_step)


                summ_writer = tf.summary.FileWriter(args.summary_path,sess.graph)    # path for summary
                summary = tf.summary.merge_all()                                     # merges all the summary collected in the default graph.    summary = tf.summary.merge_all()       
                summary_val = sess.run(summary)                                      # no need of feeding as they are not placeholders
                summ_writer.add_summary(summary_val,global_step)                     # adding summary
                
                # saving the model
                saver.save(sess,args.ckpt_savedir,global_step=tf.train.get_global_step())
                print(global_step)

            #Breaking condition
            if np.floor(global_step) == args.no_iterations:
                break
    