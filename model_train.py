import argparse
import tensorflow as tf
import numpy as np
from resnet_crowd import crowd,calculate_loss
from input_data import input_data
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inputs to the code")

    parser.add_argument("--input_record_file",type=str,default='/home/saivinay/Documents/jipmer-crowd-analysis/shanghai_dataset/train.tfrecords',help="path to TFRecord file with training examples")
    parser.add_argument("--batch_size",type=int,default=16,help="Batch Size")
    parser.add_argument("--log_directory",type = str,default='./log_dir',help="path to tensorboard log")
    parser.add_argument("--ckpt_savedir",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/checkpoints/',help="path to save checkpoints")
    parser.add_argument("--load_ckpt",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/checkpoints/',help="path to load checkpoints from")
    parser.add_argument("--save_freq",type = int,default=100,help="save frequency")
    parser.add_argument("--display_step",type = int,default=1,help="display frequency")
    parser.add_argument("--summary_freq",type = int,default=100,help="summary writer frequency")
    parser.add_argument("--no_epochs",type=int,default=5,help="number of epochs for training")
    parser.add_argument("--learning_rate",type=float,default=0.001,help="learning rate for training")

    args = parser.parse_args()

    learning_rate = args.learning_rate
    batch_size  = args.batch_size
    num_epochs = args.no_epochs

    # initializing placeholders
    # x = tf.placeholder(dtype=tf.float32,shape=[None,224,224,3])
    # y_count = tf.placeholder(dtype=tf.float32,shape=[None, 1])
    # y_heatmap = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3]) ## SHape?

    TFRecord_file = args.input_record_file
    iterator = input_data(TFRecord_file,batch_size=args.batch_size)     # defining an iterator to iterate our the dataset
    images,labels,count = iterator.get_next()                           # gives the next batch of data

    global_step_tensor = tf.train.get_or_create_global_step()       

    model = crowd(images)
    loss = calculate_loss(model, labels, count)                         # calculating loss    
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss,global_step=global_step_tensor) # optimizer

    init = tf.global_variables_initializer()                            # for initializing global variables

    summary_path = '/home/saivinay/Documents/jipmer-crowd-analysis/summary'

    tf_loss_summary = tf.summary.scalar('Loss',tf.reduce_mean(loss))    # summary for loss
    tf.summary.scalar('Count', tf.reduce_mean(count))                   # summary for count

    saver = tf.train.Saver()                                            # creating a saver

    with tf.Session() as sess:
        print(loss)
        summ_writer = tf.summary.FileWriter(summary_path,sess.graph)    # path for summary
        summary = tf.summary.merge_all()                                # merges all the summary collected in the default graph.
        sess.run(init)                                                  # initializing all global variables
        
        for epoch in range(num_epochs):                                 # epoches is number of training examples/batch size  

            global_step,_,loss_val,summary_val  = sess.run([global_step_tensor,optimizer,loss,summary])
            
            if global_step%(args.summary_freq) == 0:
                # print("Loss at epoch", '%04d' %(epoch+1), "is ",loss_val )
                f"Loss : {loss_val}, epoch : {epoch} "
                summ_writer.add_summary(summary_val,epoch)                  # adding summary

            if global_step%(args.save_freq)==0:
                saver.save(sess,args.ckpt_savedir,global_step=tf.train.get_global_step())

        
        
    



            



