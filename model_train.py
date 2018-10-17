import argparse
import tensorflow as tf
import numpy as np
from resnet_crowd import crowd,calculate_loss
from input_data import input_data
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inputs to the code")

    parser.add_argument("--input_record_file",type=str,default='/home/saivinay/Documents/crowd/train.tfrecords',help="path to TFRecord file with training examples")
    parser.add_argument("--batch_size",type=int,default=16,help="Batch Size")
    parser.add_argument("--log_directory",type = str,default='./log_dir',help="path to tensorboard log")
    parser.add_argument("--ckpt_savedir",type = str,default='/home/saivinay/Documents/crowd/checkpoints/',help="path to save checkpoints")
    parser.add_argument("--load_ckpt",type = str,default='/home/saivinay/Documents/crowd/checkpoints/',help="path to load checkpoints from")
    parser.add_argument("--save_freq",type = int,default=100,help="save frequency")
    parser.add_argument("--display_step",type = int,default=1,help="display frequency")
    parser.add_argument("--summary_freq",type = int,default=100,help="summary writer frequency")
    parser.add_argument("--no_epochs",type=int,default=5,help="number of epochs for training")
    parser.add_argument("--learning_rate",type=float,default=0.001,help="learning rate for training")

    args = parser.parse_args()


    # learning_rate = 0.01
    # batch_size = 32
    # num_epochs = 50
    learning_rate = args.learning_rate
    batch_size  = args.batch_size
    num_epochs = args.no_epochs

    # initializing placeholders
    # x = tf.placeholder(dtype=tf.float32,shape=[None,224,224,3])
    # y_count = tf.placeholder(dtype=tf.float32,shape=[None, 1])
    # y_heatmap = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3]) ## SHape?

    # model = crowd(x)

    # print (model.summary())

    # loss = calculate_loss(model,y_heatmap,y_count)

    TFRecord_file = '/home/saivinay/Documents/crowd/train.tfrecords'
    iterator = input_data(TFRecord_file,batch_size=args.batch_size)
    images,labels,count = iterator.get_next()

    model = crowd(images)
    loss = calculate_loss(model, labels, count)
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss) 

    TFRecord_file = args.input_record_file

    init = tf.global_variables_initializer() 

    PATH = '/home/saivinay/Documents/crowd/summary'

    tf_loss_summary = tf.summary.scalar('Loss',tf.reduce_mean(loss))
    tf.summary.scalar('Count', tf.reduce_mean(count))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print(loss)
        summ_writer = tf.summary.FileWriter(PATH,sess.graph)
        summary = tf.summary.merge_all()
        sess.run(init)
        
        for epoch in range(num_epochs):


            # _,loss  = sess.run([optimizer,loss],feed_dict={x:images,y_heatmap:labels,y_count:count})
            _,loss_val,summary_val  = sess.run([optimizer,loss,summary])

            print("Loss at epoch", '%04d' %(epoch+1), "is ",loss_val )
            summ_writer.add_summary(summary_val,epoch)

        
        saver.save(sess,args.ckpt_savedir)
        # meta_graph_def = saver.export_meta_graph(filename='/checkpoints/model.meta')



            



