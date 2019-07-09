import tensorflow as tf
import numpy as np
import get_age_set as gt
import tensorflow.contrib.layers

# 卷积层 包括一个输入x，权值矩阵w，过滤器的数量和shape.padding参数，scope名称
def convLayer(x, filterHeight, filterWeight, strideX, strideY, filterNumber, name=None, padding='SAME'):
    channels = int(x.shape[-1])
    with tf.variable_scope(name) as scope:
        w = tf.Variable(tf.truncated_normal([filterHeight, filterWeight, channels, filterNumber], stddev=0.1),
                        name='weight')
        b = tf.Variable(tf.constant([filterNumber], dtype='float'), name='bias')
        w_conv = tf.nn.conv2d(x, w, [1, strideX, strideY, 1], padding=padding) + b
        return tf.nn.sigmoid(w_conv, name=scope.name)

def poolLayer(x, poolHeight, poolWeight, strideX, strideY, name=None, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, poolHeight, poolWeight, 1], strides=[1, strideX, strideY, 1], padding=padding,
                          name=name)

def flatLayer(x, inputNodesNum, outputNodesNum, name=None):
    with tf.variable_scope(name) as scope:
        w = tf.Variable(tf.truncated_normal([inputNodesNum, outputNodesNum]), name='weight')
        b = tf.Variable(tf.constant([outputNodesNum], dtype='float'), name='bias')
        out = tf.matmul(x, w) + b
        return tf.nn.sigmoid(out)

def drop_out(x, keep_prob, name=None):
    return tf.nn.dropout(x, keep_prob, name=name)


def ageModel(x, keep_prob):

    X = tf.reshape(x, [-1, 227, 227, 3])
    conv1_1 = convLayer(x=X, filterHeight=7, filterWeight=7, strideX=1,strideY=1, filterNumber=64,name='conv1_1')
    conv1_2 = convLayer(conv1_1, 7, 7, 1, 1, 64, 'conv1_2')
    pool_1 = poolLayer(conv1_2, 2, 2, 2, 2, 'pool_1')
    ''' 
        (1, 227, 227, 64) conv13
        (1, 227, 227, 64) conv2 
         (1, 114, 114, 64) pool1
        '''
    conv2_1 = convLayer(pool_1, 3, 3, 1, 1, 128, 'conv2_1')
  #  conv2_2 = convLayer(conv2_1, 3, 3, 1, 1, 128, 'conv2_2')
    pool_2 = poolLayer(conv2_1, 2, 2, 2, 2, 'pool_2')
    # pool_2 = poolLayer(conv2_2, 2, 2, 2, 2, 'pool_2')
    '''(1, 114, 114, 128)
    (1, 114, 114, 128)
    (1, 57, 57, 128)'''




    conv3_1 = convLayer(pool_2, 3, 3, 1, 1, 256, 'conv3_1')
    #conv3_2 = convLayer(conv3_1, 3, 3, 1, 1, 256, 'conv3_2')
   # conv3_3 = convLayer(conv3_2, 3, 3, 1, 1, 256, 'conv3_3')
    #conv3_4 = convLayer(conv3_3, 3, 3, 1, 1, 256, 'conv3_4')
    pool_3 = poolLayer(conv3_1, 2, 2, 2, 2, 'pool_3')
    # pool_3 = poolLayer(conv3_4, 2, 2, 2, 2, 'pool_3')
    '''
    (1, 57, 57, 256)
    (1, 57, 57, 256)
    (1, 57, 57, 256)
    (1, 57, 57, 256)
    (1, 29, 29, 256)
        '''


    conv4_1 = convLayer(pool_3, 3, 3, 1, 1, 512, 'conv4_1')
    #conv4_2 = convLayer(conv4_1, 3, 3, 1, 1, 512, 'conv4_2')
    # conv4_3 = convLayer(conv4_2, 3, 3, 1, 1, 512, 'conv4_3')
    # conv4_4 = convLayer(conv4_3, 3, 3, 1, 1, 512, 'conv4_4')
    # pool_4 = poolLayer(conv4_4, 2, 2, 2, 2, 'pool_4')
    pool_4 = poolLayer(conv4_1, 2, 2, 2, 2, 'pool_4')
    '''(1, 29, 29, 512)
        (1, 29, 29, 512)
        (1, 29, 29, 512)
        (1, 29, 29, 512)
        (1, 15, 15, 512)
    '''
    conv5_1 =convLayer(pool_4,3,3,1,1,512,'conv5_1')
    #conv5_2 =convLayer(conv5_1,3,3,1,1,512,'conv5_2')
    # conv5_3 =convLayer(conv5_2,3,3,1,1,512,'conv5_3')
    # conv5_4 =convLayer(conv5_3,3,3,1,1,512,'conv5_4')
    # pool_5 =poolLayer(conv5_4,2,2,2,2,'pool_5')
    pool_5 = poolLayer(conv5_1, 2, 2, 2, 2, 'pool_5')
    '''
    (1, 15, 15, 512)
    (1, 15, 15, 512)
    (1, 15, 15, 512)
    (1, 15, 15, 512)
    (1, 8, 8, 512)'''

    # print(conv5_1.shape)
    # print(conv5_2.shape)
    # print(conv5_3.shape)
    # print(conv5_4.shape)
    # print(pool_5.shape)

    flat_input = tf.reshape(pool_5, [-1, 8 * 8* 512])
    flat1 = flatLayer(flat_input, 8 * 8 * 512, 4096, 'flatLayer1')
    drop1 = drop_out(flat1, keep_prob=keep_prob,name='drop1')

    flat2 = flatLayer(drop1, 4096, 1024, 'flatLayer2')
    drop2 = drop_out(flat2, keep_prob=keep_prob,name= 'drop2')

    wfc3 = tf.Variable(tf.truncated_normal([1024, 8],stddev=0.1))

    sss = tf.matmul(drop2,wfc3)
    output = tf.nn.softmax(sss)
    # print(output.shape)
    #(1*8)
    return output

#
# f1= open('trainImagesAndLabels.txt','r')
# data_set = f1.read().splitlines()
# f1.close()
# age_images,target =gt.get_one_batch(flag=0,batch=128,image_set=data_set)
# a =age_images[0,:,:,:]
# print(a.shape)
# b=ageModel(age_images,0.5)