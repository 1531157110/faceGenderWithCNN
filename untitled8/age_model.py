import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.contrib.layers  as tfcl
def convolutional(x,keep_prob):
    #定义Weight变量，输入shape，返回变量的参数。其中我们使用了tf.truncted_normal产生随机变量来进行初始化：
    def weight_variable(shape):
        initial =tf.random.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    #定义biase变量，输入shape，返回变量的一些参数。其中我们使用tf.constant常量函数来进行初始化：
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    # 定义卷积操作。tf.nn.conv2d函数是Tensorflow里面的二维的卷积函数，
    # x是图片的所有参数，W是卷积层的权重，
    #然后定义步长strides=[1,1,1,1]值。strides[0]和strides[3]的两个1是默认值，意思是不对样本个数和channel进行卷积，中间两个1代表padding是在x方向运动一步，y方向运动一步，padding采用的方式实“SAME”就是0填充。

    def conv2d_1(x, W):
        return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding="SAME")  # padding="SAME"用零填充边界
    # def conv2d_2(x,W):
    #     return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
    # def conv2d_3(x,W):
    #     return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
    # def conv2d_4(x,W):
    #     return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
    # def conv2d_5(x,W):
    #     return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
    #定义池化操作
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")


    x_image =tf.reshape(x,[-1,227,227,3])

    w_conv1 =weight_variable([7,7,3,64])#第一层filter
    b_conv1=bias_variable([64])#第一层偏移量
    y_conv1 =tfcl.batch_norm(conv2d_1(x_image,w_conv1)+b_conv1,decay=0.,is_training=False)
    h_conv1 = tf.nn.sigmoid(y_conv1)
    h_pool1=max_pool_2x2(h_conv1)

    w_conv2= weight_variable([3,3,64,128])
    b_conv2 =bias_variable([128])
    y_conv2 = tfcl.batch_norm(conv2d_1(h_pool1,w_conv2)+b_conv2,decay=0.9,is_training=False)
    h_conv2 =tf.nn.sigmoid(y_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    w_conv3 = weight_variable([3,3,128,256])
    b_conv3 = bias_variable([256])
    y_conv3 = tfcl.batch_norm(conv2d_1(h_pool2,w_conv3)+b_conv3,decay=0.9,is_training=False)
    h_conv3 = tf.nn.sigmoid(y_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    w_conv4 = weight_variable([3, 3, 256, 512])
    b_conv4 = bias_variable([512])
    y_conv4 = tfcl.batch_norm(conv2d_1(h_pool3, w_conv4) + b_conv4, decay=0.9, is_training=False)
    h_conv4 = tf.nn.sigmoid(y_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

    w_conv5 = weight_variable([3, 3, 512, 512])
    b_conv5 = bias_variable([512])
    y_conv5 = tfcl.batch_norm(conv2d_1(h_pool4, w_conv5) + b_conv5, decay=0.9, is_training=False)
    h_conv5 = tf.nn.sigmoid(y_conv5)
    h_pool5 = max_pool_2x2(h_conv5)
    # print('h_conv3:', h_conv3.shape)
    # print('h_pool3:', h_pool3.shape)
    #10*10*1152 5*5*1152
    flat_input = tf.reshape(h_pool5,[-1,8*8*512])

    w_fc1 = weight_variable([8*8*512,1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.sigmoid(tf.matmul(flat_input,w_fc1)+b_fc1)
    h_fc1_drop =tf.nn.dropout(h_fc1,keep_prob)

    w_fc2 = weight_variable([1024,8])
    b_fc2 = bias_variable([8])
    y_conv =tf.matmul(h_fc1_drop,w_fc2)+b_fc2
    prediction = tf.nn.softmax(y_conv)
    return prediction,\
           [w_conv1,b_conv1,w_conv2,b_conv2,w_conv3,b_conv3,w_conv4,b_conv4,w_conv5,b_conv5,w_fc1,b_fc1,w_fc2,b_fc2]



# age_classes =['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
# sex_classes = ['f','m']
# #
# f1= open('image_age.txt','r')
# image_age = list(f1.read().replace("'",'').replace('[','').replace(']','').splitlines())
# f1.close()
#
# print('年龄训练数据集数据个数：',len(image_age))#17393个数据
#
# # 下面代码测试网络是不是通的
# age_images,age_targets=gt.get_one_batch(flag=0,batch=100,image_set=image_age)
# # traindata = slim.flatten(age_images)
# print(age_images.shape)
# print(age_targets.shape)
# print(age_targets[0,:])
# print(age_images[0,:,:,:].shape)
#
# a1 =age_images[0,:,:,:].reshape([1,227,227,3])
# print(a1.shape)
# convolutional(age_images[0,:,:,:],0.5)
#
# # # #
# xs = tf.compat.v1.placeholder(tf.float32,[None,227,227,3])
# keep_drop = tf.placeholder(tf.float32)
# y,v= convolutional(xs,keep_drop)
#
#
# with tf.compat.v1.Session() as sess:
#     init =tf.compat.v1.global_variables_initializer()
#     sess.run(init)
#     a=sess.run(y,feed_dict={xs:age_images[0:2,:,:,:],keep_drop:0.5})
#     b=sess.run(v, feed_dict={xs: age_images[0:2, :, :, :], keep_drop: 0.5})
#
# print(a.shape)
# ax = np.argmax(a)
# print('输出：',ax)
# print(len(b))
# print(b[0].shape)





#
# t=age_images[0,:,:,:]
# r=Image.fromarray(t[:,:,0]).convert('L')
# g=Image.fromarray(t[:,:,1]).convert('L')
# b=Image.fromarray(t[:,:,2]).convert('L')
# #b g r
# T2=Image.merge('RGB',(b,g,r))
# # plt.imshow(r)
# # plt.imshow(g)
# # plt.imshow(b)
# plt.imshow(T2)
# plt.show()