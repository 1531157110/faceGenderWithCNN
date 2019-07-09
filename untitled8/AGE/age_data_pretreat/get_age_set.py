import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
#下面函数用于获取一批数据 由于电脑内存问题 每次只能读取一部分数据
#point指导获取数据是第几批 batch是每一批数据的数量 flag*batch不能超过17393
# f1= open('image_age.txt','r')
# # data_set = list(f1.read().replace("'",'').replace('[','').replace(']','').splitlines())
# # f1.close()
import tensorflow.contrib.slim as slim
def exchange(path):
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
def get_one_batch(flag=0,batch=10,data_set=None):
    age_images = np.zeros([batch, 227, 227, 3])
    age_targets = np.zeros([batch, 8])
    begin = flag*batch
    for i in range(batch):
        '''首先获取图片的路径，然后读取图片进行放缩存入[1,width,heigth,3]的数组之中'''
        path=data_set[begin+i].split(' ')[0]
        image=exchange(path)
        age_images[i,:,:,:]=cv2.resize(image,(227,227))
        age = int(data_set[begin+i].split(' ')[2])
        age_targets[i,age]=1.0
    age_images=np.array(age_images,dtype='float32')/255.0
    age_targets=np.array(age_targets,dtype='float32')
    return age_images,age_targets


#
# age_images,age_targets=get_one_batch(image_set=data_set)
# print(age_images.shape)
# print(age_targets.shape)
# print(age_targets[0,:])
# print(age_images[0,:,:,:])
#
# t=age_images[0,:,:,:]
# b=Image.fromarray(t[:,:,0]).convert('L')
# g=Image.fromarray(t[:,:,1]).convert('L')
# r=Image.fromarray(t[:,:,2]).convert('L')
# #b g r
# image_b = cv2.merge([b, np.zeros(b.shape, np.uint8), np.zeros(b.shape, np.uint8)])
# T2=Image.merge('RGB',(b,g,r))
# # plt.imshow(r)
# # plt.imshow(g)
# # plt.imshow(b)
# plt.imshow(image_b)
# plt.show()
#

#
#
#
#
# jpg_data = tf.placeholder(dtype=tf.string)
# decode_jpg = tf.image.decode_jpeg(jpg_data, channels=3)
# resize = tf.image.resize_images(decode_jpg, [227, 227])
# resize = tf.cast(resize, tf.uint8) / 255
#
# def resize_image(file_name):
#     with tf.gfile.FastGFile(repr(file_name), 'r') as f:
#         image_data = f.read()
#     with tf.Session() as sess:
#         image = sess.run(resize, feed_dict={jpg_data: image_data})
#     return image
#
# # 应该先处理好图片或使用string_input_producer
# def get_next_batch(data_set, batch_size=128,flag=0):
#     batch_x = []
#     batch_y = []
#     begin = flag*batch_size
#     for i in range(batch_size):
#         batch_x.append(resize_image(data_set[begin+i].split(',')[0]))
#         batch_y.append(data_set[begin+i].split(',')[1])
#     return batch_x, batch_y
#
# a,b=get_next_batch(data_set=data_set)
# print(a[0])
# print(b[0])