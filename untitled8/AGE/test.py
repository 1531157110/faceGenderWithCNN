import  numpy as np
import tensorflow as tf
import age_model
import cv2
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

f1= open('trainImagesAndLabels.txt','r')
data_set = f1.read().splitlines()
f1.close()

batchsize = 32
print(len(data_set)//10)
sess =tf.Session()
with tf.compat.v1.variable_scope('train_age') as scope:
    xs = tf.compat.v1.placeholder(tf.float32,[None,227,227,3],name='xInput')
    keep_drop = tf.compat.v1.placeholder(tf.float32)

    y, variables = age_model.convolutional(xs,keep_drop)

ys = tf.compat.v1.placeholder(tf.float32,[None,8],name='yOutput')

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
init =tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver(variables)
saver.restore(sess,"savers/train_age40.ckpt")
print('模型恢复成功')
x = 0
for i in range(len(data_set) // batchsize):
    age_images, target = get_one_batch(flag=i, batch=batchsize, data_set=data_set)
    accur = sess.run(accuracy, feed_dict={xs: age_images, ys: target, keep_drop: 1.0})
    x += accur
    print(i,'  ',x/(i+1))

    #sess.run(y,feed_dict={xs:age_images,keep_drop:1.0}).flatten().tolist()

# print(x/347)
sess.close()