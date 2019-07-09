import  numpy as np
import tensorflow as tf
import age_model
import cv2

age_classes =['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
def Tssss(path):
    image = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
    age_images = cv2.resize(image,(227,227))
    xxi = np.array(age_images,dtype='float32')/255.0
    sess =tf.Session()
    with tf.compat.v1.variable_scope('train_age') as scope:
        xs = tf.compat.v1.placeholder(tf.float32,[None,227,227,3],name='xInput')
        keep_drop = tf.compat.v1.placeholder(tf.float32)
        y, variables = age_model.convolutional(xs,keep_drop)
    init =tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(variables)
    saver.restore(sess,"savers/train_age.ckpt")
    print('模型恢复成功')

    xxi = xxi.reshape([1,227,227,3])
    ap = sess.run(y,feed_dict={xs:xxi,keep_drop:1.0}).flatten().tolist()
    ap1 =np.array(ap)
    label = np.argmax(ap1)
    result = age_classes[label]

    #print(result)
    sess.close()
    return result

# #
f1= open('testImagesAndLabels.txt','r')
data_set = f1.read().splitlines()
f1.close()

aaa=data_set[0].split(' ')[0]
print(aaa)
#
rs = Tssss(path=aaa)
print(rs)