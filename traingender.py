from keras.layers import Conv2D, MaxPooling2D
from keras.layers import  Dense, Dropout, Flatten
from keras.optimizers import SGD
from read_data import getPicLab
import numpy as np
import keras
import tensorflow as tf
# from tensorflow import keras
from keras.optimizers import SGD


IMAGE_SIZE = 32
raw_images, raw_labels = getPicLab()
# print(raw_labels[1])
raw_images, raw_labels = np.asarray(raw_images, dtype = np.float32), np.asarray(raw_labels, dtype = np.int32) #把图像转换为float类型，方便归一化
from sklearn.model_selection import  train_test_split
train_input, valid_input, train_output, valid_output =train_test_split(raw_images,raw_labels,test_size = 0.1)
train_input /= 255.0
valid_input /= 255.0

my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=6, min_delta=0.00001, monitor='val_loss')]

# face_recognition_model = keras.Sequential()
IMAGE_W=IMAGE_SIZE
IMAGE_H=IMAGE_SIZE
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), input_shape=(IMAGE_W, IMAGE_H, 3), strides=(1, 1), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(96, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.summary()

learning_rate = 0.01

decay = 1e-6

momentum = 0.8

nesterov = True

sgd_optimizer = SGD(lr = learning_rate, decay = decay,

                    momentum = momentum, nesterov = nesterov)

model.compile(loss = 'categorical_crossentropy',

                               optimizer = sgd_optimizer,

                               metrics = ['accuracy'])
model.fit(x=train_input,
          y=train_output,
          batch_size=24,
          epochs=10,
          verbose=2,
          shuffle=True,
          validation_data = (valid_input, valid_output)
          )

MODEL_PATH = 'face_model.h5'
model.save(MODEL_PATH)
print("保存成功")




















# learning_rate = 0.01
#
# decay = 1e-6
#
# momentum = 0.8
#
# nesterov = True
#
# sgd_optimizer = SGD(lr = learning_rate, decay = decay,
#
#                     momentum = momentum, nesterov = nesterov)
#
# face_recognition_model.compile(loss = 'categorical_crossentropy',
#
#                                optimizer = sgd_optimizer,
#
#                                metrics = ['accuracy'])
#
# batch_size = 24 #每批训练数据量的大小
#
# epochs =100
#
# face_recognition_model.fit(train_input, train_output,
#
#                            epochs = epochs,
#
#                            batch_size = batch_size,
#
#                            shuffle = True,
#
#                            validation_data = (valid_input, valid_output))
#
# print(face_recognition_model.evaluate(valid_input, valid_output, verbose=0))
# # for si in range(200):
# #     mm=valid_input[si]
# #     mm=mm.reshape((1,100,100,3))
# #     print(face_recognition_model.predict_classes(mm),np.argmax(valid_output[si]))
# #
#
#
# MODEL_PATH = 'face_model.h5'
#
# face_recognition_model.save(MODEL_PATH)
