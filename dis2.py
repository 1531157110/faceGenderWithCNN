import cv2
from contextlib import contextmanager
from read_data import getPicLab
import numpy as np
import  keras
import dlib
from read_data import resize_without_deformation
from keras.models import load_model
#加载级联分类器模型：
CASE_PATH = "C:/Users/li/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASE_PATH)
#加载卷积神经网络模型：
face_recognition_model = keras.Sequential()
MODEL_PATH = 'face_model.h5'
face_recognition_model = load_model(MODEL_PATH)
# 打开摄像头，获取图片并灰度化：

@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()
def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            # get video frame
            ret, image = cap.read()
            if not ret:
                raise RuntimeError("Failed to capture image")
            yield image

# cap = cv2.VideoCapture(0)
# ret, image = cap.read()
def getFromcamera():
    IMAGE_SIZE=32
    image_generator=yield_images()
    detector = dlib.get_frontal_face_detector()
    for image in image_generator:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 人脸检测：
        # # cv2.imshow("gray",gray)
        # print(np.shape(gray))
        # print(np.shape(image))
        faces = face_cascade.detectMultiScale(gray,
                                              scaleFactor=1.2,
                                              minNeighbors=5,
                                              minSize=(2, 2), ) # 根据检测到的坐标及尺寸裁剪、无形变resize、并送入模型运算，得到结果后在人脸上打上矩形框并在矩形框上方写上识别结果：
        for (x, y, width, height) in faces:
            # cv2.imshow('img', image[y:y + height, x:x + width])
            img = image[y:y + height, x:x + width]
            img = resize_without_deformation(img)
            img = img.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
            img = np.asarray(img, dtype=np.float32)

            # face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            img /= 255.0
            result = face_recognition_model.predict_classes(img)
            # print(np.shape(result))
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if result[0] == 1:
                cv2.putText(image, 'man', (x, y - 2), font, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(image, 'woman' , (x, y - 2), font, 0.7, (0, 255, 0), 2)
        cv2.imshow('', image)
        key = cv2.waitKey(-1) if False else cv2.waitKey(30)
        if key == 27:  # ESC
            break

def getFromFile():
    IMAGE_SIZE = 40
    raw_images, raw_labels = getPicLab
    # print(raw_labels[1])
    raw_images, raw_labels = np.asarray(raw_images, dtype=np.float32), np.asarray(raw_labels,dtype=np.int32)  # 把图像转换为float类型，方便归一化
    for si in range(240):
        mm = raw_images[si] / 255.0
        mm = mm.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        print(face_recognition_model.predict_classes(mm),si)
    # print(face_recognition_model.evaluate(raw_images/255.0, raw_labels, verbose=0))
if __name__ == '__main__':
    # getFromFile()
    getFromcamera()

