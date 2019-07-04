import  keras
from keras.models import load_model
import cv2
import numpy as np
CASE_PATH = "C:/Users/li/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASE_PATH)
face_recognition_model = keras.Sequential()
MODEL_PATH = 'face_model.h5'
face_recognition_model = load_model(MODEL_PATH)

def resize_without_deformation(image, size = (32, 32)):
    height, width,_ = image.shape
    longest_edge = max(height, width)
    top, bottom, left, right = 0, 0, 0, 0
    if height < longest_edge:
        height_diff = longest_edge - height
        top = int(height_diff / 2)
        bottom = height_diff - top
    elif width < longest_edge:
        width_diff = longest_edge - width
        left = int(width_diff / 2)
        right = width_diff - left
    image_with_border = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = [0, 0, 0])
    resized_image = cv2.resize(image_with_border, size)
    return resized_image


#输入的shape为（宽,高,3），函数自动切分出图片有头部分，用于预测
#返回值为<class 'numpy.ndarray'>类型,shape为(1,)，只有一个元素，0表示女，1表示男
def getGenderForecast(img):
    #加载卷积神经网络模型：
    IMAGE_SIZE = 32
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.2,
                                          minNeighbors=5,
                                          minSize=(
                                          2, 2), )  # 根据检测到的坐标及尺寸裁剪、无形变resize、并送入模型运算，得到结果后在人脸上打上矩形框并在矩形框上方写上识别结果：
    for (x, y, width, height) in faces:
        # cv2.imshow('img', image[y:y + height, x:x + width])
        img = img[y:y + height, x:x + width]
        img = resize_without_deformation(img)
        img = img.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        img = np.asarray(img, dtype=np.float32)
        img /= 255.0
        result = face_recognition_model.predict_classes(img)
    return result

if __name__ == '__main__':
    img=cv2.imread('21-2m.jpg')
    result=getGenderForecast(img)
    print(type(result))
    print(np.shape(result))
    print(result)
