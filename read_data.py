import numpy as np
from utils import load_data
from keras.utils import np_utils
from pathlib import Path
from matplotlib import pyplot as plt
import cv2

def getPicLab():
    input_path = "outputdata.mat"
    output_path = Path(__file__).resolve().parent.joinpath("a")
    output_path.mkdir(parents=True, exist_ok=True)
    image, gender, age, _, image_size, _ = load_data(input_path)
    #X_data中存储了像素值 32*32*3
    #y_data_g中存储了性别 长度为2
    #y_data_a中存储了年龄 长度为101
    X_data = image
    y_data_g = np_utils.to_categorical(gender, 2)
    return X_data,y_data_g

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

if __name__ == '__main__':
    a,b=getPicLab()
    print(np.shape(a))
    print(np.shape(b))

