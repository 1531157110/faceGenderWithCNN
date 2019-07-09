from PIL import Image, ImageTk #导入模块，用以读取图片
from tkinter import filedialog #文本对话选择框
import tkinter as tk           #用以制作简易界面
import cv2                     #图片裁剪时所用
import getGenderModel          #性别预测模型
#from age_predict import predict_age

from save import getTrainingData    #自定义python文件，引入从Webcam上捕捉人脸的函数
#from matplotlib import pyplot as plt   #测试绘制图片
import tensorflow as tf
import untitled8.age_model
import numpy as np
#裁剪图片
def resize(w_box, h_box, pil_image):
    w, h = pil_image.size
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)

#单击选择按钮时，回调的函数，用以选择图片和显示图片
def select():
    #定义全局变量
    global lb_text      #文本显示标签，显示当前图片路径
    global imLabel      #显示原图片的标签
    global imLabel2     #显示剪切后留下的头像（人脸）
    global the_image    #存储读取的图片对象
    global the_image2   #存储裁剪后的头像（人脸）对象
    global filename

    filename = tk.filedialog.askopenfilename()      #获取图像路径

    if filename != '':
        lb_text.configure(text="您选择的文件是：\n" + filename,justify='left',width=60,height=8,anchor = 'w')
        the_image = cv2.imread(filename)
        #测试
        # print(the_image)
        # plt.imshow(the_image)
        # plt.show()
        im_temp = Image.open(filename)
        im_temp = resize(250, 250, im_temp)     #对图片进行裁剪
        im_show1 = ImageTk.PhotoImage(im_temp)

        the_image2 = getGenderModel.getFace(the_image)  #获取图像中的人脸
        face = cv2.resize(the_image2, (250, 250,))
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        im_show2 = ImageTk.PhotoImage(face)

        imLabel.configure(image=im_show1)
        imLabel.image = im_show1        #为了使标签刷新后能成功显示图像，需要保持依赖
        imLabel.pack()

        imLabel2.configure(image=im_show2)
        imLabel2.image = im_show2
        imLabel2.pack()
    else:
        lb_text.configure(text="您没有选择任何文件")

#预测性别
def callback_predict_gender():
    result = getGenderModel.getGenderForecast(the_image)
    if result == 0:
        var.set("女")
    else:
        var.set("男")

#预测年龄
age_classes =['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
sess_age = tf.Session()
with tf.compat.v1.variable_scope('train_age') as scope:
    xs = tf.compat.v1.placeholder(tf.float32, [None, 227, 227, 3], name='xInput')
    keep_drop = tf.compat.v1.placeholder(tf.float32)
    y, variables = untitled8.age_model.convolutional(xs, keep_drop)
init = tf.global_variables_initializer()
sess_age.run(init)
saver1 = tf.train.Saver(variables)
saver1.restore(sess_age, "savers/train_age.ckpt")
print('模型恢复成功')

def callback_predict_age(file_path):
    print(file_path)
    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    age_images = cv2.resize(image, (227, 227))
    xxi = np.array(age_images, dtype='float32') / 255.0
    xxi = xxi.reshape([1, 227, 227, 3])
    ap = sess_age.run(y, feed_dict={xs: xxi, keep_drop: 1.0}).flatten().tolist()
    ap1 = np.array(ap)
    label = np.argmax(ap1)
    print(label)
    res = age_classes[label]
    print(res)
    # print(result)
    # sess.close()
    var2.set(str(res))

#保存图片
def saveImg():
    filename = tk.filedialog.asksaveasfilename()
    print(the_image2)
    cv2.imwrite(filename, the_image2)

#从Webcam上获取图片
def custom_image():
    getTrainingData('getTrainData', 0, 'training_data_me/', 100)  # 注意这里的training_data_xx 文件夹就在程序工作目录下

if __name__ == '__main__':

    #定义全局变量
    global the_image
    global the_image2
    global root         #主框架
    global frame_left   #主框架中左边一侧，用以存储按钮及预测值
    global frame_top    #主框架上侧，用以显示原图像
    global frame_bottom #用以显示头像（脸部）
    global filename

    the_image = 0
    the_image2 = []
    filename =''

    root = tk.Tk()
    root.geometry("700x700+100+0")
    root.title("Gender&Age Forecast System")
    frame_left = tk.Frame(root)
    frame_top = tk.Frame(root)
    frame_bottom = tk.Frame(root)

    #文本标签
    global lb_text
    lb_text = tk.Label(frame_left, text='没有文件！！', font=('ariel', 10, 'bold'), bd=16, fg="steel blue")
    lb_text.pack()

    #选择图像的按钮
    btn_select_img = tk.Button(frame_left, padx=16, pady=8, bd=10, fg="black", font=('ariel', 16, 'bold'), width=10, text="选择图片",
                 bg="powder blue", command=select)
    btn_select_img.pack()

    var = tk.StringVar()
    var.set("man")

    #预测性别的按钮
    btn_predict_gender = tk.Button(frame_left, padx=16, pady=8, bd=10, fg="black", font=('ariel', 16, 'bold'), width=10, text="预测性别",
                     bg="powder blue", command=callback_predict_gender)
    btn_predict_gender.pack()

    text1 = tk.Label(frame_left, textvariable=var, font=('ariel', 16, 'bold'), bd=16, fg="steel blue")
    text1.pack()

    var2 = tk.StringVar()
    var2.set("20")

    #预测年龄的按钮
    btn_predict_age = tk.Button(frame_left, padx=16, pady=8, bd=10, fg="black", font=('ariel', 16, 'bold'), width=10, text="预测年龄",
                     bg="powder blue", command=lambda: callback_predict_age(file_path=filename))

    btn_predict_age.pack()
    text2 = tk.Label(frame_left, textvariable=var2, font=('ariel', 16, 'bold'), bd=16, fg="steel blue")
    text2.pack()

    #保存图像的按钮
    btn_save_img = tk.Button(frame_left, padx=16, pady=8, bd=10, fg="black", font=('ariel', 16, 'bold'), width=10, text="保存头像",
                     bg="powder blue", command=saveImg)
    btn_save_img.pack()

    #自定义图像（webcam）
    btn_save_img = tk.Button(frame_left, padx=16, pady=8, bd=10, fg="black", font=('ariel', 16, 'bold'), width=10,text="打开摄像头",
                     bg="powder blue", command=custom_image)
    btn_save_img.pack()

    im = Image.open("testImg/mayun.jpg")
    im = resize(250, 250, im)
    img = ImageTk.PhotoImage(im)
    # print(img)
    #定义原图片标签
    im_title_original = tk.Label(frame_top, text='原图片', font=('ariel', 20, 'bold'), bd=16, fg="steel blue")
    im_title_original.pack()

    global imLabel
    imLabel = tk.Label(frame_top, image=img, width=250, height=250, bg='#F0FFFF')
    imLabel.pack(side=tk.RIGHT)

    #默认打开的图片及裁剪后的头像
    im2 = Image.open("testImg/mayun.jpg")
    im2 = resize(250, 250, im2)
    img2 = ImageTk.PhotoImage(im2)
    # print(img)

    #定义头像标签
    im_title_head = tk.Label(frame_bottom, text='头像', font=('ariel', 20, 'bold'), bd=16, fg="steel blue")
    im_title_head.pack()

    global imLabel2
    imLabel2 = tk.Label(frame_bottom, image=img2, width=250, height=250, bg="#F0FFFF")
    imLabel2.pack(side=tk.RIGHT)

    frame_left.pack(side=tk.LEFT)
    frame_top.pack(side=tk.TOP)
    frame_bottom.pack(padx=10, pady=10, side=tk.BOTTOM)
    root.mainloop()
