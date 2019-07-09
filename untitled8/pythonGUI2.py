from tkinter import *
import tkinter.filedialog
from PIL import Image,ImageTk
import cv2
import getGenderModel

the_image=0
the_image2=[]
def xz():
    filename = tkinter.filedialog.askopenfilename()
    if filename != '':
        lb.config(text = "您选择的文件是：\n"+filename)
        global the_image
        the_image=cv2.imread(filename)
        # print(the_image)
        im = Image.open(filename)
        im = resize(250, 250, im)
        img = ImageTk.PhotoImage(im)
        global the_image2
        the_image2=getGenderModel.getFace(the_image)
        face= cv2.resize(the_image2,(250, 250,))

        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        img2 = ImageTk.PhotoImage(face)
        imLabel2.config(image=img2)
        imLabel.config(image=img).pack()

        # global the_image2

    else:
        lb.config(text = "您没有选择任何文件")
def callback():
    result = getGenderModel.getGenderForecast(the_image)
    if result==0:
        var.set("女")
    else:
        var.set("男")
def callback2():
    var2.set("22")

def saveImg():
    filename = tkinter.filedialog.asksaveasfilename()
    print(the_image2)
    cv2.imwrite(filename,the_image2)

root = Tk()
root.geometry("700x700+100+0")
root.title("Gender&Age Forecast System")
frame1 = Frame(root)

frame2 = Frame(root)
frame3 = Frame(root)


def resize(w_box, h_box, pil_image):
    w, h = pil_image.size
    f1 = 1.0*w_box/w
    f2 = 1.0*h_box/h
    factor = min([f1, f2])
    width = int(w*factor)
    height = int(h*factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)

lb = Label(frame1,text = '没有文件！！',font=('ariel' ,10,'bold'), bd=16, fg="steel blue")
lb.pack()
btn=Button(frame1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=10, text="选择图片", bg="powder blue",command=xz)
btn.pack()


var = StringVar()
var.set("man")


Button1=Button(frame1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=10, text="预测性别", bg="powder blue",command=callback)
Button1.pack()

text1 = Label(frame1,textvariable=var,font=('ariel' ,16,'bold'), bd=16, fg="steel blue")
text1.pack()


var2 = StringVar()
var2.set("20")
Button2=Button(frame1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=10, text="预测年龄", bg="powder blue",command=callback2)

Button2.pack()
text2 = Label(frame1,textvariable=var2,font=('ariel' ,16,'bold'), bd=16, fg="steel blue")
text2.pack()

Button3=Button(frame1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=10, text="保存头像", bg="powder blue",command=saveImg)
Button3.pack()

im=Image.open("testImg/mayun.jpg")
im=resize(250,250,im)
img=ImageTk.PhotoImage(im)
# print(img)
imTitle=Label(frame2,text='原图片',font=('ariel' ,20,'bold'), bd=16, fg="steel blue")
imTitle.pack()
global imLabel
imLabel=Label(frame2,image=img,width=250,height=250,bg='#F0FFFF')
imLabel.pack(side=RIGHT)


im2=Image.open("testImg/mayun.jpg")
im2=resize(250,250,im2)
img2=ImageTk.PhotoImage(im2)
# print(img)
imTitle2=Label(frame3,text='头像',font=('ariel' ,20,'bold'), bd=16, fg="steel blue")
imTitle2.pack()
global imLabel2
imLabel2=Label(frame3,image=img2,width=250,height=250,bg="#F0FFFF")
imLabel2.pack(side=RIGHT)

frame1.pack(side=LEFT)
frame2.pack(side=TOP)
frame3.pack(padx=10, pady=10,side=BOTTOM)

mainloop()