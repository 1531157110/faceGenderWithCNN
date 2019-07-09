import os
import glob
import numpy as np
import random

age_classes =['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
sex_classes = ['f','m']
# 使用方法：在其他文件导入 然后加上这条语句image_age,image_sex=pretreat.get_set()
#数据集路径
pan = 'D:\\'
face_set_fold =os.path.join(pan,'AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification')

#txt文档路径 txt记载了照片的信息

fold_0_data =os.path.join(face_set_fold,'fold_0_data.txt')
fold_1_data =os.path.join(face_set_fold,'fold_1_data.txt')
fold_2_data =os.path.join(face_set_fold,'fold_2_data.txt')
fold_3_data =os.path.join(face_set_fold,'fold_3_data.txt')
fold_4_data =os.path.join(face_set_fold,'fold_4_data.txt')

#照片的路径
face_images_set = os.path.join(face_set_fold,'aligned')


def get_data(fold_x_data):
    images_age_set=[]#年龄数据集
    images_sex_set=[]#性别数据集
    with open(fold_x_data,'r') as f:

        f1= f.read()

        aa =True#一个标志位 跳过第一行
        for line in f1.splitlines():
            temp = []
            if aa ==True:
                aa=False
                continue
            # line = x.split()
            temp.append(line.split('\t')[0])#user_id
            temp.append(line.split('\t')[1])#original_image
            temp.append(line.split('\t')[3])#age
            temp.append(line.split('\t')[4])#性别 f：女 m：男

            # print(temp)

            image_path =os.path.join(face_images_set,temp[0])

            #xpath = os.path.join(image_path,temp[1])
            if os.path.exists(image_path):
                # print('ssss')
                images =glob.glob(image_path+'/*.jpg')
                for image in images:
                    # print(image)
                    if temp[1] in image:
                        image = image.replace('\\', '/')
                        break
                if temp[2] in age_classes :
                            # print(image)
                    images_age_set.append([image,age_classes.index(temp[2])])
                if temp[3] in sex_classes:
                    images_sex_set.append([image,sex_classes.index(temp[3])])

    return images_age_set,images_sex_set
def get_set():

    #获取五个文档的所有图片的路径，并对每个路径标记，年龄数据集标记位（图片路径，年龄标记）性别数据集（图片路径，性别标记）
    image_age0,image_sex0=get_data(fold_0_data)
    image_age1,image_sex1=get_data(fold_1_data)
    image_age2,image_sex2=get_data(fold_2_data)
    image_age3,image_sex3=get_data(fold_3_data)
    image_age4,image_sex4=get_data(fold_4_data)

    #所有数据集连起来
    image_age = image_age0+image_age1+image_age2+image_age3+image_age4
    image_sex = image_sex0+image_sex1+image_sex2+image_sex3+image_sex4

    random.shuffle(image_age)
    random.shuffle(image_sex)
    with open('image_age.txt','w+',encoding='UTF-8') as w1:
        for i in range(len(image_age)):
            w1.write(str(image_age[i]))
            w1.write('\n')
      #  w1.write(str(image_age))

    with open('image_sex.txt','w+',encoding='UTF-8') as w2:
        #w2.write(str(image_sex))
        for i in range(len(image_sex)):
            w2.write(str(image_sex[i]))
            w2.write('\n')
    #return image_age,image_sex

#image_age,image_sex=get_set()
# print(image_age[0])
# print(image_sex[0])









