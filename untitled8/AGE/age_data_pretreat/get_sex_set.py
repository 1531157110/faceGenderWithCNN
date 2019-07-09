import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

age_classes =['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
sex_classes = ['f','m']

#打开性别训练集的文本文档
f1= open('image_sex.txt','r')
image_sex = list(f1.read().replace("'",'').replace('[','').replace(']','').splitlines())
f1.close()

print('性别训练数据集数据个数：',len(image_sex))#17492个数据

#下面函数用于获取一批数据 由于电脑内存问题 每次只能读取一部分数据
#point指导获取数据是第几批 batch是每一批数据的数量 flag*batch不能超过17492
def get_one_batch(flag=0,batch=100):
    age_images = np.zeros([batch, 227, 227, 3])
    age_targets = np.zeros([batch, 2])

    begin = flag*batch
    for i in range(batch):
        '''首先获取图片的路径，然后读取图片进行放缩存入[1,width,heigth,3]的数组之中'''
        path=image_sex[begin+i].split(',')[0]
        picture = cv2.resize(cv2.imread(path),(227,227))

        age_images[i,:,:,:]=picture

        sex_index = int(image_sex[begin+i].split(',')[1])
        age_targets[i,sex_index]=1.0

    return age_images,age_targets


sex_images,sex_targets=get_one_batch(173)
print(sex_images.shape)
print(sex_targets.shape)

print(sex_targets[20,:])
# print(age_images[3,:,:,:])

t=sex_images[20,:,:,:]
r=Image.fromarray(t[:,:,0]).convert('L')
g=Image.fromarray(t[:,:,1]).convert('L')
b=Image.fromarray(t[:,:,2]).convert('L')
#b g r
T2=Image.merge('RGB',(b,g,r))
plt.imshow(T2)
plt.show()


