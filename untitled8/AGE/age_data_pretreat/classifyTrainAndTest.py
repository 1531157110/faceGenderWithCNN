'''将图片分为训练集和测试集 测试集占0.2'''
f1= open('image_age.txt','r')
data_set = list(f1.read().replace("'",'').replace('[','').replace(']','').splitlines())
f1.close()
#17393个数据 12800个训练数据
print(len(data_set))
n = len(data_set)

train_list = data_set[0:-3478]
test_list = data_set[-3478:]

print(len(train_list))
print(len(test_list))

with open('trainImagesAndLabels.txt','w+') as f2:
    for i in range(len(train_list)):
        f2.write(train_list[i].split(',')[0])
        f2.write(' ')
        f2.write(train_list[i].split(',')[1])
        f2.write('\n')

with open('testImagesAndLabels.txt','w+') as f3:
    for i in range(len(test_list)):
        f3.write(train_list[i].split(',')[0])
        f3.write(' ')
        f3.write(train_list[i].split(',')[1])
        f3.write('\n')
