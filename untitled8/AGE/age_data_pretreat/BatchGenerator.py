import tensorflow as tf

def gaussian_nosie_layer(input_image,std):
    noise = tf.random_normal(shape=tf.shape(input_image),mean=0.0,stddev=std,dtype=tf.float32)
    noise_image = tf.cast(input_image,tf.float32)+noise

    noise_image = tf.clip_by_value(noise_image,0,1.0)
    return noise

def parse_data(filename):
    ''' 导入数据，进行预处理，输出两张图像,
    分别是输入图像和目标图像（例如，在图像去噪中，输入的是一张带噪声图像，目标图像是无噪声图像）
    args:
        filaneme, 图片的路径
    eeturn:
        输入图像，目标图像'''

    image =tf.read_file(filename)
    image = tf.image.decode_jpeg(image,channels=3)

    #数据预处理，
    '''
    '''
    image = tf.image.resize_images(image,[227,227])

    # # 随机提取patch
    # image = tf.random_crop(image, size=(277,277, 3))
    # # 数据增强，随机水平翻转图像
    # image = tf.image.random_flip_left_right(image)
    #图像归一化
    image1 = tf.cast(image,tf.float32)/255.0

    #加噪声
    # n_image= gaussian_nosie_layer(image,0.5)
    #
    # return n_image,image
    return image1


def train_generator(batchsize,paths):
    '''生成器，用于产生训练数据
    Args:
        batchsize,训练的batch size
        shuffle, 是否随机打乱batch

    Returns:
        训练需要的数据
        '''

    with tf.Session() as sess:
        #创建数据库

        train_dataset = tf.data.Dataset().from_tensor_slices((paths))
        #预处理数据
        train_dataset = train_dataset.map(parse_data())

        #设置batchsize
        train_dataset = train_dataset.batch(batchsize)
        #无限重复数据
        train_dataset = train_dataset.repeat()

        #打乱
        # if shuffle:
        #     train_dataset = train_dataset.shuffle(buffer_size=4)

        #创建迭代器
        train_iterator = train_dataset.make_initializable_iterator()

        sess.run(train_iterator.initializer)
        train_batch = train_iterator.get_next()

        #开始生成数据

        while True:
            try:
                x_batch,y_batch = sess.run(train_batch)
                yield (x_batch,y_batch)
            except:
                #如果没有train_dataset= train_dataset.repet()
                #数据遍历完了 就会抛出异常
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch=train_iterator.get_next()
                x_batch,y_batch = sess.run(train_batch)
                yield (x_batch,y_batch)





f1= open('trainImagesAndLabels.txt','r')
data_set = f1.read().splitlines()
f1.close()
imageDirectory = []

print(data_set[0])