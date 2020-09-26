from keras.datasets import mnist
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

HEIGHT = 28
WIDTH = 28

# alpha>0,beta<1,论文中只字未提如何选取具体的值，应该是看效果吧
ALPHA=0.02
BETA=0.2

# 进行变异

"""仿射变换其三个参数分别为:输入图像,变换矩阵,输出图像大小
deepHunter中选择四种：平移、缩放、剪切、旋转
x=x_train[0]，输入的图像
"""


# 沿着横纵轴缩放
def affine_scaling(x, x_rate, y_rate):
    height, width = x.shape[:2]  # 获取图像的高和宽
    M = np.array([
        [x_rate, 0, 0],
        [0, y_rate, 0]
    ], dtype=np.float32)
    img = cv2.warpAffine(x, M, (height, width))
    return img


# 平移 x_dis,y_dis可为负数
def affine_translation(x, x_dis, y_dis):
    height, width = x.shape[:2]  # 获取图像的高和宽
    M = np.array([
        [1, 0, x_dis],
        [0, 1, y_dis]
    ], dtype=np.float32)
    img = cv2.warpAffine(x, M, (height, width))
    return img


# 剪裁一个子区域并补边
def affine_shear(x):
    height, width = x.shape[:2]  # 获取图像的高和宽

    # 随机起点和终点
    x1 = random.randint(0, width)
    y1 = random.randint(0, height)

    x2 = random.randint(0, width)
    y2 = random.randint(0, height)

    if x1 == x2:
        if x2 == width:
            x1 = x1 - 1
        else:
            x2 = x2 + 1

    if y1 == y2:
        if y2 == height:
            y1 = y1 - 1
        else:
            y2 = y2 + 1

    # 裁剪坐标为[y0:y1, x0:x1]
    cropped = x[np.min([y1, y2]):np.max([y1, y2]), np.min([x1, x2]):np.max([x1, x2])]

    M = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    img = cv2.warpAffine(cropped, M, (height, width))

    return img


# 旋转
def affine_rotate(x, angle, scale=1.0):
    height, width = x.shape[:2]  # 获取图像的高和宽
    center = (width / 2, height / 2)  # 默认中心
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img = cv2.warpAffine(x, M, (height, width))
    return img


# 噪声
def pixel_noise(x, prob):
    """
        添加椒盐噪声
        prob:噪声比例,0.1
        """
    output = np.zeros(x.shape, np.uint8)
    thres = 1 - prob
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = x[i][j]
    return output

# 高斯模糊。中心点领域，
def pixel_blur(x,ksize,sigma):
    """
    https://blog.csdn.net/wuqindeyunque/article/details/103694900?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param
    这个函数可以根据ksize和sigma求出对应的高斯核，计算公式为
    sigma = 0.3*((ksize-1)*0.5-1)+0.8
    当ksize=3时，sigma=0.8
    当ksize=5时，sigma为1.1.
    kszie,sigma越大越模糊

    """
    img = cv2.GaussianBlur(x, (ksize, ksize),sigma)
    return img

# 对比度/亮度
def pixel_contrast(x,brightness,contrast):
    # 图像混合( cv2.addWeighted() )
    # 这也是图像添加，但是对图像赋予不同的权重，使得它具有混合感或透明感。
    pic_turn = cv2.addWeighted(x, contrast, x, 0, brightness)
    return pic_turn


# 查看
def show():
    # cv2.imshow('image1', x_train[0])
    cv2.imshow('image1', pixel_noise(x_train[0],0.2))
    # cv2.imshow('image', pixel_blur(x_train[0], 6))
    # cv2.imshow('image2', pixel_contrast(x_train[0], 100, 1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# 转成vgg16可用的图像
def vgg16_convert(imgs):
    imgs = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in imgs]  # 变成RGB的
    imgs = np.concatenate([arr[np.newaxis] for arr in imgs]).astype('float32')
    return imgs

# 判断是否有意义
def is_satisfied(seed,output):
    height, width = seed.shape[:2]  # 获取图像的高和宽
    l0,l_inf=0,0
    for row in range(height):  # 遍历高
        for col in range(width):
            p_seed=seed[row,col]
            p_output=output[row,col]
            if p_seed!=p_output: # 像素值发生改变
                l0=l0+1

            minus=abs(p_seed-p_output)
            if minus>l_inf:
                l_inf=minus # 像素值发生的最大改变

    if l0<ALPHA*height*width:
        if l_inf<=255:
            return True
        else:
            return False
    else:
        if l_inf<255*BETA:
            return True
        else:return False




def transform(s,seed):
    height, width = seed.shape[:2]  # 获取图像的高和宽
    if s==0: # 缩放
        x_rate=random.random()*5
        y_rate=random.random()*5
        return affine_scaling(seed,x_rate,y_rate)
    if s==1: # 平移
        x_dis = random.randint(0,width/2)
        y_dis = random.randint(0,height/2)
        return affine_translation(seed,x_dis,y_dis)
    if s==2: # 剪裁
        return affine_shear(seed)
    if s==3: # 旋转
        angle=random.randint(0,360)
        return affine_rotate(seed,angle)
    if s==4: # 噪声
        prob=random.random()/5
        return pixel_noise(seed,prob)
    if s==5: # 模糊
        ksize=random.randint(1,5)
        sigma=random.random()*10
        return pixel_blur(seed,ksize,sigma)
    if s==6: # 对比度/亮度
        brightness=random.randint(1,100)
        contrast=random.random()
        return pixel_contrast(seed,brightness,contrast)



def random_pick(state):
    if state==0: # 可用选择一次仿射
        s=random.randint(0,6)
        return s
    else:
        s=random.randint(4,6)
        return s

# alg2
def mutate(try_num,seed):
    """
    :param try_num: 最大尝试次数
    :param seed: 初始种子
    :return: 变异成功的新种子或者原种子
    """
    state=0
    I=I0=I01=seed
    I1=None
    t=-1
    for i in range(try_num):
        if state == 0:
            t=random_pick(state)
        else:
            t=random_pick(state)

        I1=transform(t,I)

        if is_satisfied(I01,I1):
            if t>4:
                state=1
                I01=transform(t,I0)
                return I1
    return I


# show()


# X_train = vgg16_convert(x_train)
# X_test = vgg16_convert(x_test)
