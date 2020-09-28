import abc
import cv2
import numpy as np
import random

class Mutator(metaclass=abc.ABCMeta):
    """
        An interface for mutator
    """
    # alpha>0,beta<1,论文中只字未提如何选取具体的值，应该是看效果吧
    ALPHA = 0.02
    BETA = 0.2

    def mutate(self, seed):
        """
            Mutation seed
            :param:
                seed -- original seed
            :return：
                new_seed -- a mutant seed
        """
        pass

    """仿射变换其三个参数分别为:输入图像,变换矩阵,输出图像大小
    deepHunter中选择四种：平移、缩放、剪切、旋转
    x=x_train[0]，输入的图像
    """

    # 沿着横纵轴缩放
    def affine_scaling(self,img, x_rate, y_rate):
        height, width = img.shape[:2]  # 获取图像的高和宽
        M = np.array([
            [x_rate, 0, 0],
            [0, y_rate, 0]
        ], dtype=np.float32)
        mutant = cv2.warpAffine(img, M, (height, width))
        return mutant

    # 平移 x_dis,y_dis可为负数
    def affine_translation(self,img, x_dis, y_dis):
        height, width = img.shape[:2]  # 获取图像的高和宽
        M = np.array([
            [1, 0, x_dis],
            [0, 1, y_dis]
        ], dtype=np.float32)
        mutant = cv2.warpAffine(img, M, (height, width))
        return mutant

    # 剪裁一个子区域并补边
    def affine_shear(self,img):
        height, width = img.shape[:2]  # 获取图像的高和宽

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
        cropped = img[np.min([y1, y2]):np.max([y1, y2]), np.min([x1, x2]):np.max([x1, x2])]

        M = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)

        mutant = cv2.warpAffine(cropped, M, (height, width))

        return mutant

    # 旋转
    def affine_rotate(self,x, angle, scale=1.0):
        height, width = x.shape[:2]  # 获取图像的高和宽
        center = (width / 2, height / 2)  # 默认中心
        M = cv2.getRotationMatrix2D(center, angle, scale)
        mutant = cv2.warpAffine(x, M, (height, width))
        return mutant

    # 噪声
    def pixel_noise(self,img, prob):
        """
            添加椒盐噪声
            prob:噪声比例,0.1
            """
        output = np.zeros(img.shape, np.uint8)
        thres = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]
        return output

    # 高斯模糊。中心点领域，
    def pixel_blur(self,img, ksize, sigma):
        """
        https://blog.csdn.net/wuqindeyunque/article/details/103694900?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param
        这个函数可以根据ksize和sigma求出对应的高斯核，计算公式为
        sigma = 0.3*((ksize-1)*0.5-1)+0.8
        当ksize=3时，sigma=0.8
        当ksize=5时，sigma为1.1.
        kszie,sigma越大越模糊

        """
        mutant = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        return mutant

    # 对比度/亮度
    def pixel_contrast(self,img, brightness, contrast):
        # 图像混合( cv2.addWeighted() )
        # 这也是图像添加，但是对图像赋予不同的权重，使得它具有混合感或透明感。
        pic_turn = cv2.addWeighted(img, contrast, img, 0, brightness)
        return pic_turn

    # 判断是否有意义
    def is_satisfied(self,img, output):
        height, width = img.shape[:2]  # 获取图像的高和宽
        l0, l_inf = 0, 0
        for row in range(height):  # 遍历高
            for col in range(width):
                p_seed = img[row, col]
                p_output = output[row, col]
                if p_seed != p_output:  # 像素值发生改变
                    l0 = l0 + 1

                minus = abs(p_seed - p_output)
                if minus > l_inf:
                    l_inf = minus  # 像素值发生的最大改变

        if l0 < self.ALPHA * height * width:
            if l_inf <= 255:
                return True
            else:
                return False
        else:
            if l_inf < 255 * self.BETA:
                return True
            else:
                return False

    def transform(self, state, img):
        height, width = img.shape[:2]  # 获取图像的高和宽
        if state == 0:  # 缩放
            x_rate = random.random() * 5
            y_rate = random.random() * 5
            return self.affine_scaling(img, x_rate, y_rate)
        if state == 1:  # 平移
            x_dis = random.randint(0, width / 2)
            y_dis = random.randint(0, height / 2)
            return self.affine_translation(img, x_dis, y_dis)
        if state == 2:  # 剪裁
            return self.affine_shear(img)
        if state == 3:  # 旋转
            angle = random.randint(0, 360)
            return self.affine_rotate(img, angle)
        if state == 4:  # 噪声
            prob = random.random() / 5
            return self.pixel_noise(img, prob)
        if state == 5:  # 模糊
            ksize = random.randint(1, 5)
            sigma = random.random() * 10
            return self.pixel_blur(img, ksize, sigma)
        if state == 6:  # 对比度/亮度
            brightness = random.randint(1, 100)
            contrast = random.random()
            return self.pixel_contrast(img, brightness, contrast)

    def random_pick(self,state):
        if state == 0:  # 可用选择一次仿射
            s = random.randint(0, 6)
            return s
        else:
            s = random.randint(4, 6)
            return s

    # deepHunter alg2
    def image_mutate(self,try_num, seed):
        """
        :param try_num: 最大尝试次数
        :param seed: 初始种子(单个图)
        :return: 变异成功的新种子或者原种子
        """
        state = 0
        I = I0 = I01 = seed
        I1 = None
        t = -1
        for i in range(try_num):
            if state == 0:
                t = self.random_pick(state)
            else:
                t = self.random_pick(state)

            I1 = self.transform(t, I)

            if self.is_satisfied(I01, I1):
                if t > 4:
                    state = 1
                    I01 = self.transform(t, I0)
                    return I1
        return I






