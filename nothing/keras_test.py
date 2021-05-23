from myUtils import coverageUtils as coverage_functions
from myUtils import csvUtils as utils
import numpy as np
from scipy import special
import copy
import tensorflow as tf
from muoperator import muop_util as mu

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Model
import time


# 对MNIST数据做简单修改，添加噪音
def do_basic_mutations(element, a_min=-1.0, a_max=1.0):
    image, label = element
    sigma = 0.5
    noise = np.random.normal(size=image.shape, scale=sigma)

    mutated_image = noise + image

    mutated_image = np.clip(
        mutated_image, a_min=a_min, a_max=a_max
    )

    mutated_element = [mutated_image, label]
    return mutated_element


# 进行变异
def mutate(element):
    image, label = element

    mutated_image = mu.image_mutate(5, image)

    mutated_element = [mutated_image, label]
    return mutated_element


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('load mnist data----------')

# print(x_train.shape)
# print(x_test.shape)

# 重新定义数据格式，归一化 归一化以后shape由28*28变784一维的了
# x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
# x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
#
# # print(x_train.shape)
# # print(x_test.shape)
#
# # # 转one-hot标签
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)


# x_test = x_test.tolist()
# y_test = y_test.tolist()
# x_test = x_test * 2
# y_test = y_test * 2
#
# x_test = np.array(x_test)
# y_test = np.array(y_test)
# print(len(x_test))

data = []
for i in range(len(x_test)):
    image_info = []
    image_info.append(x_test[i])
    image_info.append(y_test[i])
    data.append(image_info)

x_test = x_test.tolist()
y_test = y_test.tolist()

# # mutated_image_data = []
# for i in range(len(data)):
#     image_info = data[i]
#     # mutated_image_info = do_basic_mutations(image_info)
#     for itr in range(10):
#         # image_info = do_basic_mutations(image_info)
#         image_info = mutate(image_info)
#     # x_test.append(mutated_image_info[0])
#     # y_test.append(mutated_image_info[1])
#     mutated_image_info = image_info
#     x_test[i] = mutated_image_info[0]
#     y_test[i] = mutated_image_info[1]
#     # print(len(mutated_image_info[0]))
#     # print(mutated_image_info[0])
#     # print(len(mutated_image_info[1]))
#     # print(mutated_image_info[1])

x_test = np.array(x_test)
y_test = np.array(y_test)
# print(x_test.shape)
# x_test=x_test.reshape(10000,28,28,1)
# 解决方法：原因是输入的第一个dimension是bachsize，所以需要将数据reshape为(1，X, X, X)。

# 载入模型
# model = load_model('11.h5')



# print(model.get_config())


def load_neuron(path, x_input):
    """
        计算k-multisection覆盖率
        :param model: 待测模型
        :param x_input: 测试输入,例子。上面的x_test
        :return: csv_path:存放神经元上下界信息的路径列表（多个csv文件)
                input_list : 待计算覆盖率的神经元信息列表（测试数据形成的神经元信息）
    """
    model = load_model(path)

    config = model.get_config()  # 详细信息

    layers = config['layers']  # 各层的信息

    print('load model successfully....')
    print('config--------')
    print(config)
    print('layers------')
    print(layers)

    csv_path = []  # 存放神经元上下界信息的路径列表（多个csv文件)
    input_list = []  # 待计算覆盖率的神经元信息列表（测试数据形成的神经元信息）

    # 取某一层的输出为输出新建为model，采用函数模型. 每个层都需要计算覆盖率吗
    for item in layers:
        layer_name = item['config']['name']
        layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        # 获取各层输出信息
        layer_output = layer_model.predict(x_input)
        print(layer_name)
        print(layer_output)
        print(len(layer_output), len(layer_output[0]))  # 二维数组

        # 翻转矩阵
        reverse_layer_output = utils.reverse_list(layer_output)

        # 得到各层各个神经元最大值和最小值
        layer_boundary = utils.get_boundary(reverse_layer_output)

        # 将最大值最小值保存为csv文件 具体存放路径待定
        layer_boundary_list = utils.save_boundary_list(layer_boundary,'./csv/'+ layer_name + '_boundary.csv')
        csv_path.append(layer_name + '_boundary.csv')

        layer_output_list = utils.save_layer_output_list(layer_output, './csv/'+layer_name + '_output.csv')
        csv_path.append(layer_name + '_output.csv')

        # print(layer_name + "_boundary_list", len(layer_boundary_list), ":::", len(layer_boundary_list[0]))

        for size in range(len(layer_output)):
            data_size_input_list = []
            layer_sub_list = layer_output[size]

            layer_sub_input_list = []

            for neuron_sum in range(len(layer_sub_list)):
                layer_sub_input_list.append([layer_sub_list[neuron_sum]])
            data_size_input_list.append(layer_sub_input_list)

            input_list.append(data_size_input_list)

    print('execute---------')
    return csv_path, input_list

# x_test=x_test.reshape(-1,28,28,1)
# csv_path, input_list = load_neuron('../model/LeNet5.h5', x_test)

# time0 = time.time()
# coverage0 = coverage_functions.neuron_coverage(csv_path)
# print("basic coverage:", coverage0)

# time1 = time.time()
# coverage1 = coverage_functions.k_multisection_neuron_coverage(10,csv_path,input_list)
# print("k_multisection coverage:", coverage1)
#
# time2 = time.time()
# coverage2 = coverage_functions.neuron_boundary_coverage(csv_path,input_list)
# print("neuron boundary coverage:", coverage2)
#
#
# time3 = time.time()
# coverage3 = coverage_functions.strong_neuron_activation_coverage(csv_path,input_list)
# print("strong neuron activation coverage:", coverage3)
#
#
# time4 = time.time()
# coverage4 = coverage_functions.top_k_neuron_coverage(2, input_list)
# print("top-k neuron coverage:", coverage4)
#
#
# time5 = time.time()
# coverage5 = coverage_functions.top_neuron_patterns(2, input_list)
# print("top-k patterns:", coverage5)
