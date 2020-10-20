# 工具类

import csv
import numpy as np
import copy


# 翻转列表
def reverse_list(li):
    """
    翻转矩阵
    :param li:待翻转列表
    :return:翻转完成后的列表
    """
    reversed_list = [
        [li[i][j] for i in range(len(li))]
        for j in range(len(li[0]))
    ]
    return reversed_list


# 获取列表中每一元素最大值和最小值
def get_boundary(mylist):
    """
    获取列表中每一元素的最大值和最小值
    :param mylist:待计算的列表
    :return:含有每行最大值和最小值的列表
    """
    boundary_list = []
    for i in range(len(mylist)):
        li = []
        subli = mylist[i]
        max = np.max(subli)
        min = np.min(subli)
        li.append(min)
        li.append(max)
        boundary_list.append(li)

    return boundary_list


# 将各层神经元边界（上界、下界）保存到csv文件中
def save_boundary_list(li, path):
    """
    将各层神经元边界（上界、下界）保存到csv文件中
    :param li:待保存的列表
    :param path:保存的路径
    :return:
    """
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in li:
            writer.writerow(row)


# 从csv文件中获取各层个神经元边界值（上界和下界）
def get_boundary_from_file(path):
    """
    从csv文件中获取各层个神经元边界值（上界和下界）
    :param path: 文件的路径
    :return: 包含神经元信息(上界、下界）的列表
    """
    with open(path, 'r') as csvfile:
        data = csvfile.readlines()

    boundary_list = []
    for i in range(len(data)):
        substr = data[i]
        li = substr.split(',')
        strr = li[-1]
        lii = strr.split('\n')
        li[-1] = lii[0]
        for j in range(len(li)):
            li[j] = float(li[j])

        boundary_list.append(li)

    return boundary_list


# 获取多个csv文件
def get_all_boundary_file(path_list):
    """
    获取多个csv文件
    :param path_list:
    :return:多个csv文件中列表组成的列表
    """
    all_boundary_list = []
    for i in range(len(path_list)):
        boundary_list = get_boundary_from_file(path_list[i])
        all_boundary_list.append(boundary_list)

    return all_boundary_list


# 将神经元信息列表转换为k-multisection
def covert_to_k_multisection(k, input_list):
    """
    将神经元信息列表转换为k-multisection
    :param k: 每个神经元信息（上下界）转换为k个列表
    :param input_list:待转换的列表
    :return:转换后列表
    """
    k_multisection_all_layer_list = []

    for i in range(len(input_list)):
        layer_boundary_list = input_list[i]
        k_multisection_layer_list = []

        for j in range(len(layer_boundary_list)):
            neuron_boundary = layer_boundary_list[j]
            low = neuron_boundary[0]
            upper = neuron_boundary[-1]
            increment = (upper - low) / k
            k_multisection_neuron_list = []

            for size in range(k - 1):
                k_multisection_neuron_list.append(list([low + size * increment, low + (size + 1) * increment]))
            k_multisection_neuron_list.append(list([upper - increment, upper]))

            k_multisection_layer_list.append(k_multisection_neuron_list)

        k_multisection_all_layer_list.append(k_multisection_layer_list)

    return k_multisection_all_layer_list


# 计算k_multisection总数
def k_multisection_sum(input_list):
    neuron_sum = 0
    for layer in range(len(input_list)):
        layer_list = input_list[layer]
        for size in range(len(layer_list)):
            neuron_list = layer_list[size]
            for k in range(len(neuron_list)):
                neuron_sum += 1
    return neuron_sum


# 获取覆盖标签信息列表
def get_label_list(input_list):
    """
    获取覆盖标签信息列表
    :param input_list:传入的覆盖信息列表
    :return:覆盖标签信息列表
    """
    all_label_list = []
    for layer in range(len(input_list)):
        layer_label_list = []
        layer_list = input_list[layer]
        for size in range(len(layer_list)):
            neuron_label_list = []
            neuron_list = layer_list[size]
            for k in range(len(neuron_list)):
                label = [0]
                neuron_label_list.append(label)
            layer_label_list.append(neuron_label_list)
        all_label_list.append(layer_label_list)

    return all_label_list


# 判断单个神经元信息是否被过覆盖
def is_neuron_coveraged(neuron_info_list, k_multisection_list):
    """
    判断单个神经元信息是否被过覆盖
    :param neuron_info_list: 包含单个神经元信息的列表
    :param k_multisection_list:包含训练过程中该神经元信息的列表
    :return:被覆盖返回True，否则返回False
    """
    result = False
    index = None
    neuron_info = neuron_info_list[0]
    for k in range(len(k_multisection_list)):
        sub_boundary = k_multisection_list[k]
        if (neuron_info >= sub_boundary[0]) and (neuron_info <= sub_boundary[1]):
            result = True
            index = k

    return result, index


# 获取神经元边界信息标签列表
def get_boundary_coverage_label_list(input_list):

    neuron_boundary_label_list = []
    for layer in range(len(input_list)):
        layer_info_list = input_list[layer]
        layer_label_list = []
        for size in range(len(layer_info_list)):
            neuron_info_list = layer_info_list[size]
            neuron_label_list = []
            for boundary in range(len(neuron_info_list)):
                neuron_label_list.append([0])

            layer_label_list.append(neuron_label_list)
        neuron_boundary_label_list.append(layer_label_list)

    return neuron_boundary_label_list


# 判断神经元信息是否超过上界或下界
def is_upper_or_lower(neuron_info_list, boundary_list):

    """
    判断神经元信息是否超过上界或下界
    :param neuron_info_list: 包含的那个神经元信息的列表
    :param boundary_list: 神经元信息边界列表
    :return:超多上界或下界返回True,否则返货False
    """
    result = False
    flag = None
    neuron_info_value = neuron_info_list[0]
    if neuron_info_value < boundary_list[0]:
        result = True
        flag = 0
    elif neuron_info_value > boundary_list[1]:
        result = True
        flag = 1
    else:
        result = False

    return result, flag


# 获取top-k神经元信息标签列表
def get_top_k_neuron_label_list(input_list):
    """
    获取top-k神经元信息标签列表
    :param input_list: 含有神经元信息的列表
    :return: 标签列表，用于计算覆盖率
    """
    net_model = input_list[0]
    top_k_neuron_label_list = copy.deepcopy(net_model)
    for layer in range(len(top_k_neuron_label_list)):
        layer_label_list = top_k_neuron_label_list[layer]
        for size in range(len(layer_label_list)):
            layer_label_list[size][0] = 0

    return top_k_neuron_label_list


# 获取列表中top-k值索引
def get_top_k_index_list(k, input_list):
    """
    获取列表中top-k值索引
    :param k:
    :param input_list:
    :return:
    """
    top_k_index_list = []
    for ite in range(k):

        max_index = 0
        for size in range(len(input_list)):
            if input_list[size][0] > input_list[max_index][0] and size not in top_k_index_list:
                max_index = size
        top_k_index_list.append(max_index)

    return top_k_index_list

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
