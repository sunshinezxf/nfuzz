#  覆盖计算方法

import Utils as utils
import copy
from scipy import special

def neuron_coverage(path_list,threshold=0.25):
    coveraged_sum = 0
    coverage_sum = 0

    all_output_list = utils.get_all_output_file(path_list)
    for file in all_output_list:
        for layer in file:
            for i in range(len(layer)):
                if layer[i]>=threshold:
                    coveraged_sum+=1
                coverage_sum+=1

    coverage = coveraged_sum / coverage_sum

    return coverage


# todo:只要最简单的就行了
# k-multisection Neuron Coverage
def k_multisection_neuron_coverage(k, path_list, all_input_list):
    """
    计算k-multisection覆盖率
    :param k: 将神经元输出上、下界平均分为k组
    :param path_list: 存放神经元上下界信息的路径列表（多个csv文件)
    :param all_input_list: 待计算覆盖率的神经元信息列表（测试数据形成的神经元信息）
    :return: 计算得到的覆盖率
    """
    # all_boundary_list = path_list
    all_boundary_list = utils.get_all_boundary_file(path_list)
    output_list = utils.covert_to_k_multisection(k, all_boundary_list)  # 将每个神经元信息转化成k个小的上下界信息
    all_label_list = utils.get_label_list(output_list)  # 标签列表，用于计算覆盖率

    k_multisection_sum = utils.k_multisection_sum(output_list)  # 总神经元个数N * k
    coveraged_sum = 0  # 被覆盖的个数
    for data_size in range(len(all_input_list)):
        input_list = all_input_list[data_size]
        for layer in range(len(input_list)):  # ==>3层
            layer_info_list = input_list[layer]  # ==》第一层信息
            for size in range(len(layer_info_list)):
                neuron_info_list = layer_info_list[size]

                boundary_info_list = output_list[layer][size]
                # print("被覆盖")
                # print(all_label_list[layer][size])
                # else:
                # print("未覆盖")
                # print(all_label_list[layer][size])
                if utils.is_neuron_coveraged(neuron_info_list, boundary_info_list)[0]:
                    index = utils.is_neuron_coveraged(neuron_info_list, boundary_info_list)[1]
                    all_label_list[layer][size][index][0] = 1

    for layer in range(len(all_label_list)):
        layer_label_list = all_label_list[layer]
        for size in range(len(layer_label_list)):
            neuron_label_list = layer_label_list[size]
            # print(neuron_label_list)
            for k in range(len(neuron_label_list)):
                if neuron_label_list[k][0] == 1:
                    coveraged_sum += 1

    coverage = coveraged_sum / k_multisection_sum  # 计算覆盖率

    return coverage


# Neuron Boundary Coverage
def neuron_boundary_coverage(path_list, all_input_list):
    """
    计算Neuron Boundary覆盖率
    :param path_list:包含神经元信息的文件路径列表
    :param all_input_list:待计算覆盖率的神经元信息列表（测试数据形成的神经元信息）
    :return:计算得到的覆盖率
    """
    coveraged_sum = 0
    coverage_sum = 0

    all_boundary_list = utils.get_all_boundary_file(path_list)
    neuron_boundary_label_list = utils.get_boundary_coverage_label_list(all_boundary_list)

    for data_size in range(len(all_input_list)):
        input_list = all_input_list[data_size]
        for layer in range(len(input_list)):
            layer_input_list = input_list[layer]
            for size in range(len(layer_input_list)):
                neuron_input_list = layer_input_list[size]
                boundary_list = all_boundary_list[layer][size]
                result, flag = utils.is_upper_or_lower(neuron_input_list, boundary_list)
                if result is True and flag == 0:
                    neuron_boundary_label_list[layer][size][0][0] = 1
                if result is True and flag == 1:
                    neuron_boundary_label_list[layer][size][1][0] = 1

    for layer in range(len(neuron_boundary_label_list)):
        layer_label_list = neuron_boundary_label_list[layer]
        for size in range(len(layer_label_list)):
            neuron_label_list = layer_label_list[size]
            for lower_and_upper in range(len(neuron_label_list)):
                if neuron_label_list[lower_and_upper][0] == 1:
                    coveraged_sum += 1

    for layer in range(len(neuron_boundary_label_list)):
        layer_label_list = neuron_boundary_label_list[layer]
        coverage_sum += len(layer_label_list) * 2

    coverage = coveraged_sum / coverage_sum

    return coverage


# Strong Neuron Activation Coverage
def strong_neuron_activation_coverage(path_list, all_input_list):
    """
    计算Strong Neuron Action覆盖率
    :param path_list:包含神经元信息的文件路径列表
    :param all_input_list:待计算覆盖率的神经元信息列表（测试数据形成的神经元信息）
    :return:计算得到的覆盖率
    """
    coveraged_sum = 0
    coverage_sum = 0

    all_boundary_list = utils.get_all_boundary_file(path_list)
    neuron_boundary_label_list = utils.get_boundary_coverage_label_list(all_boundary_list)

    for data_size in range(len(all_input_list)):
        input_list = all_input_list[data_size]
        for layer in range(len(input_list)):
            layer_input_list = input_list[layer]
            for size in range(len(layer_input_list)):
                neuron_input_list = layer_input_list[size]
                boundary_list = all_boundary_list[layer][size]
                result, flag = utils.is_upper_or_lower(neuron_input_list, boundary_list)
                if result is True and flag == 1:
                    neuron_boundary_label_list[layer][size][1][0] = 1

    for layer in range(len(neuron_boundary_label_list)):
        layer_label_list = neuron_boundary_label_list[layer]
        for size in range(len(layer_label_list)):
            neuron_label_list = layer_label_list[size]
            for lower_and_upper in range(len(neuron_label_list)):
                if neuron_label_list[lower_and_upper][0] == 1:
                    coveraged_sum += 1

    for layer in range(len(neuron_boundary_label_list)):
        layer_label_list = neuron_boundary_label_list[layer]
        coverage_sum += len(layer_label_list)

    coverage = coveraged_sum / coverage_sum

    return coverage


# Top-k Neuron Coverage
def top_k_neuron_coverage(k, all_input_list):
    """
    计算Top-k Neuron 覆盖率
    :param k:前k个最大值
    :param all_input_list:神经元信息
    :return:计算得到的覆盖率
    """
    coveraged_sum = 0
    coverage_sum = 0

    top_k_neuron_label_list = utils.get_top_k_neuron_label_list(all_input_list)

    for data_size in range(len(all_input_list)):
        input_list = all_input_list[data_size]
        for layer in range(len(input_list)):
            layer_list = input_list[layer]
            top_k_index_list = utils.get_top_k_index_list(k, layer_list)
            layer_top_k_neuron_label_list = top_k_neuron_label_list[layer]
            for ite in range(len(top_k_index_list)):
                layer_top_k_neuron_label_list[top_k_index_list[ite]][0] = 1

    for layer in range(len(top_k_neuron_label_list)):
        layer_top_k_neuron_label_list = top_k_neuron_label_list[layer]
        coverage_sum += len(layer_top_k_neuron_label_list)
        for size in range(len(layer_top_k_neuron_label_list)):
            if layer_top_k_neuron_label_list[size][0] == 1:
                coveraged_sum += 1

    coverage = coveraged_sum / coverage_sum
    return coverage


# Top-k Neuron Patterns
def top_neuron_patterns(k, all_input_list):

    neuron_sum_each_layer = []
    net_model = copy.deepcopy(all_input_list[0])
    for layer in range(len(net_model)):
        neuron_sum_each_layer.append(len(net_model[layer]))

    patterns_sum = 1
    for ite in range(len(neuron_sum_each_layer)):
        patterns_sum *= int(special.perm(neuron_sum_each_layer[ite], k))

    coveraged_patterns = []

    for data_size in range(len(all_input_list)):
        input_list = all_input_list[data_size]
        neuron_patterns = []
        for layer in range(len(input_list)):
            layer_list = input_list[layer]
            top_k_index_list = utils.get_top_k_index_list(k, layer_list)
            neuron_patterns.append(top_k_index_list)
        if neuron_patterns not in coveraged_patterns:
            coveraged_patterns.append(neuron_patterns)

    # coverage = len(coveraged_patterns) / patterns_sum

    return len(coveraged_patterns)



