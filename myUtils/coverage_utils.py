import copy
import numpy
from scipy import special
from myUtils import csv_utils
from keras.models import Model

def load_neuron(model, x_input):
    """
        加载神经元信息
        :param model: 待测模型
        :param x_input: 测试输入列表,例子。上面的x_test
        :return: input_list : 待计算覆盖率的神经元信息列表（测试数据形成的神经元信息）
    """

    config = model.get_config()  # 详细信息

    layers = config['layers']  # 各层的信息

    print('load model successfully....')
    print('config--------')
    print(config)
    print('layers------')
    print(layers)

    csv_path = []  # 存放神经元上下界信息的路径列表（多个csv文件)
    input_list = []  # 待计算覆盖率的神经元信息列表（测试数据形成的神经元信息）
    all_output_list = []  # 所有输出
    all_layer_boundary = []  # 所有边界值

    # 获取第一层输入的shape
    first_layer = model.get_layer(index=0)
    input_shape = first_layer.input_shape

    # 一般来说输入的shape是三维的
    # if len(input_shape) == 4:
    #     print('x_input shape4:', x_input.shape)
    #     x_input = x_input.reshape(-1, x_input.shape[0], x_input.shape[1], x_input.shape[2])

    # 取某一层的输出为输出新建为model，采用函数模型. todo:每个层都需要计算覆盖率吗，pool层可以不用
    for item in layers:
        layer_name = item['config']['name']
        layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        # 获取各层输出信息
        x_input=x_input.reshape(x_input.shape[0],x_input.shape[1],x_input.shape[2],1)
        layer_output = layer_model.predict(x_input)
        all_output_list.append(layer_output)
        print(layer_name,layer_output.shape)
        # print(layer_output)
        # print(len(layer_output), len(layer_output[0]))  # 二维数组

        # 翻转矩阵
        if len(layer_output) > 0:
            reverse_layer_output = csv_utils.transpose(layer_output)
            # 得到各层各个神经元最大值和最小值
            layer_boundary = csv_utils.get_boundary(reverse_layer_output)
            all_layer_boundary.append(layer_boundary)

            # 将最大值最小值保存为csv文件 具体存放路径待定
            # layer_boundary_list = utils.save_boundary_list(layer_boundary,'./csv/'+ layer_name + '_boundary.csv')
            # csv_path.append(layer_name + '_boundary.csv')

            # layer_output_list = utils.save_layer_output_list(layer_output, './csv/'+layer_name + '_output.csv')
            # csv_path.append(layer_name + '_output.csv')

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
    return all_layer_boundary, all_output_list, input_list


def neuron_coverage(all_output_list, threshold=0.25):
    """
    计算所有层的神经元覆盖率
    :param all_output_list:
    :param threshold:
    :return:
    """

    coveraged_sum = 0
    coverage_sum = 0

    for file in all_output_list:
        for layer in file:
            # print('shape:', layer.shape)
            for i in range(len(layer)):
                # layer维度不统一 可以拉平
                if isinstance(layer[i], numpy.ndarray):
                    # 至少是二维的
                    # print("----------------二维")
                    # print(layer[i],type(layer[i]))
                    for j in range(len(layer[i])):
                        if isinstance(layer[i][j], numpy.ndarray):
                            for k in range(len(layer[i][j])):
                                if layer[i][j][k] >= threshold:
                                    coveraged_sum += 1
                                coverage_sum += 1
                        else:
                            if layer[i][j] >= threshold:
                                coveraged_sum += 1
                            coverage_sum += 1

                else:
                    # 一维的
                    # print("--------------------------------------------一维")
                    # print(layer[i],type(layer[i]))
                    if layer[i] >= threshold:
                        coveraged_sum += 1
                    coverage_sum += 1
    if coverage_sum == 0:
        return 0

    coverage = coveraged_sum / coverage_sum

    return coverage


def k_multi_section_neuron_coverage(k, path_list, all_input_list):
    """
    计算k-multiSection覆盖率
    :param k: 将神经元输出上、下界平均分为k组
    :param path_list: 存放神经元上下界信息的路径列表（多个csv文件)
    :param all_input_list: 待计算覆盖率的神经元信息列表（测试数据形成的神经元信息）
    :return: 计算得到的覆盖率
    """
    # all_boundary_list = path_list
    all_boundary_list = csv_utils.get_all_boundary_file(path_list)

    output_list = csv_utils.covert_to_k_multisection(k, all_boundary_list)  # 将每个神经元信息转化成k个小的上下界信息
    all_label_list = csv_utils.get_label_list(output_list)  # 标签列表，用于计算覆盖率

    k_multi_section_sum = csv_utils.k_multisection_sum(output_list)  # 总神经元个数N * k
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
                if csv_utils.is_neuron_coveraged(neuron_info_list, boundary_info_list)[0]:
                    index = csv_utils.is_neuron_coveraged(neuron_info_list, boundary_info_list)[1]
                    all_label_list[layer][size][index][0] = 1

    for layer in range(len(all_label_list)):
        layer_label_list = all_label_list[layer]
        for size in range(len(layer_label_list)):
            neuron_label_list = layer_label_list[size]
            # print(neuron_label_list)
            for k in range(len(neuron_label_list)):
                if neuron_label_list[k][0] == 1:
                    coveraged_sum += 1

    coverage = coveraged_sum / k_multi_section_sum  # 计算覆盖率

    return coverage


def neuron_boundary_coverage(path_list, all_input_list):
    """
    计算Neuron Boundary覆盖率
    :param path_list:包含神经元信息的文件路径列表
    :param all_input_list:待计算覆盖率的神经元信息列表（测试数据形成的神经元信息）
    :return:计算得到的覆盖率
    """
    coveraged_sum = 0
    coverage_sum = 0

    all_boundary_list = csv_utils.get_all_boundary_file(path_list)
    neuron_boundary_label_list = csv_utils.get_boundary_coverage_label_list(all_boundary_list)

    for data_size in range(len(all_input_list)):
        input_list = all_input_list[data_size]
        for layer in range(len(input_list)):
            layer_input_list = input_list[layer]
            for size in range(len(layer_input_list)):
                neuron_input_list = layer_input_list[size]
                boundary_list = all_boundary_list[layer][size]
                result, flag = csv_utils.is_upper_or_lower(neuron_input_list, boundary_list)
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


def strong_neuron_activation_coverage(path_list, all_input_list):
    """
    计算Strong Neuron Action覆盖率
    :param path_list:包含神经元信息的文件路径列表
    :param all_input_list:待计算覆盖率的神经元信息列表（测试数据形成的神经元信息）
    :return:计算得到的覆盖率
    """
    coveraged_sum = 0
    coverage_sum = 0

    all_boundary_list = csv_utils.get_all_boundary_file(path_list)
    neuron_boundary_label_list = csv_utils.get_boundary_coverage_label_list(all_boundary_list)

    for data_size in range(len(all_input_list)):
        input_list = all_input_list[data_size]
        for layer in range(len(input_list)):
            layer_input_list = input_list[layer]
            for size in range(len(layer_input_list)):
                neuron_input_list = layer_input_list[size]
                boundary_list = all_boundary_list[layer][size]
                result, flag = csv_utils.is_upper_or_lower(neuron_input_list, boundary_list)
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


def top_k_neuron_coverage(k, all_input_list):
    """
    计算Top-k Neuron 覆盖率
    :param k:前k个最大值
    :param all_input_list:神经元信息
    :return:计算得到的覆盖率
    """
    coveraged_sum = 0
    coverage_sum = 0

    top_k_neuron_label_list = csv_utils.get_top_k_neuron_label_list(all_input_list)

    for data_size in range(len(all_input_list)):
        input_list = all_input_list[data_size]
        for layer in range(len(input_list)):
            layer_list = input_list[layer]
            top_k_index_list = csv_utils.get_top_k_index_list(k, layer_list)
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
            top_k_index_list = csv_utils.get_top_k_index_list(k, layer_list)
            neuron_patterns.append(top_k_index_list)
        if neuron_patterns not in coveraged_patterns:
            coveraged_patterns.append(neuron_patterns)

    # coverage = len(coveraged_patterns) / patterns_sum

    return len(coveraged_patterns)
