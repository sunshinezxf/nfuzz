import numpy as np
import myUtils.muop_util as mu_util
from coverage.NC import NeuronCoverages
from myUtils import model_utils, coverage_metrics, coverage_utils

def process(batch_pool, model, max_loop=10):
    """
    deepHunter的主流程
    :param max_loop:
    :param batch_pool:
    :param model:
    :return:
    """

    x_mutant = []
    y_mutant = []

    # 收集无效的变异种子
    failed_test = []

    for i in range(max_loop):
        print("epoch", i, '---------------------------')
        # 随机选择一个batch进行变异
        batch = batch_pool.select_next()

        # 筛选failedTest
        mu_batch = mu_util.batch_mutate(batch)
        valid_mu_batch, failed_mu_batch = model_utils.select_failed_test(mu_batch, model)

        # 收集无效的变异种子
        failed_test.append(failed_mu_batch)

        new_x_test = []
        new_y_test = []

        for j in range(len(valid_mu_batch)):
            new_x_test.append(valid_mu_batch[j][0])
            new_y_test.append(valid_mu_batch[j][1])
            x_mutant.append(valid_mu_batch[j][0])
            y_mutant.append(valid_mu_batch[j][1])
            # print(valid_mu_batch[j][0])
            # print(valid_mu_batch[j][1])

        # 计算神经元覆盖率 todo:batch包含多个输入，覆盖率计算如何进行
        # new_batch = np.array(new_x_test).reshape(-1, 28, 28, 1)
        # print(new_batch.shape)

        # all_layer_boundary, all_output_list, input_list = coverage_utils.load_neuron(model,np.array(new_x_test))
        # coverage0 = coverage_utils.neuron_coverage(all_output_list)
        # print("basic coverage:", coverage0)
        basic_coverage= NeuronCoverages(model)
        for test_input in new_x_test:
            basic_coverage.update_coverage(np.array(test_input))
            coverage0 = basic_coverage.get_coverage()
            print("basic coverage:", coverage0)


        # 变异后的种子加入pool
        batch_pool.add_batch(valid_mu_batch)

    print('mutation done')
    # 评估
    # model_utils.evaluate(np.array(x_mutant), np.array(y_mutant), model)
    return x_mutant, y_mutant
