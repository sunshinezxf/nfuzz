from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from keras.models import Model

class AbstractCoverage(ABC):
    def __init__(self, model):
        self.model = model
        self.layers = self.get_layers_with_neurons()

    @abstractmethod
    def update_coverage(self, input_data) -> dict:
        """
        计算覆盖率
        :param input_data:
        :return:
        """
        pass

    @abstractmethod
    def get_coverage(self) -> dict:
        """
        :return:对应的覆盖率
        """
        pass

    @staticmethod
    def neuron_nums(shape):
        """
        :param shape:
        :return:神经元总数
        """
        return np.prod([dim for dim in shape if dim is not None])

    def init_dict(self) -> dict:
        """
        :return:存储各神经元是否被激活的字典
        """
        coverage_dict = defaultdict(bool)
        for layer in self.layers:
            for index in range(self.neuron_nums(layer.output_shape)):  # product of dims
                coverage_dict[(layer.name, index)] = False
        return coverage_dict

    def get_model_activations(self, input_data):
        """
        :return:所有中间层的输出
        """
        intermediate_layer_model = Model(inputs=self.model.input, outputs=[layer.output for layer in self.layers])
        intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
        return intermediate_layer_outputs

    def get_layers_with_neurons(self):
        """
        :return:模型的各个中间层
        """
        return [layer for layer in self.model.layers if
                'flatten' not in layer.name and
                'input' not in layer.name
                ]

    def reshape_input(self, input_data):
        """
        对输入进行reshape
        :param input_data:
        :return:
        """
        input_shape = self.model.layers[0].input_shape

        if input_shape[0] is None:
            if len(input_data.shape) == 2:
                if len(input_shape) == 3:
                    input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1])
                else:
                    input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1], input_shape[-1])
            else:
                if len(input_shape) == 4:
                    input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1], input_data.shape[2])
                else:
                    input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1], input_data.shape[2],
                                                    input_shape[-1])
        else:
            if len(input_data.shape) < len(input_shape):
                input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], input_shape[-1])

        return input_data