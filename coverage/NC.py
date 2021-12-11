import numpy as np
from coverage.abstract_coverage import AbstractCoverage

class NeuronCoverages(AbstractCoverage):
    """
    最基本的神经元覆盖率
    """

    def __init__(self, model, activation_threshold=0.25):
        AbstractCoverage.__init__(self,model)
        self.coverage_dict = {}
        self.batch_dict=self.init_dict()
        self.threshold = activation_threshold

    def update_coverage(self, input_data):
        self.coverage_dict=self.init_dict()
        input_data=self.reshape_input(input_data)
        layer_names = [layer.name for layer in self.layers]

        intermediate_layer_activations = self.get_model_activations(input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            activations_shape = layer_activations.shape
            for neuron_index in range(self.neuron_nums(activations_shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, activations_shape)]  # 多维展开成一维
                if neuron_activation > self.threshold:
                    self.coverage_dict[(layer_name, neuron_index)] = True

    def update_batch_coverage(self,input_datas):
        """
        计算一个batch的覆盖率.区别在于是否刷新coverage dict
        :param input_datas:a batch of input data
        :return:
        """

        layer_names = [layer.name for layer in self.layers]

        for input_data in input_datas:
            input_data = self.reshape_input(input_data)
            intermediate_layer_activations = self.get_model_activations(input_data)

            for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
                layer_activations = intermediate_layer_output[0]
                activations_shape = layer_activations.shape
                for neuron_index in range(self.neuron_nums(activations_shape)):
                    neuron_activation = layer_activations[np.unravel_index(neuron_index, activations_shape)]  # 多维展开成一维
                    if neuron_activation > self.threshold:
                        self.batch_dict[(layer_name, neuron_index)] = True

    def get_coverage(self) -> dict:
        covered_neurons = sum(neuron for neuron in self.coverage_dict.values() if neuron)
        total_neurons = len(self.coverage_dict)
        batch_covered_neurons = sum(neuron for neuron in self.batch_dict.values() if neuron)
        return {
            'total_neurons': total_neurons,
            'covered_neurons': covered_neurons,
            'neuron_coverage': covered_neurons / float(total_neurons),
            'batch_covered_neurons': batch_covered_neurons,
            'batch_neuron_coverage': batch_covered_neurons / float(total_neurons)
        }
