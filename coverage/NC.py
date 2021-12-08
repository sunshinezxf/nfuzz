import numpy as np
from coverage.abstract_coverage import AbstractCoverage

class NeuronCoverages(AbstractCoverage):
    """
    最基本的神经元覆盖率
    """

    def __init__(self, model, activation_threshold=0.25):
        AbstractCoverage.__init__(self,model)
        self.threshold = activation_threshold

    def update_coverage(self, input_data):
        input_data=self.reshape_input(input_data)
        layer_names = [layer.name for layer in self.layers]

        intermediate_layer_outputs = self.get_model_outputs(self.layers, input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_outputs):
            layer_activations = intermediate_layer_output[0]
            activations_shape = layer_activations.shape
            for neuron_index in range(self.neuron_nums(activations_shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, activations_shape)]  # 多维展开成一维
                if neuron_activation > self.threshold:
                    self.coverage_dict[(layer_name, neuron_index)] = True
                else:
                    self.coverage_dict[(layer_name, neuron_index)] = False

    def get_coverage(self):
        covered_neurons = sum(neuron for neuron in self.coverage_dict.values() if neuron)
        total_neurons = len(self.coverage_dict)
        return covered_neurons / float(total_neurons)
