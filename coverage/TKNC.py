from coverage.abstract_coverage import AbstractCoverage
import numpy as np

class TopKNeuronCoverage(AbstractCoverage):
    """
    top-k neuron coverage
    度量的是每一层中成为激活值最大的前k个神经元的数量，故其定义为每层top-k神经元总数与DNN神经元总数的比值
    """
    def __init__(self, model, k=3):
        AbstractCoverage.__init__(self,model)
        self.coverage_dict = {}
        self.k = k

    def init_top_k_dict(self) -> dict:
        coverage_dict = {}
        for layer in self.layers:
            coverage_dict[layer.name] = set()
        return coverage_dict

    def update_coverage(self, input_data):
        self.coverage_dict = self.init_top_k_dict()
        layer_names = [layer.name for layer in self.layers]
        intermediate_layer_activations = self.get_model_activations(input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            neuron_activations = []
            for neuron_index in range(self.neuron_nums(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                neuron_activations.append((neuron_index, neuron_activation))
            # 取前k个activation最大的neuron index,做并集运算
            self.coverage_dict[layer_name] |= set(
                map(lambda x: x[0], sorted(neuron_activations, key=lambda x: x[1])[-self.k:]))

    def get_coverage(self) -> dict:
        top_k_neurons = sum(len(layer) for layer in self.coverage_dict.values())
        total_neurons = sum(self.neuron_nums(layer.output_shape) for layer in self.layers)
        return {
            'total_neurons': total_neurons,
            'top_k_neurons': top_k_neurons,
            'top_k_neuron_coverage': top_k_neurons / total_neurons
        }