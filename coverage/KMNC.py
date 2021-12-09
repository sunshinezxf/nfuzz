from coverage.abstract_coverage import AbstractCoverage
import numpy as np


class KMultiSectionNeuronCoverage(AbstractCoverage):
    """
    k-multiSection neuron coverage
    """

    def __init__(self, model, k=3):
        AbstractCoverage.__init__(self, model)
        self.coverage_dict = {}
        self.neuron_bounds_dict = self.init_dict()
        self.k = k

    def update_coverage(self, input_data):
        self.coverage_dict = self.init_k_multi_dict()

        layer_names = [layer.name for layer in self.layers]
        intermediate_layer_activations = self.get_model_activations(input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            for neuron_index in range(self.neuron_nums(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                lower, upper, step_size = self.neuron_bounds_dict[(layer_name, neuron_index)]
                if step_size > 0:
                    activated_section = int((neuron_activation - lower) / step_size)
                    if 0 <= activated_section < self.k:
                        self.coverage_dict[(layer_name, neuron_index)][activated_section] = True

    def update_neuron_bounds(self, input_data):
        """
        更新神经元边界
        :param input_data: 应该是一批数据得出的边界
        :return:
        """
        layer_names = [layer.name for layer in self.layers]

        intermediate_layer_activations = self.get_model_activations(input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            for neuron_index in range(self.neuron_nums(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                neuron_position = (layer_name, neuron_index)
                if not self.neuron_bounds_dict[neuron_position]:  # 初始化
                    step_size = 0
                    self.neuron_bounds_dict[neuron_position] = (neuron_activation, neuron_activation, step_size)
                else:
                    # 更新边界
                    (lower, upper, step_size) = self.neuron_bounds_dict[neuron_position]
                    if neuron_activation > upper:
                        step_size = (neuron_activation - lower) / self.k
                        self.neuron_bounds_dict[neuron_position] = (lower, neuron_activation, step_size)
                    elif neuron_activation < lower:
                        step_size = (upper - neuron_activation) / self.k
                        self.neuron_bounds_dict[neuron_position] = (neuron_activation, upper, step_size)

    def init_k_multi_dict(self) -> dict:
        """
        每个神经元分为k个section
        :return:
        """
        coverage_dict = {}
        for layer in self.layers:
            layer_name = layer.name
            for neuron_index in range(self.neuron_nums(layer.output_shape)):  # product of dims
                coverage_dict[(layer_name, neuron_index)] = [False] * self.k
        return coverage_dict

    def sections_covered(self):
        covered_sections = sum(sum(section for section in neuron) for neuron in self.coverage_dict.values())
        total_sections = len(self.coverage_dict) * self.k
        return covered_sections, total_sections, covered_sections / float(total_sections)

    def get_coverage(self) -> dict:
        covered_sections, total_sections, sections_covered_percentage = self.sections_covered()
        return {
            'total_sections': total_sections,
            'covered_sections': covered_sections,
            'k-multiSection_coverage': sections_covered_percentage
        }
