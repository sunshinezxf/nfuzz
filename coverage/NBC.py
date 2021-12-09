from coverage.abstract_coverage import AbstractCoverage
import numpy as np

class NeuronBoundaryCoverage(AbstractCoverage):
    """
    计算NBC和SNAC
    neuron boundary coverage = ( |UpperCornerNeuron| + |LowerCornerNeuron| ) / ( 2 × |N| )
    strong neuron activation coverage = |UpperCornerNeuron| / |N|
    """
    def __init__(self, model):
        AbstractCoverage.__init__(self,model)
        self.coverage_dict={}
        self.snac_dict = {}
        self.neuron_bounds_dict = self.init_dict()

    def update_neuron_bounds(self, input_data):
        """
        更新边界
        :param input_data:
        :return:
        """
        input_data=self.reshape_input(input_data)

        layer_names = [layer.name for layer in self.layers]

        intermediate_layer_activations = self.get_model_activations(input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            for neuron_index in range(self.neuron_nums(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                neuron_position = (layer_name, neuron_index)
                if not self.neuron_bounds_dict[neuron_position]:  # 初始化
                    self.neuron_bounds_dict[neuron_position] = (neuron_activation, neuron_activation)
                else:
                    (lower, upper) = self.neuron_bounds_dict[neuron_position]
                    if neuron_activation > upper:
                        self.neuron_bounds_dict[neuron_position] = (lower, neuron_activation)
                    elif neuron_activation < lower:
                        self.neuron_bounds_dict[neuron_position] = (neuron_activation, upper)

    def update_coverage(self, input_data):
        """
        更新NBC
        :param input_data:
        :return:
        """
        self.coverage_dict = self.init_neuron_boundary_dict()
        layer_names = [layer.name for layer in self.layers]
        intermediate_layer_outputs = self.get_model_activations(input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_outputs):
            layer_activations = intermediate_layer_output[0]
            for neuron_index in range(self.neuron_nums(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                low, high = self.neuron_bounds_dict[(layer_name, neuron_index)]
                if neuron_activation > high:
                    self.coverage_dict[(layer_name, neuron_index)] = (
                        self.coverage_dict[(layer_name, neuron_index)][0], True)
                elif neuron_activation < low:
                    self.coverage_dict[(layer_name, neuron_index)] = (
                        True, self.coverage_dict[(layer_name, neuron_index)][1])

    def update_strong_neuron_activation_coverage(self,input_data):
        """
        更新SNAC
        :param input_data:
        :return:
        """
        self.snac_dict=self.init_dict()

        layer_names = [layer.name for layer in self.layers]
        intermediate_layer_activations = self.get_model_activations(input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            for neuron_index in range(self.neuron_nums(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                low, high = self.neuron_bounds_dict[(layer_name, neuron_index)]
                if neuron_activation > high:
                    self.snac_dict[(layer_name, neuron_index)] = True

    def init_neuron_boundary_dict(self) -> dict:
        """
        更新边界
        :return:
        """
        coverage_dict = {}
        for layer in self.layers:
            for index in range(self.neuron_nums(layer.output_shape)):
                coverage_dict[(layer.name, index)] = (False, False)
        return coverage_dict

    def snac_covered(self):
        covered_neurons = sum(neuron for neuron in self.snac_dict.values() if neuron)
        total_neurons = len(self.snac_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def corners_covered(self):
        covered_corners = sum(sum(corner for corner in neuron) for neuron in self.coverage_dict.values())
        total_corners = len(self.coverage_dict) * 2
        return covered_corners, total_corners, covered_corners / float(total_corners),

    def get_coverage(self) -> dict:
        covered_corners, total_corners, corners_covered_percentage = self.corners_covered()
        covered_neurons, total_neurons, snac=self.snac_covered()
        return {
            'total_corners': total_corners,
            'covered_corners': covered_corners,
            'neuron_boundary_coverage': corners_covered_percentage,
            'snac_total_neurons': total_neurons,
            'snac_covered_neurons': covered_neurons,
            'strong_neuron_activation_coverage': snac
        }
