# https://github.com/mlxyz/insynth/blob/master/insynth/metrics/coverage/neuron.py

import random
from collections import defaultdict
import numpy as np
from abc import ABC, abstractmethod


class AbstractCoverageCalculator(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def update_coverage(self, input_data) -> dict:
        pass

    @abstractmethod
    def get_random_uncovered_neuron(self):
        pass

def num_neurons(shape):
    return np.prod([dim for dim in shape if dim is not None])


def _init_dict(model) -> dict:
    coverage_dict = defaultdict(bool)
    for layer in get_layers_with_neurons(model):
        for index in range(num_neurons(layer.output_shape)):  # product of dims
            coverage_dict[(layer.name, index)] = False
    return coverage_dict


def get_layers_with_neurons(model):
    return [layer for layer in model.layers if
            'flatten' not in layer.name and 'input' not in layer.name]


def get_model_activations(model, input_data):
    from tensorflow import keras
    layers = get_layers_with_neurons(model)
    intermediate_layer_model = keras.models.Model(inputs=model.input,
                                                  outputs=[layer.output for layer in
                                                           layers])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    return intermediate_layer_outputs


def neurons_covered(coverage_dict):
    covered_neurons = sum(neuron for neuron in coverage_dict.values() if neuron)
    total_neurons = len(coverage_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def get_random_uncovered_neuron(coverage_dict):
    uncovered_neurons = [key for key, covered in coverage_dict.items() if not covered]
    if uncovered_neurons:
        return random.choice(uncovered_neurons)
    else:
        return random.choice(coverage_dict.keys())


class NeuronCoverageCalculator(AbstractCoverageCalculator):
    def get_random_uncovered_neuron(self):
        return get_random_uncovered_neuron(self.coverage_dict)

    def __init__(self, model, activation_threshold=0):
        super().__init__(model)
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        self.activation_threshold = activation_threshold
        self.coverage_dict = _init_dict(model)

    def update_coverage(self, input_data):
        layers = self._layers_with_neurons
        layer_names = [layer.name for layer in layers]

        intermediate_layer_activations = get_model_activations(self.model, input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            activations_shape = layer_activations.shape
            for neuron_index in range(num_neurons(activations_shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, activations_shape)]
                if neuron_activation > self.activation_threshold:
                    self.coverage_dict[(layer_name, neuron_index)] = True

    def get_coverage(self) -> dict:
        covered_neurons, total_neurons, covered_percentage = neurons_covered(self.coverage_dict)
        return {
            'total_neurons': len(self.coverage_dict),
            'covered_neurons': covered_neurons,
            'covered_neurons_percentage': covered_percentage
        }


class StrongNeuronActivationCoverageCalculator(AbstractCoverageCalculator):
    def __init__(self, model):
        super().__init__(model)
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        self.coverage_dict = _init_dict(model)
        self.neuron_bounds_dict = _init_dict(model)

    def get_random_uncovered_neuron(self):
        return get_random_uncovered_neuron(self.coverage_dict)

    def update_neuron_bounds(self, input_data):
        layers = self._layers_with_neurons
        layer_names = [layer.name for layer in layers]

        intermediate_layer_activations = get_model_activations(self.model, input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            for neuron_index in range(num_neurons(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                neuron_position = (layer_name, neuron_index)
                if not self.neuron_bounds_dict[neuron_position]:
                    self.neuron_bounds_dict[neuron_position] = (neuron_activation, neuron_activation)
                else:
                    (lower, upper) = self.neuron_bounds_dict[neuron_position]
                    if neuron_activation > upper:
                        self.neuron_bounds_dict[neuron_position] = (lower, neuron_activation)
                    elif neuron_activation < lower:
                        self.neuron_bounds_dict[neuron_position] = (neuron_activation, upper)

    def update_coverage(self, input_data):
        layers = self._layers_with_neurons
        layer_names = [layer.name for layer in layers]

        intermediate_layer_activations = get_model_activations(self.model, input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            for neuron_index in range(num_neurons(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                _, high = self.neuron_bounds_dict[(layer_name, neuron_index)]
                if neuron_activation > high:
                    self.coverage_dict[(layer_name, neuron_index)] = True

    def get_coverage(self) -> dict:
        covered_neurons, total_neurons, covered_percentage = neurons_covered(self.coverage_dict)
        return {
            'total_neurons': len(self.coverage_dict),
            'covered_neurons': covered_neurons,
            'covered_neurons_percentage': covered_percentage
        }


class KMultiSectionNeuronCoverageCalculator(AbstractCoverageCalculator):
    def __init__(self, model, k=3):
        super().__init__(model)
        self.k = k
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        self.neuron_bounds_dict = _init_dict(model)
        self.coverage_dict = self._init_dict(model)

    def update_coverage(self, input_data):
        layers = self._layers_with_neurons
        layer_names = [layer.name for layer in layers]

        intermediate_layer_activations = get_model_activations(self.model, input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            for neuron_index in range(num_neurons(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                lower, upper, step_size = self.neuron_bounds_dict[(layer_name, neuron_index)]
                if step_size > 0:
                    activated_section = int((neuron_activation - lower) / step_size)
                    if 0 <= activated_section < self.k:
                        self.coverage_dict[(layer_name, neuron_index)][activated_section] = True

    def get_random_uncovered_neuron(self):
        uncovered_neurons = [key for key, covered in self.coverage_dict.items() if not all(covered)]
        if uncovered_neurons:
            return random.choice(uncovered_neurons)
        else:
            return random.choice(list(self.coverage_dict.keys()))

    def update_neuron_bounds(self, input_data):
        layers = self._layers_with_neurons
        layer_names = [layer.name for layer in layers]

        intermediate_layer_activations = get_model_activations(self.model, input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            for neuron_index in range(num_neurons(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                neuron_position = (layer_name, neuron_index)
                if not self.neuron_bounds_dict[neuron_position]:
                    step_size = 0
                    self.neuron_bounds_dict[neuron_position] = (neuron_activation, neuron_activation, step_size)
                else:
                    (lower, upper, step_size) = self.neuron_bounds_dict[neuron_position]
                    if neuron_activation > upper:
                        step_size = (neuron_activation - lower) / self.k
                        self.neuron_bounds_dict[neuron_position] = (lower, neuron_activation, step_size)
                    elif neuron_activation < lower:
                        step_size = (upper - neuron_activation) / self.k
                        self.neuron_bounds_dict[neuron_position] = (neuron_activation, upper, step_size)

    def _init_dict(self, model):
        coverage_dict = {}
        for layer in get_layers_with_neurons(model):
            layer_name = layer.name
            for neuron_index in range(num_neurons(layer.output_shape)):  # product of dims
                coverage_dict[(layer_name, neuron_index)] = [False] * self.k
        return coverage_dict

    def neurons_covered(self):
        covered_neurons = sum(all(neuron) for neuron in self.coverage_dict.values())
        total_neurons = len(self.coverage_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def sections_covered(self):
        covered_sections = sum(sum(section for section in neuron) for neuron in self.coverage_dict.values())
        total_sections = len(self.coverage_dict) * self.k
        return covered_sections, total_sections, covered_sections / float(total_sections)

    def get_coverage(self) -> dict:
        covered_neurons, total_neurons, covered_percentage = self.neurons_covered()
        covered_sections, total_sections, sections_covered_percentage = self.sections_covered()
        return {
            'total_neurons': total_neurons,
            'covered_neurons': covered_neurons,
            'covered_neurons_percentage': covered_percentage,
            'total_sections': total_sections,
            'covered_sections': covered_sections,
            'sections_covered_percentage': sections_covered_percentage,
        }


class NeuronBoundaryCoverageCalculator(AbstractCoverageCalculator):
    def __init__(self, model):
        super().__init__(model)
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        self.coverage_dict = self._init_dict(model)
        self.neuron_bounds_dict = _init_dict(model)

    def get_random_uncovered_neuron(self):
        uncovered_neurons = [key for key, covered in self.coverage_dict.items() if not all(covered)]
        if uncovered_neurons:
            return random.choice(uncovered_neurons)
        else:
            return random.choice(list(self.coverage_dict.keys()))

    def update_neuron_bounds(self, input_data):
        layers = self._layers_with_neurons
        layer_names = [layer.name for layer in layers]

        intermediate_layer_activations = get_model_activations(self.model, input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            for neuron_index in range(num_neurons(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                neuron_position = (layer_name, neuron_index)
                if not self.neuron_bounds_dict[neuron_position]:
                    self.neuron_bounds_dict[neuron_position] = (neuron_activation, neuron_activation)
                else:
                    (lower, upper) = self.neuron_bounds_dict[neuron_position]
                    if neuron_activation > upper:
                        self.neuron_bounds_dict[neuron_position] = (lower, neuron_activation)
                    elif neuron_activation < lower:
                        self.neuron_bounds_dict[neuron_position] = (neuron_activation, upper)

    def update_coverage(self, input_data):
        layers = self._layers_with_neurons
        layer_names = [layer.name for layer in layers]

        intermediate_layer_activations = get_model_activations(self.model, input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            for neuron_index in range(num_neurons(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                low, high = self.neuron_bounds_dict[(layer_name, neuron_index)]
                if neuron_activation > high:
                    self.coverage_dict[(layer_name, neuron_index)] = (
                        self.coverage_dict[(layer_name, neuron_index)][0], True)
                elif neuron_activation < low:
                    self.coverage_dict[(layer_name, neuron_index)] = (
                        True, self.coverage_dict[(layer_name, neuron_index)][1])

    def neurons_covered(self):
        covered_neurons = sum(all(neuron) for neuron in self.coverage_dict.values())
        total_neurons = len(self.coverage_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def corners_covered(self):
        covered_corners = sum(sum(corner for corner in neuron) for neuron in self.coverage_dict.values())
        total_corners = len(self.coverage_dict) * 2
        return covered_corners, total_corners, covered_corners / float(total_corners)

    def get_coverage(self) -> dict:
        covered_neurons, total_neurons, covered_percentage = self.neurons_covered()
        covered_corners, total_corners, corners_covered_percentage = self.corners_covered()
        return {
            'total_neurons': total_neurons,
            'covered_neurons': covered_neurons,
            'covered_neurons_percentage': covered_percentage,
            'total_corners': total_corners,
            'covered_corners': covered_corners,
            'corners_covered_percentage': corners_covered_percentage,
        }

    def _init_dict(self, model) -> dict:
        coverage_dict = {}
        for layer in get_layers_with_neurons(model):
            for index in range(num_neurons(layer.output_shape)):  # product of dims
                coverage_dict[(layer.name, index)] = (False, False)
        return coverage_dict


class TopKNeuronCoverageCalculator(AbstractCoverageCalculator):
    def get_random_uncovered_neuron(self):
        uncovered_neurons = []
        for layer in get_layers_with_neurons(self.model):
            for neuron_index in range(num_neurons(layer.output_shape)):
                if neuron_index not in self.coverage_dict[layer.name]:
                    uncovered_neurons.append((layer.name, neuron_index))
        if uncovered_neurons:
            return random.choice(uncovered_neurons)
        else:
            return None

    def __init__(self, model, k=3):
        super().__init__(model)
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        self.coverage_dict = self._init_dict(model)
        self.k = k

    def _init_dict(self, model) -> dict:
        coverage_dict = {}
        for layer in get_layers_with_neurons(model):
            coverage_dict[layer.name] = set()
        return coverage_dict

    def update_coverage(self, input_data):
        layers = self._layers_with_neurons
        layer_names = [layer.name for layer in layers]

        intermediate_layer_activations = get_model_activations(self.model, input_data)

        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            neuron_activations = []
            for neuron_index in range(num_neurons(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                neuron_activations.append((neuron_index, neuron_activation))
            self.coverage_dict[layer_name] |= set(
                map(lambda x: x[0], sorted(neuron_activations, key=lambda x: x[1])[-self.k:]))

    def get_coverage(self) -> dict:
        top_k_neurons = sum(len(layer) for layer in self.coverage_dict.values())
        total_neurons = sum(num_neurons(layer.output_shape) for layer in get_layers_with_neurons(self.model))
        return {
            'total_neurons': total_neurons,
            'top_k_neurons': top_k_neurons,
            'top_k_neuron_coverage_percentage': top_k_neurons / total_neurons
        }


class TopKNeuronPatternsCalculator(AbstractCoverageCalculator):
    def get_random_uncovered_neuron(self):
        pass

    def __init__(self, model, k=3):
        super().__init__(model)
        self.k = k
        self.coverage_dict = self._init_dict()

    def _init_dict(self) -> set:
        coverage_dict = set()
        return coverage_dict

    def update_coverage(self, input_data):
        self._layers_with_neurons = get_layers_with_neurons(self.model)
        layers = self._layers_with_neurons
        layer_names = [layer.name for layer in layers]

        intermediate_layer_activations = get_model_activations(self.model, input_data)
        neuron_activations = []
        for layer_name, intermediate_layer_output in zip(layer_names, intermediate_layer_activations):
            layer_activations = intermediate_layer_output[0]
            layer_neuron_activations = []
            for neuron_index in range(num_neurons(layer_activations.shape)):
                neuron_activation = layer_activations[np.unravel_index(neuron_index, layer_activations.shape)]
                layer_neuron_activations.append((neuron_index, neuron_activation))
            neuron_activations.extend(
                map(lambda x: layer_name + '_' + str(x[0]),
                    sorted(layer_neuron_activations, key=lambda x: x[1])[-self.k:]))
        self.coverage_dict |= {tuple(neuron_activations)}

    def get_coverage(self) -> dict:
        return {
            'total_patterns': len(self.coverage_dict),
        }