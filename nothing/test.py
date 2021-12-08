from myUtils import csv_utils
from keras.models import load_model
from keras.utils import plot_model

model = load_model('../model/LeNet5.h5')
print(model.layers[0].input_shape)
print(model.layers[0].input_shape[0] is None)