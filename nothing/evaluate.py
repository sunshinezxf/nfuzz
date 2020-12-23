from keras.models import load_model
from nothing import LeNet5
from keras import Model

model = load_model('model/LeNet5.h5')

x_train,y_train,x_test,y_test,y_train_new,y_test_new= LeNet5.load_mnist()
for layer in model.layers:
    for weight in layer.weights:
        print(weight.name,weight)

layer_name = 'conv2d_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(x_test[0])
print(intermediate_output)
loss,accuracy = model.evaluate(x_test,y_test_new)
print(loss,accuracy)




