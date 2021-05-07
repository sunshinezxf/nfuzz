import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.datasets import mnist

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_train = x_train.astype("float32")
    y_train = y_train.astype("float32")
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_test = x_test.astype("float32")
    y_test = y_test.astype("float32")
    x_train /= 255
    x_test /= 255

    y_train_new = np_utils.to_categorical(num_classes=10, y=y_train)
    y_test_new = np_utils.to_categorical(num_classes=10, y=y_test)
    return x_train,y_train,x_test,y_test,y_train_new,y_test_new


model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid', input_shape=(28, 28, 1), activation='tanh')) #C1
model.add(MaxPooling2D(pool_size=(2, 2))) #S2
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh')) #C3
model.add(MaxPooling2D(pool_size=(2, 2))) #S4
model.add(Flatten())
model.add(Dense(120, activation='tanh')) #C5
model.add(Dense(84, activation='tanh')) #F6
model.add(Dense(10, activation='softmax')) #output
model.summary()

x_train,y_train,x_test,y_test,y_train_new,y_test_new=load_mnist()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
history=model.fit(x_train,y_train_new,batch_size=64,epochs=2,verbose=1,validation_split=0.2,shuffle=True)
loss,accuracy = model.evaluate(x_test,y_test_new)
print(loss,accuracy)
print(history.history)

# -> https://github.com/keras-team/keras/issues/2378
model.save("../model/LeNet5.h5",overwrite=True)


