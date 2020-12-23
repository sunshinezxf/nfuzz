import cv2
from muoperator.TransposeMutator import TransposeMutator
from myUtils.ConverterUtils import path_2_ndarray_convert

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
cv2.imshow("image",x_train[0])
cv2.waitKey(0)

#
# # test transpose_mutator
# transpose_mutator = TransposeMutator()
# original_img = path_2_ndarray_convert("../../untitiled.png")
# # show original image
# cv2.imshow("image", original_img)
# cv2.waitKey(0)
# # mutate image
# img = transpose_mutator.mutate(original_img)
# # show mutated image
# cv2.imshow("image", img)
# cv2.waitKey(0)
