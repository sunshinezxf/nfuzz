import cv2
from muoperator.TransposeMutator import TransposeMutator
from utils.ConverterUtils import path_2_ndarray_convert

# test transpose_mutator
transpose_mutator = TransposeMutator()
original_img = path_2_ndarray_convert("D:\\Github\\nfuzz\\test\\images\\2.jpg")
# show original image
cv2.imshow("image", original_img)
cv2.waitKey(0)
# mutate image
img = transpose_mutator.mutate(original_img)
# show mutated image
cv2.imshow("image", img)
cv2.waitKey(0)
