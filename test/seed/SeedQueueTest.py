import cv2

from utils.ConverterUtils import path_2_ndarray_convert
from seed.BaseSeedQueue import BaseSeedQueue

seedQueue = BaseSeedQueue()
for i in range(1, 6):
    seedQueue.push(path_2_ndarray_convert("D:\\Github\\nfuzz\\test\\images\\" + str(i) +".jpg"))

while not seedQueue.empty():
    cv2.imshow("image", seedQueue.pop())
    cv2.waitKey(0)
else:
    cv2.destroyAllWindows()
    cv2.destroyWindow("image")