import cv2

from utils.ConverterUtils import path_2_ndarray_convert
from seed.SeedQueue import SeedQueue

seedQueue = SeedQueue()
seedQueue.push(path_2_ndarray_convert("https://file.xiaohuashifu.top/user/avatar/5c1a1937-30b3-4039-8a42-cee4f7e1fd64.jpg"))
seedQueue.push(path_2_ndarray_convert("C:\\Users\\lenovo\\Desktop\\testimage1.jpg"))

cv2.imshow("image", seedQueue.pop())
cv2.waitKey(0)
cv2.imshow("image", seedQueue.pop())
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.destroyWindow("image")