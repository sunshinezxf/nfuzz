import cv2
from muoperator.TransposeMutator import TransposeMutator
from myUtils.converterUtils import path_2_ndarray_convert
from seed import BatchPool
from seed.generator import ImageRandomGenerator as ImageGenerator
from seed import BaseSeedQueue as seedQueue
from muoperator import Mutator
from nothing import keras_test

seed_generator = ImageGenerator.ImageRandomGenerator(None)

seed = seed_generator.generate()  # 随机生成or导入

seed_queue = seedQueue.SeedQueue()

mutator=Mutator.Mutator()

# 存入种子队列
for img in seed:
    seed_queue.push(img)


pool = BatchPool.BatchPool(seed_queue)
pool.preprocess()


batch=pool.select_next()

new_seed_batch=mutator.mutate(batch)






def main():
    pass

# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# cv2.imshow("image",x_train[0])
# cv2.waitKey(0)

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
