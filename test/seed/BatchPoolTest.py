import cv2

from seed.BaseSeedQueue import BaseSeedQueue
from seed.BatchPool import BatchPool
from seed.generator.ImageRandomGenerator import ImageRandomGenerator

# # load images
# seed_queue = BaseSeedQueue()
# for i in range(1, 11):
#     seed_queue.push(path_2_ndarray_convert("D:\\Github\\nfuzz\\test\\images\\" + str(i) +".jpg"))
#
# # construct pool
# batchPool = BatchPool(seed_queue=seed_queue, batch_size=3, p_min=0.2, gamma=1)
# batchPool.preprocess()

# load images
path_list = []
for i in range(1, 11):
    path_list.append("D:\\Github\\nfuzz\\test\\images\\" + str(i) + ".jpg")

# construct pool, this is a empty seed pool
batchPool = BatchPool(seed_queue=BaseSeedQueue(), batch_size=3, p_min=0.2, gamma=1)
# add the valid seeds that already exists
batchPool.add_seed(path_list)
# randomly generated seeds
batchPool.random_generate(5, ImageRandomGenerator((400, 400, 3)))
# assemble into batch
batchPool.preprocess()


# test initial states
for element in batchPool.pool:
    print("fuzzed_times:" + str(element["fuzzed_times"]))

# get batch from pool 10000 times
for i in range(10000):
    batch = batchPool.select_next()

# random show an image
for i in range(1):
    batch = batchPool.select_next()
    print(type(batch.shape))
    print(type(batch))
    cv2.imshow("image", batch[0])
    cv2.imwrite("D:\\Github\\nfuzz\\test\\imgbuf\\test1.png", batch[0])
    cv2.waitKey(0)
    cv2.imshow("image", batch[1])
    cv2.waitKey(0)
    cv2.imshow("image", batch[2])
    cv2.waitKey(0)

# test end states
for element in batchPool.pool:
    print("fuzzed_times:" + str(element["fuzzed_times"]))

print("gamma: " + str(batchPool.gamma))


# print("-------------------------继续加数据-----------------------------")
#
# # load images
# for i in range(10):
#     if random.random() > 0.5:
#         seed_queue.push(path_2_ndarray_convert(
#             "https://bkimg.cdn.bcebos.com/pic/35a85edf8db1cb1316f5387ed254564e92584b3a?x-bc"
#             "e-process=image/watermark,g_7,image_d2F0ZXIvYmFpa2UxNTA=,xp_5,yp_5"))
#     else:
#         seed_queue.push(path_2_ndarray_convert("C:\\Users\\lenovo\\Desktop\\testimage1.jpg"))
#
# batchPool.preprocess()
#
# # test initial states
# for element in batchPool.pool:
#     print("fuzzed_times:" + str(element["fuzzed_times"]))
#
# # get batch from pool 10000 times
# for i in range(20000):
#     batch = batchPool.select_next()
#
# # random show an image
# for i in range(1):
#     batch = batchPool.select_next()
#     print(batch.shape)
#     print(type(batch))
#     cv2.imshow("image", batch[0])
#     cv2.waitKey(0)
#     cv2.imshow("image", batch[1])
#     cv2.waitKey(0)
#     cv2.imshow("image", batch[2])
#     cv2.waitKey(0)
#
# # test end states
# for element in batchPool.pool:
#     print("fuzzed_times:" + str(element["fuzzed_times"]))
#
# print("gamma: " + str(batchPool.gamma))





# test save
# batchPool.save("D:\\buf_file\\py")