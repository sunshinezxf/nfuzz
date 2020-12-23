import cv2


def path_2_ndarray_convert(path):
    """
        Load image through file system path and convert to ndarray
        :param
            url -- file path
        :return
            ndarray -- 'numpy.ndarray'
    """
    # local image
    return cv2.imread(path)


