import cv2


def path_2_ndarray_convert(path):
    """
        Load image through HTTP request or file system path and convert to ndarray
        :param
            url -- http url | file system path
        :return
            ndarray -- 'numpy.ndarray'
        :except
            IOError -- open file failed
    """
    # http image
    if path.startswith("http"):
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            ret, img = cap.read()
            return img
        else:
            raise IOError("Open image failed.")

    # local image
    return cv2.imread(path)


