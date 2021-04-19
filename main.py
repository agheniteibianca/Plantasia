import cv2
import numpy as np
import keyboard

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized





def Treshhold():
    image = cv2.imread('arici.jpg', 0)
    a = image.copy()
    width, height = image.shape

    for i in range (0,width):
        for j in range (height):
            if a[i][j] >= 125:
                a[i][j] =255
            else:
                a[i][j] =0

    image = image_resize(image, height = 400)
    cv2.imshow("Original " , image)

    a = image_resize(a, height = 400)
    cv2.imshow("Binarizare ", a)



if __name__ == '__main__':
    Treshhold()
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

