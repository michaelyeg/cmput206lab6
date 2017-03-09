import cv2, math
from matplotlib import pyplot as plt
import numpy as np
import copy

class image:
    def __init__(self, filename):
        self.image = cv2.imread(filename, 0)
        self.filtered = []

    def gaussianBlur(self, sigma):
        k = 2*round(3*sigma)+1
        self.filtered = cv2.GaussianBlur(self.image, (int(k), int(k)), sigma)

    def LoG(self, sigma):
        k = 2*round(3*sigma)+1
        self.LoGed = cv2.GaussianBlur(self.filtered, (int(k), int(k)), sigma)
        self.LoGed = cv2.Laplacian(self.LoGed, 1, ksize=int(k))


# Put display list as a dictionary {"Title": image}
def displayVertical(imageList):
    keys = imageList.keys()
    if len(imageList) == 2:
        plt.subplot(211), plt.imshow(imageList[keys[0]], 'gray'), plt.title(keys[0])
        plt.subplot(212), plt.imshow(imageList[keys[1]], 'gray'), plt.title(keys[1])
        plt.show()
    elif len(imageList) == 3:
        plt.subplot(311), plt.imshow(imageList[keys[0]], 'gray'), plt.title(keys[0])
        plt.subplot(312), plt.imshow(imageList[keys[1]], 'gray'), plt.title(keys[1])
        plt.subplot(313), plt.imshow(imageList[keys[2]], 'gray'), plt.title(keys[2])
        plt.show()
    elif len(imageList) == 1:
        plt.subplot(111), plt.imshow(imageList[keys[0]], 'gray'), plt.title(keys[0])
        plt.show()


def main():
    img = image("lab6.bmp")
    img.gaussianBlur(5)
    displayVertical({"Input Image": img.image, "Blurred Image": img.filtered})

    # 3 level of LoG
    img.LoG(3)
    # Making a deep copy!
    level1 = copy.copy(img.LoGed)
    img.LoG(4)
    level2 = copy.copy(img.LoGed)
    img.LoG(5)
    level3 = copy.copy(img.LoGed)
    displayVertical({"Level 1": level1, "Level 2": level2, "level 3": level3})

if __name__ == "__main__":
    main()