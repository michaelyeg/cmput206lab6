import cv2, math
from matplotlib import pyplot as plt
import numpy as np
import copy, collections
import scipy.ndimage


class image:
    def __init__(self, filename):
        self.image = cv2.imread(filename, 0)
        self.filtered = []
        self.height = 0
        self.width = 0
        self.getdimension()

    def getdimension(self):
        dimension = self.image.shape
        self.height = dimension[0]
        self.width = dimension[1]
        return

    def gaussianBlur(self, sigma):
        k = int(2*round(3*sigma)+1)
        self.filtered = cv2.GaussianBlur(self.image, (k, k), sigma)

    # Create a Laplacian-of-Gaussian Volume
    def LoG(self, sigma):
        k = int(2*round(3*sigma)+1)
        self.LoGed = cv2.GaussianBlur(self.filtered, (k, k), sigma)
        self.LoGed = cv2.Laplacian(self.LoGed, ddepth=cv2.CV_64F, ksize=k)

    def blob(self, LoG):
        # Detect local minima
        localmin = scipy.ndimage.filters.minimum_filter(LoG, size=8, mode='reflect', cval=0.0, origin=0)
        # Convert local min values to binary mask
        mask = (LoG == localmin)
        mask = np.sum(mask, axis=2)
        x, y = np.nonzero(mask)
        plt.scatter(y, x, c='red')
        displayVertical({'Rough blobs detected in image': self.image})


# Put display list as a dictionary {"Title": image}
def displayVertical(imageList):
    keys = imageList.keys()
    if len(imageList) == 2:
        plt.subplot(211), plt.imshow(imageList[keys[0]]), plt.title(keys[0])
        plt.subplot(212), plt.imshow(imageList[keys[1]]), plt.title(keys[1])
        plt.show()
    elif len(imageList) == 3:
        plt.subplot(311), plt.imshow(imageList[keys[0]]), plt.title(keys[0])
        plt.subplot(312), plt.imshow(imageList[keys[1]]), plt.title(keys[1])
        plt.subplot(313), plt.imshow(imageList[keys[2]]), plt.title(keys[2])
        plt.show()
    elif len(imageList) == 1:
        plt.imshow(imageList[keys[0]]), plt.title(keys[0])
        plt.show()


def main():
    img = image("lab6.bmp")
    img.gaussianBlur(2)
    displayVertical({"Input Image": img.image, "Blurred Image": img.filtered})

    # 3 level of LoG
    img.LoG(3)
    # Making a deep copy!
    level1 = copy.copy(img.LoGed)
    img.LoG(4)
    level2 = copy.copy(img.LoGed)
    img.LoG(5)
    level3 = copy.copy(img.LoGed)
    displayVertical(collections.OrderedDict([("Level 1", level1), ("Level 2", level2), ("level 3", level3)]))

    # Part 2. Obtain a rough estimate of blob locations
    LoG = np.zeros((img.height, img.width, 3), np.float32)
    LoG[:, :, 0] = level1
    LoG[:, :, 1] = level2
    LoG[:, :, 2] = level3
    img.blob(LoG)

if __name__ == "__main__":
    main()
