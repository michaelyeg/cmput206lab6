import numpy as np
import scipy.ndimage
import cv2
import math
import matplotlib.pyplot as plt


def getLoGKernel(ksize, sigma):
    kernel = np.zeros([int(ksize), int(ksize)], dtype=np.float32)
    range = int(ksize / 2.0)
    const_1 = 1.0 / (np.pi * (sigma ** 4))
    const_2 = 1.0 / (2 * sigma * sigma)
    i = 0
    for x in xrange(-range, range):
        j = 0
        for y in xrange(-range, range):
            factor = (x * x + y * y) * const_2
            kernel[i, j] = const_1 * (1 - factor) * math.exp(-factor)
            j += 1
        i += 1
    return kernel


def applyLoG(img, ksize, sigma):
    img_gauss = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE)
    return cv2.Laplacian(img_gauss, -1, ksize, borderType=cv2.BORDER_REPLICATE)


sig = 2.5
min_scale = 3
max_scale = 5
use_builtin_log = 1
min_filter_size = 12
scatter_size = 40
scatter_col = 'r'
contour_col = (0, 0, 0)

img_name = 'lab6.bmp'

fig_id = 0
fig = []

I_rgb = cv2.imread(img_name).astype(np.float32)
I = cv2.cvtColor(I_rgb, cv2.COLOR_BGR2GRAY)

J = cv2.GaussianBlur(I, (int(2 * round(3 * sig) + 1), int(2 * round(3 * sig) + 1)), sig,
                     borderType=cv2.BORDER_REPLICATE)

fig.append(plt.figure(fig_id))
fig_id += 1
plt.subplot(2, 1, 1)
plt.imshow(I)
plt.title('Input Image')
plt.subplot(2, 1, 2)
plt.imshow(J)
plt.title('Blurred Image')

# Cell centre detection by Blob detector and fine tuning by Otsu
[h, w] = I.shape
K = np.zeros([h, w, 3])
for scale in xrange(min_scale, max_scale + 1):
    kernel_size = int(2 * math.floor(3 * scale) + 1)
    if use_builtin_log:
        K[:, :, scale - min_scale] = applyLoG(J, kernel_size, scale)
    else:
        log_kernel = getLoGKernel(kernel_size, scale)
        K[:, :, scale - min_scale] = cv2.filter2D(J, -1, log_kernel)

fig.append(plt.figure(fig_id))
fig_id += 1
plt.subplot(3, 1, 1)
level1 = K[:, :, 0]
plt.imshow(level1)
plt.title('Level 1')
plt.subplot(3, 1, 2)
level2 = K[:, :, 1]
plt.imshow(level2)
plt.title('Level 2')
plt.subplot(3, 1, 3)
level3 = K[:, :, 2]
plt.imshow(level3)
plt.title('Level 3')
fig[-1].suptitle('LoG Pyramid')

# local maxima within the volume
lm = scipy.ndimage.filters.minimum_filter(K, min_filter_size)
A = (K == lm)

# take minima within a range of scales and collapse those on x-y plane
B = np.sum(A, axis=2)

[y, x] = B.nonzero()
fig.append(plt.figure(fig_id))
fig_id += 1
plt.imshow(I)
plt.scatter(x, y, marker='.', color=scatter_col, s=scatter_size)
plt.xlim([0, I.shape[1]])
plt.ylim([0, I.shape[0]])
plt.grid(b=False)
plt.title('Rough blobs detected in the image')

# remove spurious maxima by working with Otsu threshold
t, J_otsu = cv2.threshold(J.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
B = np.multiply(B, J >= t)

[y, x] = B.nonzero()
fig.append(plt.figure(fig_id))
fig_id += 1
plt.imshow(I)
plt.scatter(x, y, marker='.', color=scatter_col, s=scatter_size)
plt.xlim([0, I.shape[1]])
plt.ylim([0, I.shape[0]])
plt.grid(b=False)
plt.title('Refined blobs detected in the image')

plt.show()