# cmput206lab6

Part 1. Create a Laplacian-of-Gaussian Volume (30%)
1. Read this image and convert it to gray scale. (2%)
2. Apply Gaussian filtering using  cv2.GaussianBlur with a suitably chosen sigma and the kernel size k computed as 

 (3%)
3. Show both the original image and its blurred version together in the same figure (2%).
• The output should look like this.
4. Create a 3 level Laplacian-of-Gaussian (LoG) volume by applying this filter at 3 different scales or sigma values to the blurred image obtained in step 2 (20%) 
• The LoG filter can be applied using either cv2.GaussianBlur followed by cv2.Laplacian or using cv2.filter2D with a manually created kernel following the expression given here. 
• The kernel size should be computed using the same expression as in step 2
• The 3 sigma values should be chosen to give best results; using consecutive integers such as 3, 4 and 5 might be a good starting point
• All 3 levels of the volume must be stored in a single 

 Numpy array where h and w are the height and width of the input image
5. Display the 3 levels of the volume together in the same figure (3%)
• The output should look like this.
Part 2. Obtain a rough estimate of blob locations (40%)
1. Detect regional minima within the LoG volume (20%). 
• As described here, regional minima are defined as "connected components of pixels with a constant intensity value, and whose external boundary pixels all have a higher value". 
• OpenCV does not provide a function to perform this so you can either use the scipy function scipy.ndimage.filters.minimum_filter or implement it yourself, for instance, by adapting the code given here.  
• The output of this step will be a binary image of the same size as the LoG volume that contains 1 at locations of the regional minima and 0 everywhere else.
• The detected minima depend heavily on the parameters used for defining the region size. For instance, if  the scipy function is used, then its second argument "size" must be chosen carefully.
• It should also be noted that the scipy function returns the actual values of the detected minima rather than their locations so additional steps will be needed to convert its output to the required  binary image.
2. Collapse this 3D binary image into a single channel image by computing the sum of corresponding pixels in the 3 channels. This can be done using np.sum (10%).
3. Show the locations of all non zero entries in this collapsed array overlaid on the input image as red points (10%). 
• The locations of non zero entries can be found using np.nonzero while the plotting can be done using plt.scatter.
• The output should look like this.
Part 3. Refine the blobs using Otsu thresholding (30%)
1. Apply Otsu thresholding on the blurred image computed in step 2 of part 1 using  cv2.threshold to obtain the optimal threshold for this image (15%)
• A tutorial on Otsu thresholding can be found here.
2. Remove all minima detected in part 2 where the pixel values in this image are less than this threshold (12%)
3. Show the remaining minima locations overlaid on the input image as red points (3%)
• The output should look like this.
The solution containing all 3 parts must be submitted as a single file named lab6.py
