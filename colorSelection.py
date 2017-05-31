import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread("family.jpg")
print("This image is {}, with dimensions: {}".format(type(image), image.shape))
# print(image[:, :, 1].shape)
xsize = image.shape[0]
ysize = image.shape[1]
color_select = np.copy(image)

#set the threshold
red_threshold = 254
green_threshold = 254
blue_threshold = 254
rgb_thresholds = [red_threshold, green_threshold, blue_threshold]

thresholds = (image[:, :, 0] < rgb_thresholds[0]) \
             | (image[:, :, 1] < rgb_thresholds[1]) \
             | (image[:, :, 2] < rgb_thresholds[2])

color_select[thresholds] = [0, 0, 0]
plt.imshow(color_select)
plt.show()            



