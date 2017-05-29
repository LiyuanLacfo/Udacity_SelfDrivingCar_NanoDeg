import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread("highway.jpg")
print("The type of image: {}, the shape of image: {}".format(type(image), image.shape))
def rgb2gray(image):
	return np.dot(image[:, :, :3], [0.299, 0.587, 0.114])
grayimg = rgb2gray(image)
print(grayimg.shape)
plt.imshow(rgb2gray(image), cmap="gray")
plt.show()

color_select = np.copy(image)
red_threshold = 240
green_threshold = 240
blue_threshold = 240
rgb_thresholds = [red_threshold, green_threshold, blue_threshold]

thresholds = (image[:, :, 0] < rgb_thresholds[0])\
            | (image[:, :, 1] < rgb_thresholds[1])\
            | (image[:, :, 2] < rgb_thresholds[2])

color_select[thresholds] = [0, 0, 0]
plt.imshow(color_select)
plt.show()

xsize = image.shape[1]
ysize = image.shape[0]

#set the mask region
left_bottom = [100, 300]
right_bottom = [500, 300]
apex = [300, 200]

# Fit lines Y = AX + B
# np.polyfit returns coefficients [A, B]
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# find the region
XX, YY = np.meshgrid(np.arange(xsize), np.arange(ysize))
region_thresholds = (YY > fit_left[0]*XX + fit_left[1])\
                   & (YY > fit_right[0]*XX + fit_right[1])\
                   & (YY < fit_bottom[0]*XX + fit_bottom[1])

region_select = np.copy(color_select)
region_select[region_thresholds] = [255, 0, 0]

plt.imshow(region_select)
plt.show()               




