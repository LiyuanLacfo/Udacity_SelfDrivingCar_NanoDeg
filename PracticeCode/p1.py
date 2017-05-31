import cv2
import numpy as np
def grayscale(img):
	""""Turn the rgb `img` into gray scale"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

def canny(img, low_threshold, high_threshold):
	""""
    `img` is an gray image
    `low_threshold` is the low threshold gradient
    `high_threshold` is the high threshold gradient
    return a binary image
	"""
	return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
	"""
    Apply mask to image. Only keeps the region defined by vertices of img, other region is set to black
	"""
	mask = np.zero_like(img)
	img_shape = img.shape
	if len(img_shape) > 2:
		mask_color = (255, )*len(img_shape)
	else:
		mask_color = 255
	cv2.fillPoly(mask, vertices, mask_color)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
	for line in lines:
		for x1, y1, x2, y2 in line:
		    cv2.line(img, (x1, y1), (x2, y2), thickness)

def hough_lines(img, rho, theta, threshold, minLength, maxGap):
	"""
    `img` is the output of Canny edge detection
    `rho` is the unit of distance of hough transform
    `theta` is the unit of angular of hough transform
    `minLength` is the minimum length line to be detected
    `maxGap` is the max gap for different line to be seen as a single line
	"""
	lines = cv2.HoughLinesP(img, rho, theta, threshold, minLength, maxGap)
	line_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	draw_lines(line_image, lines)
	return line_image

def weighted_img(img, initial_img, alpha, beta, gamma):
	return cv2.addWeighted(initial_img, alpha, img, beta, gamma)





	


