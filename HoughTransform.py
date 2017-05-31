import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

image = mpimg.imread("../PracticeImage/solidYellowLeft.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gaussianKsize = 3
blur_gray = cv2.GaussianBlur(gray, (gaussianKsize, gaussianKsize), 0)
# plt.imshow(blur_gray)
# plt.show()
low_threshold = 100
high_threshold = 300
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
# plt.imshow(edges, cmap="Greys_r")
# plt.show()

# we create a masked edge image
mask = np.zeros_like(edges)
ignore_mask_color = 255
imshape = image.shape
vertices = np.array([[(180, 520), (450, 300), (520, 300), (800, 500)]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)
# plt.imshow(masked_edges, cmap="Greys_r")
# plt.show()


#Hough Transform
minLength = 20
maxLineGap = 5

lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, minLength, maxLineGap)
print("Line slope: {}".format(lines[0].slope))
line_image = np.copy(image)*0
for line in lines:
	for x1, y1, x2, y2 in line:
		cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
plt.imshow(line_image)
plt.show()
# plt.imshow(edges, cmap="Greys_r")
# plt.show()
color_edges = np.dstack((edges, edges, edges))
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
plt.imshow(combo)
plt.show()

