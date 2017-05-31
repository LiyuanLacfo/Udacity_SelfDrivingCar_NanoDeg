import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
image = mpimg.imread("highway1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# plt.imshow(gray, cmap="gray")
# plt.show()
ksize = 3
blur_gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

low_threshold = 200
high_threshold = 500
# edges1 = cv2.Canny(gray, low_threshold, high_threshold)
edges = cv2.Canny(blur_gray, low_threshold, high_threshold, apertureSize=3)
plt.imshow(edges, cmap="Greys_r")
plt.show()
# plt.imshow(edges, cmap='Greys_r')
# plt.imshow(blur_gray, cmap="gray")
# plt.imshow(image)
# plt.show()
# lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
# for rho, theta in lines[2]:
# 	a = np.cos(theta)
# 	b = np.sin(theta)
# 	x0 = rho*a
# 	y0 = rho*b
# 	x1 = int(x0 + 10*(-b))
# 	y1 = int(y0 + 10 * a)
# 	x2 = int(x0 - 10 * (-b))
# 	y2 = int(y0 - 10 * a)
# 	cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 2)
line_image = np.copy(image)*0
minLength = 100
maxGap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, minLength, maxGap)
for line in lines:
    for x1, y1, x2, y2 in line:
	    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
color_edges = np.dstack((edges, edges, edges))
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
plt.imshow(combo)
plt.show()
# print(lines.shape)
# print(lines[0])

