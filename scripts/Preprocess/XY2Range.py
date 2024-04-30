import numpy as np
import cv2
from scipy.interpolate import griddata

img_path = r"/home/ychen921/808E/project2/Data/Set3/Set3/Oculus3/Oculus3-6.jpg"
img = cv2.imread(img_path)

X_RES = 267
Y_RES = 1400

R = np.arange(0.0025, 3.5025, 0.0025)
Az = np.arange(-30, 30, 0.225)

Az_rad = np.radians(Az)

xsc = R[:, np.newaxis] * np.sin(Az_rad)
ysc = R[:, np.newaxis] * np.cos(Az_rad)

print(xsc.shape)

x_grid = np.linspace(np.min(xsc), np.max(xsc), X_RES)
y_grid = np.linspace(np.min(ysc), np.max(ysc), Y_RES)
xx, yy = np.meshgrid(x_grid, y_grid)

points = np.column_stack((xsc.ravel(), ysc.ravel()))

interpolated_image = griddata(points, img.ravel(), (xx, yy), method='linear')

interpolated_image = np.reshape(interpolated_image, (Y_RES, X_RES))