import cv2
import numpy as np
import matplotlib.pyplot as plt


def im2figure(image):
    "plot a cv2 image in iPython console"
    img = cv2.cvtColor(image, cv2.cv.CV_BGR2RGB)
    plt.imshow(img)