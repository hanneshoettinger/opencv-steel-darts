__author__ = "Hannes Hoettinger"

import numpy as np
import cv2
import math

DEBUG = True


class Draw:
    def __init__(self):
        # 20 sectors...
        self.sectorangle = 2 * math.pi / 20

    ## improve and make circle radius accessible
    def drawBoard(self, img, calData):

        # draw board
        cv2.circle(img, (400, 400), calData.ring_radius[0], (255, 255, 255), 1)  # outside double
        cv2.circle(img, (400, 400), calData.ring_radius[1], (255, 255, 255), 1)  # inside double
        cv2.circle(img, (400, 400), calData.ring_radius[2], (255, 255, 255), 1)  # outside treble
        cv2.circle(img, (400, 400), calData.ring_radius[3], (255, 255, 255), 1)  # inside treble
        cv2.circle(img, (400, 400), calData.ring_radius[4], (255, 255, 255), 1)  # 25
        cv2.circle(img, (400, 400), calData.ring_radius[5], (255, 255, 255), 1)  # Bulls eye


        i = 0
        while (i < 20):
            cv2.line(img, (400, 400), (
                int(400 + calData.ring_radius[5] * math.cos((0.5 + i) * self.sectorangle)),
                int(400 + calData.ring_radius[5] * math.sin((0.5 + i) * self.sectorangle))), (255, 255, 255), 1)
            i = i + 1

        return img
