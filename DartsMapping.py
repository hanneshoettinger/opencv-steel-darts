__author__ = "Hannes Hoettinger"

import numpy as np
import cv2
# import math
# import pickle
from Classes import *

DEBUG = True


def get_transformed_location(x_coord, y_coord, cal_data):
    try:
        # transform only the hit point with the saved transformation matrix
        # ToDo: idea for second camera -> transform complete image and overlap both images to find dart location?
        dart_loc_temp = np.array([[x_coord, y_coord]], dtype="float32")
        dart_loc_temp = np.array([dart_loc_temp])
        dart_loc = cv2.perspectiveTransform(dart_loc_temp, cal_data.transformation_matrix)
        new_dart_loc = tuple(dart_loc.reshape(1, -1)[0])

        return new_dart_loc

    # system not calibrated
    except AttributeError as err1:
        print(err1)
        return -1, -1

    except NameError as err2:
        # not calibrated error
        print(err2)
        return -2, -2


# Returns dartThrow (score, multiplier, angle, magnitude) based on x,y location
def get_dart_region(dart_loc, cal_data):
    try:
        height = 800
        width = 800

        dart_info = DartDef()

        # find the magnitude and angle of the dart
        vx = (dart_loc[0] - width / 2)
        vy = (height / 2 - dart_loc[1])

        # reference angle for atan2 conversion
        ref_angle = 81

        dart_info.magnitude = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
        dart_info.angle = math.fmod(((math.atan2(vy, vx) * 180 / math.pi) + 360 - ref_angle), 360)

        angle_diff_mul = int(dart_info.angle / 18.0)

        # starting from the 20 points
        if angle_diff_mul == 0:
            dart_info.base = 20
        elif angle_diff_mul == 1:
            dart_info.base = 5
        elif angle_diff_mul == 2:
            dart_info.base = 12
        elif angle_diff_mul == 3:
            dart_info.base = 9
        elif angle_diff_mul == 4:
            dart_info.base = 14
        elif angle_diff_mul == 5:
            dart_info.base = 11
        elif angle_diff_mul == 6:
            dart_info.base = 8
        elif angle_diff_mul == 7:
            dart_info.base = 16
        elif angle_diff_mul == 8:
            dart_info.base = 7
        elif angle_diff_mul == 9:
            dart_info.base = 19
        elif angle_diff_mul == 10:
            dart_info.base = 3
        elif angle_diff_mul == 11:
            dart_info.base = 17
        elif angle_diff_mul == 12:
            dart_info.base = 2
        elif angle_diff_mul == 13:
            dart_info.base = 15
        elif angle_diff_mul == 14:
            dart_info.base = 10
        elif angle_diff_mul == 15:
            dart_info.base = 6
        elif angle_diff_mul == 16:
            dart_info.base = 13
        elif angle_diff_mul == 17:
            dart_info.base = 4
        elif angle_diff_mul == 18:
            dart_info.base = 18
        elif angle_diff_mul == 19:
            dart_info.base = 1
        else:
            # something went wrong
            dart_info.base = -300

        # Calculating multiplier (and special cases for Bull's Eye):
        for i in range(0, len(cal_data.ring_radius)):
            # Find the ring that encloses the dart
            if dart_info.magnitude <= cal_data.ring_radius[i]:
                # Bull's eye, adjust base score
                if i == 0:
                    dart_info.base = 25
                    dart_info.multiplier = 2
                elif i == 1:
                    dart_info.base = 25
                    dart_info.multiplier = 1
                # triple ring
                elif i == 3:
                    dart_info.multiplier = 3
                # double ring
                elif i == 5:
                    dart_info.multiplier = 2
                # single
                elif i == 2 or i == 4:
                    dart_info.multiplier = 1
                # finished calculation
                break

        # miss
        if dart_info.magnitude > cal_data.ring_radius[5]:
            dart_info.base = 0
            dart_info.multiplier = 0

        return dart_info

    # system not calibrated
    except AttributeError as err1:
        print(err1)
        dart_info = DartDef()
        return dart_info

    except NameError as err2:
        # not calibrated error
        print(err2)
        dart_info = DartDef()
        return dart_info
