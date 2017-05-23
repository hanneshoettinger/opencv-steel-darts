__author__ = "Hannes Hoettinger"

import numpy as np
import cv2
import math
import pickle
from Classes import *

DEBUG = True


def getTransformedLocation(x_coord,y_coord, calData):
    try:
            # transform only the hit point with the saved transformation matrix
            # ToDo: idea for second camera -> transform complete image and overlap both images to find dart location?
            dart_loc_temp = np.array([[x_coord, y_coord]], dtype="float32")
            dart_loc_temp = np.array([dart_loc_temp])
            dart_loc = cv2.perspectiveTransform(dart_loc_temp, calData.transformation_matrix)
            new_dart_loc = tuple(dart_loc.reshape(1, -1)[0])

            return new_dart_loc

    #system not calibrated
    except AttributeError as err1:
        print err1
        return (-1, -1)

    except NameError as err2:
        #not calibrated error
        print err2
        return (-2, -2)


#Returns dartThrow (score, multiplier, angle, magnitude) based on x,y location
def getDartRegion(dart_loc, calData):
    try:
            height = 800
            width = 800

            dartInfo = DartDef()

            #find the magnitude and angle of the dart
            vx = (dart_loc[0] - width/2)
            vy = (height/2 - dart_loc[1])

            # reference angle for atan2 conversion
            ref_angle = 81

            dartInfo.magnitude = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
            dartInfo.angle = math.fmod(((math.atan2(vy,vx) * 180/math.pi) + 360 - ref_angle), 360)

            angleDiffMul = int((dartInfo.angle) / 18.0)

            #starting from the 20 points
            if angleDiffMul == 0:
                dartInfo.base = 20
            elif angleDiffMul == 1:
                dartInfo.base = 5
            elif angleDiffMul == 2:
                dartInfo.base = 12
            elif angleDiffMul == 3:
                dartInfo.base = 9
            elif angleDiffMul == 4:
                dartInfo.base = 14
            elif angleDiffMul == 5:
                dartInfo.base = 11
            elif angleDiffMul == 6:
                dartInfo.base = 8
            elif angleDiffMul == 7:
                dartInfo.base = 16
            elif angleDiffMul == 8:
                dartInfo.base = 7
            elif angleDiffMul == 9:
                dartInfo.base = 19
            elif angleDiffMul == 10:
                dartInfo.base = 3
            elif angleDiffMul == 11:
                dartInfo.base = 17
            elif angleDiffMul == 12:
                dartInfo.base = 2
            elif angleDiffMul == 13:
                dartInfo.base = 15
            elif angleDiffMul == 14:
                dartInfo.base = 10
            elif angleDiffMul == 15:
                dartInfo.base = 6
            elif angleDiffMul == 16:
                dartInfo.base = 13
            elif angleDiffMul == 17:
                dartInfo.base = 4
            elif angleDiffMul == 18:
                dartInfo.base = 18
            elif angleDiffMul == 19:
                dartInfo.base = 1
            else:
                #something went wrong
                dartInfo.base = -300

            #Calculating multiplier (and special cases for Bull's Eye):
            for i in range(0, len(calData.ring_radius)):
                #Find the ring that encloses the dart
                if dartInfo.magnitude <= calData.ring_radius[i]:
                    #Bull's eye, adjust base score
                    if i == 0:
                        dartInfo.base = 25
                        dartInfo.multiplier = 2
                    elif i == 1:
                        dartInfo.base = 25
                        dartInfo.multiplier = 1
                    #triple ring
                    elif i == 3:
                        dartInfo.multiplier = 3
                    #double ring
                    elif i == 5:
                        dartInfo.multiplier = 2
                    #single
                    elif i == 2 or i == 4:
                        dartInfo.multiplier = 1
                    #finished calculation
                    break

            #miss
            if dartInfo.magnitude > calData.ring_radius[5]:
                dartInfo.base = 0
                dartInfo.multiplier = 0

            return dartInfo


    #system not calibrated
    except AttributeError as err1:
        print err1
        dartInfo = DartDef()
        return dartInfo

    except NameError as err2:
        #not calibrated error
        print err2
        dartInfo = DartDef()
        return dartInfo

