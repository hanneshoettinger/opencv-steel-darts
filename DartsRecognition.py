__author__ = "Hannes Hoettinger"

import numpy as np
import cv2
import time
import cv2.cv as cv
import math
import pickle
from Classes import *
from MathFunctions import *
from DartsMapping import *
from Draw import *

DEBUG = False

winName = "test2"


def cam2gray(cam):
    success, image = cam.read()
    img_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return success, img_g


def getThreshold(cam, t):
    success, t_plus = cam2gray(cam)
    dimg = cv2.absdiff(t, t_plus)

    blur = cv2.GaussianBlur(dimg, (5, 5), 0)
    blur = cv2.bilateralFilter(blur, 9, 75, 75)
    _, thresh = cv2.threshold(blur, 60, 255, 0)

    return thresh


def diff2blur(cam, t):
    _, t_plus = cam2gray(cam)
    dimg = cv2.absdiff(t, t_plus)

    ## kernel size important -> make accessible
    # filter noise from image distortions
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(dimg, -1, kernel)

    return t_plus, blur


def getCorners(img_in):
    # number of features to track is a distinctive feature
    ## FeaturesToTrack important -> make accessible
    edges = cv2.goodFeaturesToTrack(img_in, 640, 0.0008, 1, mask=None, blockSize=3, useHarrisDetector=1, k=0.06)  # k=0.08
    corners = np.int0(edges)

    return corners


def filterCorners(corners):
    cornerdata = []
    tt = 0
    mean_corners = np.mean(corners, axis=0)
    for i in corners:
        xl, yl = i.ravel()
        # filter noise to only get dart arrow
        ## threshold important -> make accessible
        if abs(mean_corners[0][0] - xl) > 180:
            cornerdata.append(tt)
        if abs(mean_corners[0][1] - yl) > 120:
            cornerdata.append(tt)
        tt += 1

    corners_new = np.delete(corners, [cornerdata], axis=0)  # delete corners to form new array

    return corners_new


def filterCornersLine(corners, rows, cols):
    [vx, vy, x, y] = cv2.fitLine(corners, cv.CV_DIST_HUBER, 0, 0.1, 0.1)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)

    cornerdata = []
    tt = 0
    for i in corners:
        xl, yl = i.ravel()
        # check distance to fitted line, only draw corners within certain range
        distance = dist(0, lefty, cols - 1, righty, xl, yl)
        if distance > 40:  ## threshold important -> make accessible
            cornerdata.append(tt)

        tt += 1

    corners_final = np.delete(corners, [cornerdata], axis=0)  # delete corners to form new array

    return corners_final


def getRealLocation(corners_final, mount):

    if mount == "right":
        loc = np.argmax(corners_final, axis=0)
    else:
        loc = np.argmin(corners_final, axis=0)

    locationofdart = corners_final[loc]

    # check if dart location has neighbouring corners (if not -> continue)
    cornerdata = []
    tt = 0
    for i in corners_final:
        xl, yl = i.ravel()
        distance = abs(locationofdart.item(0) - xl) + abs(locationofdart.item(1) - yl)
        if distance < 40:  ## threshold important
            tt += 1
        else:
            cornerdata.append(tt)

    if tt < 3:
        corners_temp = cornerdata
        maxloc = np.argmax(corners_temp, axis=0)
        locationofdart = corners_temp[maxloc]
        print "### used different location due to noise!"

    return locationofdart


def getDarts(cam_R, cam_L, calData_R, calData_L, playerObj, GUI):

    finalScore = 0
    count = 0
    breaker = 0
    ## threshold important -> make accessible
    minThres = 2000/2
    maxThres = 15000/2

    # save score if score is below 1...
    old_score = playerObj.score

    # Read first image twice (issue somewhere) to start loop:
    _, _ = cam2gray(cam_R)
    _, _ = cam2gray(cam_L)
    # wait for camera
    time.sleep(0.1)
    success, t_R = cam2gray(cam_R)
    _, t_L = cam2gray(cam_L)

    while success:
        # wait for camera
        time.sleep(0.1)
        # check if dart hit the board
        thresh_R = getThreshold(cam_R, t_R)
        thresh_L = getThreshold(cam_L, t_L)

        print cv2.countNonZero(thresh_R)
        ## threshold important
        if (cv2.countNonZero(thresh_R) > minThres and cv2.countNonZero(thresh_R) < maxThres) \
            or (cv2.countNonZero(thresh_L) > minThres and cv2.countNonZero(thresh_L) < maxThres):
            # wait for camera vibrations
            time.sleep(0.2)
            # filter noise
            t_plus_R, blur_R = diff2blur(cam_R, t_R)
            t_plus_L, blur_L = diff2blur(cam_L, t_L)
            # get corners
            corners_R = getCorners(blur_R)
            corners_L = getCorners(blur_L)

            testimg = blur_R.copy()

            # dart outside?
            if corners_R.size < 40 and corners_L.size < 40:
                print "### dart not detected"
                continue

            # filter corners
            corners_f_R = filterCorners(corners_R)
            corners_f_L = filterCorners(corners_L)

            # dart outside?
            if corners_f_R.size < 30 and corners_f_L.size < 30:
                print "### dart not detected"
                continue

            # find left and rightmost corners#
            rows, cols = blur_R.shape[:2]
            corners_final_R = filterCornersLine(corners_f_R, rows, cols)
            corners_final_L = filterCornersLine(corners_f_L, rows, cols)

            _, thresh_R = cv2.threshold(blur_R, 60, 255, 0)
            _, thresh_L = cv2.threshold(blur_L, 60, 255, 0)

            # check if it was really a dart
            print cv2.countNonZero(thresh_R)
            if cv2.countNonZero(thresh_R) > maxThres*2 or cv2.countNonZero(thresh_L) > maxThres*2:
                continue

            print "Dart detected"
            # dart was found -> increase counter
            breaker += 1

            dartInfo = DartDef()

            # get final darts location
            try:
                dartInfo_R = DartDef()
                dartInfo_L = DartDef()

                dartInfo_R.corners = corners_final_R.size
                dartInfo_L.corners = corners_final_L.size

                locationofdart_R = getRealLocation(corners_final_R, "right")
                locationofdart_L = getRealLocation(corners_final_L, "left")

                # check for the location of the dart with the calibration
                dartloc_R = getTransformedLocation(locationofdart_R.item(0), locationofdart_R.item(1), calData_R)
                dartloc_L = getTransformedLocation(locationofdart_L.item(0), locationofdart_L.item(1), calData_L)
                # detect region and score
                dartInfo_R = getDartRegion(dartloc_R, calData_R)
                dartInfo_L = getDartRegion(dartloc_L, calData_L)

                cv2.circle(testimg, (locationofdart_R.item(0), locationofdart_R.item(1)), 10, (255, 255, 255), 2, 8)
                cv2.circle(testimg, (locationofdart_R.item(0), locationofdart_R.item(1)), 2, (0, 255, 0), 2, 8)
            except:
                print "Something went wrong in finding the darts location!"
                breaker -= 1
                continue

            # "merge" scores
            if dartInfo_R.base == dartInfo_L.base and dartInfo_R.multiplier == dartInfo_L.multiplier:
                dartInfo = dartInfo_R
            # use the score of the image with more corners
            else:
                if dartInfo_R.corners > dartInfo_L.corners:
                    dartInfo = dartInfo_R
                else:
                    dartInfo = dartInfo_L

            print dartInfo.base, dartInfo.multiplier

            if breaker == 1:
                GUI.dart1entry.insert(10,str(dartInfo.base * dartInfo.multiplier))
                dart = int(GUI.dart1entry.get())
                cv2.imwrite("frame2.jpg", testimg)     # save dart1 frame
            elif breaker == 2:
                GUI.dart2entry.insert(10,str(dartInfo.base * dartInfo.multiplier))
                dart = int(GUI.dart2entry.get())
                cv2.imwrite("frame3.jpg", testimg)     # save dart2 frame
            elif breaker == 3:
                GUI.dart3entry.insert(10,str(dartInfo.base * dartInfo.multiplier))
                dart = int(GUI.dart3entry.get())
                cv2.imwrite("frame4.jpg", testimg)     # save dart3 frame

            playerObj.score -= dart

            if playerObj.score == 0 and dartInfo.multiplier == 2:
                playerObj.score = 0
                breaker = 3
            elif playerObj.score <= 1:
                playerObj.score = old_score
                breaker = 3

            # save new diff img for next dart
            t_R = t_plus_R
            t_L = t_plus_L

            if playerObj.player == 1:
                GUI.e1.delete(0,'end')
                GUI.e1.insert(10,playerObj.score)
            else:
                GUI.e2.delete(0,'end')
                GUI.e2.insert(10,playerObj.score)

            finalScore += (dartInfo.base * dartInfo.multiplier)

            if breaker == 3:
                break

            #cv2.imshow(winName, tnow)

        # missed dart
        elif cv2.countNonZero(thresh_R) < maxThres/2 or cv2.countNonZero(thresh_L) < maxThres/2:
            continue

        # if player enters zone - break loop
        elif cv2.countNonZero(thresh_R) > maxThres/2 or cv2.countNonZero(thresh_L) > maxThres/2:
            break

        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyWindow(winName)
            break

        count += 1

    GUI.finalentry.delete(0, 'end')
    GUI.finalentry.insert(10,finalScore)

    print finalScore


if __name__ == '__main__':
    print "Welcome to darts!"
    img = cv2.imread("D:\Projekte\PycharmProjects\DartsScorer\Darts\Dartboard_2.png")
    img2 = cv2.imread("D:\Projekte\PycharmProjects\DartsScorer\Darts\Dartboard_3.png")

    vidcap = cv2.VideoCapture("C:\Users\hanne\OneDrive\Projekte\GitHub\darts\Darts\Darts_Testvideo_9_1.mp4")
    from_video = True

# if DEBUG:
#     loc_x = dartloc[0]  # 400 + dartInfo.magnitude * math.tan(dartInfo.angle * math.pi/180)
#     loc_y = dartloc[1]  # 400 + dartInfo.magnitude * math.tan(dartInfo.angle * math.pi/180)
#     cv2.circle(debug_img, (int(loc_x), int(loc_y)), 2, (0, 255, 0), 2, 8)
#     cv2.circle(debug_img, (int(loc_x), int(loc_y)), 6, (0, 255, 0), 1, 8)
#     string = "" + str(dartInfo.base) + "x" + str(dartInfo.multiplier)
#     # add text (before clear with rectangle)
#     cv2.rectangle(debug_img, (600, 700), (800, 800), (0, 0, 0), -1)
#     cv2.putText(debug_img, string, (600, 750), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, 8)
#     cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
#     cv2.namedWindow("raw", cv2.WINDOW_NORMAL)
#     cv2.namedWindow("test", cv2.WINDOW_NORMAL)
#     cv2.imshow(winName, debug_img)
#     cv2.imshow("raw", t_plus_copy)
#     cv2.imshow("test", testimg)