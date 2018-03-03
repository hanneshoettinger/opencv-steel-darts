__author__ = "Hannes Hoettinger"

import cv2                   #open cv2
import cv2.cv as cv          #open cv
import time
import numpy as np
from threading import Thread
from threading import Event
import sys
import math
import pickle
import os.path
from im2figure import *
from numpy.linalg import inv
from MathFunctions import *
from Classes import *
from Draw import *
from VideoCapture import VideoStream


DEBUG = False

ring_arr = []
winName3 = "hsv image colors?"
winName4 = "Calibration?"
winName5 = "Choose Ring"


def nothing(x):
    pass


def destinationPoint(i, calData):
    dstpoint = [(calData.center_dartboard[0] + calData.ring_radius[5] * math.cos((0.5 + i) * calData.sectorangle)),
                (calData.center_dartboard[1] + calData.ring_radius[5] * math.sin((0.5 + i) * calData.sectorangle))]

    return dstpoint


def transformation(imCalRGB, calData, tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4):

    points = calData.points

    ## sectors are sometimes different -> make accessible
    # used when line rectangle intersection at specific segment is used for transformation:
    newtop = destinationPoint(calData.dstpoints[0], calData)
    newbottom = destinationPoint(calData.dstpoints[1], calData)
    newleft = destinationPoint(calData.dstpoints[2], calData)
    newright = destinationPoint(calData.dstpoints[3], calData)

    # get a fresh new image
    new_image = imCalRGB.copy()

    # create transformation matrix
    src = np.array([(points[0][0]+tx1, points[0][1]+ty1), (points[1][0]+tx2, points[1][1]+ty2),
                    (points[2][0]+tx3, points[2][1]+ty3), (points[3][0]+tx4, points[3][1]+ty4)], np.float32)
    dst = np.array([newtop, newbottom, newleft, newright], np.float32)
    transformation_matrix = cv2.getPerspectiveTransform(src, dst)

    new_image = cv2.warpPerspective(new_image, transformation_matrix, (800, 800))

    # draw image
    drawBoard = Draw()
    new_image = drawBoard.drawBoard(new_image, calData)

    cv2.circle(new_image, (int(newtop[0]), int(newtop[1])), 2, cv.CV_RGB(255, 255, 0), 2, 4)
    cv2.circle(new_image, (int(newbottom[0]), int(newbottom[1])), 2, cv.CV_RGB(255, 255, 0), 2, 4)
    cv2.circle(new_image, (int(newleft[0]), int(newleft[1])), 2, cv.CV_RGB(255, 255, 0), 2, 4)
    cv2.circle(new_image, (int(newright[0]), int(newright[1])), 2, cv.CV_RGB(255, 255, 0), 2, 4)

    cv2.imshow('manipulation', new_image)

    return transformation_matrix


def manipulateTransformationPoints(imCal, calData):

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('tx1', 'image', 0, 20, nothing)
    cv2.createTrackbar('ty1', 'image', 0, 20, nothing)
    cv2.createTrackbar('tx2', 'image', 0, 20, nothing)
    cv2.createTrackbar('ty2', 'image', 0, 20, nothing)
    cv2.createTrackbar('tx3', 'image', 0, 20, nothing)
    cv2.createTrackbar('ty3', 'image', 0, 20, nothing)
    cv2.createTrackbar('tx4', 'image', 0, 20, nothing)
    cv2.createTrackbar('ty4', 'image', 0, 20, nothing)
    cv2.setTrackbarPos('tx1', 'image', 10)
    cv2.setTrackbarPos('ty1', 'image', 10)
    cv2.setTrackbarPos('tx2', 'image', 10)
    cv2.setTrackbarPos('ty2', 'image', 10)
    cv2.setTrackbarPos('tx3', 'image', 10)
    cv2.setTrackbarPos('ty3', 'image', 10)
    cv2.setTrackbarPos('tx4', 'image', 10)
    cv2.setTrackbarPos('ty4', 'image', 10)
    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)
    imCal_copy = imCal.copy()
    while (1):
        cv2.imshow('image', imCal_copy)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # get current positions of four trackbars
        tx1 = cv2.getTrackbarPos('tx1', 'image') - 10
        ty1 = cv2.getTrackbarPos('ty1', 'image') - 10
        tx2 = cv2.getTrackbarPos('tx2', 'image') - 10
        ty2 = cv2.getTrackbarPos('ty2', 'image') - 10
        tx3 = cv2.getTrackbarPos('tx3', 'image') - 10
        ty3 = cv2.getTrackbarPos('ty3', 'image') - 10
        tx4 = cv2.getTrackbarPos('tx4', 'image') - 10
        ty4 = cv2.getTrackbarPos('ty4', 'image') - 10
        s = cv2.getTrackbarPos(switch, 'image')
        if s == 0:
            imCal_copy[:] = 0
        else:
            # transform the image to form a perfect circle
            transformation_matrix = transformation(imCal, calData, tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4)

    return transformation_matrix


def autocanny(imCal):
    # apply automatic Canny edge detection using the computed median
    sigma = 0.33
    v = np.median(imCal)
    #lower = int(max(0, (1.0 - sigma) * v))
    #upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(imCal, 250, 255)

    return edged


def findEllipse(thresh2, image_proc_img):

    Ellipse = EllipseDef()

    contours, hierarchy = cv2.findContours(thresh2, 1, 2)

    minThresE = 200000/4
    maxThresE = 1000000/4

    ## contourArea threshold important -> make accessible
    for cnt in contours:
        try:  # threshold critical, change on demand?
            if minThresE < cv2.contourArea(cnt) < maxThresE:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(image_proc_img, ellipse, (0, 255, 0), 2)

                x, y = ellipse[0]
                a, b = ellipse[1]
                angle = ellipse[2]

                center_ellipse = (x, y)

                a = a / 2
                b = b / 2

                cv2.ellipse(image_proc_img, (int(x), int(y)), (int(a), int(b)), int(angle), 0.0, 360.0,
                            cv.CV_RGB(255, 0, 0))
        # corrupted file
        except:
            print "error"
            return Ellipse, image_proc_img

    Ellipse.a = a
    Ellipse.b = b
    Ellipse.x = x
    Ellipse.y = y
    Ellipse.angle = angle
    return Ellipse, image_proc_img


def findSectorLines(edged, image_proc_img, angleZone1, angleZone2):
    p = []
    intersectp = []
    lines_seg = []
    counter = 0

    # fit line to find intersec point for dartboard center point
    lines = cv2.HoughLines(edged, 1, np.pi / 80, 100, 100)

    ## sector angles important -> make accessible
    for rho, theta in lines[0]:
        # split between horizontal and vertical lines (take only lines in certain range)
        if theta > np.pi / 180 * angleZone1[0] and theta < np.pi / 180 * angleZone1[1]:

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))

            for rho1, theta1 in lines[0]:

                if theta1 > np.pi / 180 * angleZone2[0] and theta1 < np.pi / 180 * angleZone2[1]:

                    a = np.cos(theta1)
                    b = np.sin(theta1)
                    x0 = a * rho1
                    y0 = b * rho1
                    x3 = int(x0 + 2000 * (-b))
                    y3 = int(y0 + 2000 * (a))
                    x4 = int(x0 - 2000 * (-b))
                    y4 = int(y0 - 2000 * (a))

                    if y1 == y2 and y3 == y4:  # Horizontal Lines
                        diff = abs(y1 - y3)
                    elif x1 == x2 and x3 == x4:  # Vertical Lines
                        diff = abs(x1 - x3)
                    else:
                        diff = 0

                    if diff < 200 and diff is not 0:
                        continue

                    cv2.line(image_proc_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.line(image_proc_img, (x3, y3), (x4, y4), (255, 0, 0), 1)

                    p.append((x1, y1))
                    p.append((x2, y2))
                    p.append((x3, y3))
                    p.append((x4, y4))

                    intersectpx, intersectpy = intersectLines(p[counter], p[counter + 1], p[counter + 2],
                                                              p[counter + 3])

                    # consider only intersection close to the center of the image
                    if intersectpx < 200 or intersectpx > 900 or intersectpy < 200 or intersectpy > 900:
                        continue

                    intersectp.append((intersectpx, intersectpy))

                    lines_seg.append([(x1, y1), (x2, y2)])
                    lines_seg.append([(x3, y3), (x4, y4)])

                    cv2.line(image_proc_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.line(image_proc_img, (x3, y3), (x4, y4), (255, 0, 0), 1)

                    # point offset
                    counter = counter + 4

    return lines_seg, image_proc_img


def ellipse2circle(Ellipse):
    angle = (Ellipse.angle) * math.pi / 180
    x = Ellipse.x
    y = Ellipse.y
    a = Ellipse.a
    b = Ellipse.b

    # build transformation matrix http://math.stackexchange.com/questions/619037/circle-affine-transformation
    R1 = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    R2 = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])

    T1 = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
    T2 = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

    D = np.array([[1, 0, 0], [0, a / b, 0], [0, 0, 1]])

    M = T2.dot(R2.dot(D.dot(R1.dot(T1))))

    return M


def getEllipseLineIntersection(Ellipse, M, lines_seg):
    center_ellipse = (Ellipse.x, Ellipse.y)
    circle_radius = Ellipse.a
    M_inv = np.linalg.inv(M)

    # find line circle intersection and use inverse transformation matrix to transform it back to the ellipse
    intersectp_s = []
    for lin in lines_seg:
        line_p1 = M.dot(np.transpose(np.hstack([lin[0], 1])))
        line_p2 = M.dot(np.transpose(np.hstack([lin[1], 1])))
        inter1, inter_p1, inter2, inter_p2 = intersectLineCircle(np.asarray(center_ellipse), circle_radius,
                                                                 np.asarray(line_p1), np.asarray(line_p2))
        if inter1:
            inter_p1 = M_inv.dot(np.transpose(np.hstack([inter_p1, 1])))
            if inter2:
                inter_p2 = M_inv.dot(np.transpose(np.hstack([inter_p2, 1])))
                intersectp_s.append(inter_p1)
                intersectp_s.append(inter_p2)

    return intersectp_s


def getTransformationPoints(image_proc_img, mount):

    imCalHSV = cv2.cvtColor(image_proc_img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(imCalHSV, -1, kernel)
    h, s, imCal = cv2.split(blur)

    ## threshold important -> make accessible
    #ret, thresh = cv2.threshold(imCal, 140, 255, cv2.THRESH_BINARY_INV)
    ret, thresh = cv2.threshold(imCal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ## kernel size important -> make accessible
    # very important -> removes lines outside the outer ellipse -> find ellipse
    kernel = np.ones((5, 5), np.uint8)
    thresh2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("thresh2", thresh2)

    # find enclosing ellipse
    Ellipse, image_proc_img = findEllipse(thresh2, image_proc_img)

    # return the edged image
    edged = autocanny(thresh2)  # imCal
    cv2.imshow("test", edged)

    # find 2 sector lines -> horizontal and vertical sector line -> make angles accessible? with slider?
    if mount == "right":
        angleZone1 = (Ellipse.angle - 5, Ellipse.angle + 5)
        angleZone2 = (Ellipse.angle - 100, Ellipse.angle - 80)
        lines_seg, image_proc_img = findSectorLines(edged, image_proc_img, angleZone1, angleZone2)
    else:
        lines_seg, image_proc_img = findSectorLines(edged, image_proc_img, angleZone1=(80, 120), angleZone2=(30, 40))

    cv2.imshow("test4", image_proc_img)

    # ellipse 2 circle transformation to find intersection points -> source points for transformation
    M = ellipse2circle(Ellipse)
    intersectp_s = getEllipseLineIntersection(Ellipse, M, lines_seg)

    source_points = []

    try:
        new_intersect = np.mean(([intersectp_s[0],intersectp_s[4]]), axis=0, dtype=np.float32)
        source_points.append(new_intersect) # top
        new_intersect = np.mean(([intersectp_s[1], intersectp_s[5]]), axis=0, dtype=np.float32)
        source_points.append(new_intersect) # bottom
        new_intersect = np.mean(([intersectp_s[2], intersectp_s[6]]), axis=0, dtype=np.float32)
        source_points.append(new_intersect) # left
        new_intersect = np.mean(([intersectp_s[3], intersectp_s[7]]), axis=0, dtype=np.float32)
        source_points.append(new_intersect) # right
    except:
        pointarray = np.array(intersectp_s)
        top_idx = [np.argmin(pointarray[:, 1])][0]
        bot_idx = [np.argmax(pointarray[:, 1])][0]
        if mount == "right":
            left_idx = [np.argmin(pointarray[:, 0])][0]
            right_idx = [np.argmax(pointarray[:, 0])][0]
        else:
            left_idx = [np.argmax(pointarray[:, 0])][0]
            right_idx = [np.argmin(pointarray[:, 0])][0]
        source_points.append(intersectp_s[top_idx])  # top
        source_points.append(intersectp_s[bot_idx])  # bottom
        source_points.append(intersectp_s[left_idx])  # left
        source_points.append(intersectp_s[right_idx])  # right

    cv2.circle(image_proc_img, (int(source_points[0][0]), int(source_points[0][1])), 3, cv.CV_RGB(255, 0, 0), 2, 8)
    cv2.circle(image_proc_img, (int(source_points[1][0]), int(source_points[1][1])), 3, cv.CV_RGB(255, 0, 0), 2, 8)
    cv2.circle(image_proc_img, (int(source_points[2][0]), int(source_points[2][1])), 3, cv.CV_RGB(255, 0, 0), 2, 8)
    cv2.circle(image_proc_img, (int(source_points[3][0]), int(source_points[3][1])), 3, cv.CV_RGB(255, 0, 0), 2, 8)

    winName2 = "th circles?"
    cv2.namedWindow(winName2, cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow(winName2, image_proc_img)

    end = cv2.waitKey(0)
    if end == 13:
        cv2.destroyAllWindows()
        return source_points


def calibrate(cam_R, cam_L):

    try:
        success, imCalRGB_R = cam_R.read()
        _, imCalRGB_L = cam_L.read()

    except:
        print "Could not init cams"
        return

    imCal_R = imCalRGB_R.copy()
    imCal_L = imCalRGB_L.copy()

    imCalRGBorig = imCalRGB_R.copy()

    cv2.imwrite("frame1_R.jpg", imCalRGB_R)     # save calibration frame
    cv2.imwrite("frame1_L.jpg", imCalRGB_L)  # save calibration frame

    global calibrationComplete
    calibrationComplete = False

    while calibrationComplete == False:
        #Read calibration file, if exists
        if os.path.isfile("calibrationData_R.pkl"):
            try:
                calFile = open('calibrationData_R.pkl', 'rb')
                calData_R = CalibrationData()
                calData_R = pickle.load(calFile)
                calFile.close()

                calFile = open('calibrationData_L.pkl', 'rb')
                calData_L = CalibrationData()
                calData_L = pickle.load(calFile)
                calFile.close()

                #copy image for old calibration data
                transformed_img_R = imCalRGB_R.copy()
                transformed_img_L = imCalRGB_L.copy()

                transformed_img_R = cv2.warpPerspective(imCalRGB_R, calData_R.transformation_matrix, (800, 800))
                transformed_img_L = cv2.warpPerspective(imCalRGB_L, calData_L.transformation_matrix, (800, 800))

                draw_R = Draw()
                draw_L = Draw()

                transformed_img_R = draw_R.drawBoard(transformed_img_R, calData_R)
                transformed_img_L = draw_L.drawBoard(transformed_img_L, calData_L)

                cv2.imshow("Right Cam", transformed_img_R)
                cv2.imshow("Left Cam", transformed_img_L)

                test = cv2.waitKey(0)
                if test == 13:
                    cv2.destroyAllWindows()
                    #we are good with the previous calibration data
                    calibrationComplete = True
                    return calData_R, calData_L
                else:
                    cv2.destroyAllWindows()
                    calibrationComplete = True
                    #delete the calibration file and start over
                    os.remove("calibrationData_R.pkl")
                    os.remove("calibrationData_L.pkl")
                    #restart calibration
                    calibrate(cam_R, cam_L)

            #corrupted file
            except EOFError as err:
                print err

        # start calibration if no calibration data exists
        else:

            calData_R = CalibrationData()
            calData_L = CalibrationData()

            imCal_R = imCalRGB_R.copy()
            imCal_L = imCalRGB_L.copy()

            calData_R.points = getTransformationPoints(imCal_R, "right")
            # 13/6: 0 | 6/10: 1 | 10/15: 2 | 15/2: 3 | 2/17: 4 | 17/3: 5 | 3/19: 6 | 19/7: 7 | 7/16: 8 | 16/8: 9 |
            # 8/11: 10 | 11/14: 11 | 14/9: 12 | 9/12: 13 | 12/5: 14 | 5/20: 15 | 20/1: 16 | 1/18: 17 | 18/4: 18 | 4/13: 19
            # top, bottom, left, right
            # 12/9, 2/15, 8/16, 13/4
            calData_R.dstpoints = [12, 2, 8, 18]
            calData_R.transformation_matrix = manipulateTransformationPoints(imCal_R, calData_R)

            calData_L.points = getTransformationPoints(imCal_L, "left")
            # 12/9, 2/15, 8/16, 13/4
            calData_L.dstpoints = [12, 2, 8, 18]
            calData_L.transformation_matrix = manipulateTransformationPoints(imCal_L, calData_L)

            cv2.destroyAllWindows()

            print "The dartboard image has now been normalized."
            print ""

            cv2.imshow(winName4, imCal_R)
            test = cv2.waitKey(0)
            if test == 13:
                cv2.destroyWindow(winName4)
                cv2.destroyAllWindows()

            #write the calibration data to a file
            calFile = open("calibrationData_R.pkl", "wb")
            pickle.dump(calData_R, calFile, 0)
            calFile.close()

            calFile = open("calibrationData_L.pkl", "wb")
            pickle.dump(calData_L, calFile, 0)
            calFile.close()

            calibrationComplete = True

            return calData_R, calData_L


    cv2.destroyAllWindows()


if __name__ == '__main__':
    print "Welcome to darts!"

    cam_R = VideoStream(src=2).start()
    cam_L = VideoStream(src=3).start()

    calibrate(cam_R, cam_L)
