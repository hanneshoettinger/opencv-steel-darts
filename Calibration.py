import math
import os.path
import pickle
import sys
import time
from threading import Event, Thread

import cv2  # open cv2
import numpy as np

from Classes import CalibrationData, EllipseDef
from Draw import Draw
from MathFunctions import intersectLineCircle, intersectLines
from utils.ReferenceImages import loadReferenceImages
from utils.ManipulateImages import *

DEBUG = False

ring_arr = []
winName3 = "hsv image colors?"
winName4 = "Calibration?"
winName5 = "Choose Ring"


def _noop(x):
    pass


def destinationPoint(i, calData):
    dstpoint_cos = math.cos((0.5 + i) * calData.sectorangle)
    dstpoint_sin = math.sin((0.5 + i) * calData.sectorangle)
    dstpoint_0 = calData.center_dartboard[0] + calData.ring_radius[5]
    dstpoint_1 = calData.center_dartboard[1] + calData.ring_radius[5]

    dstpoint = [(dstpoint_0 * dstpoint_cos),
                (dstpoint_1 * dstpoint_sin)]

    return dstpoint


def transformation(img, calData, tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4):

    points = calData.points

    # sectors are sometimes different -> make accessible
    # used when line rectangle intersection at specific
    # segment is used for transformation:
    newtop = destinationPoint(calData.dstpoints[0], calData)
    newbottom = destinationPoint(calData.dstpoints[1], calData)
    newleft = destinationPoint(calData.dstpoints[2], calData)
    newright = destinationPoint(calData.dstpoints[3], calData)

    im_copy = img.copy()

    # create transformation matrix
    src = np.array([(points[0][0]+tx1, points[0][1]+ty1),
                    (points[1][0]+tx2, points[1][1]+ty2),
                    (points[2][0]+tx3, points[2][1]+ty3),
                    (points[3][0]+tx4, points[3][1]+ty4)], np.float32)
    dst = np.array([newtop, newbottom, newleft, newright], np.float32)
    print(src)
    transformation_matrix = cv2.getPerspectiveTransform(src, dst)

    im_copy = cv2.warpPerspective(im_copy, transformation_matrix, (800, 800))

    # draw image
    drawBoard = Draw()
    im_copy = drawBoard.drawBoard(im_copy, calData)

    cv2.circle(im_copy, (int(newtop[0]), int(
        newtop[1])), 2, (255, 255, 0), 2, 4)
    cv2.circle(im_copy, (int(newbottom[0]), int(
        newbottom[1])), 2, (255, 255, 0), 2, 4)
    cv2.circle(im_copy, (int(newleft[0]), int(
        newleft[1])), 2, (255, 255, 0), 2, 4)
    cv2.circle(im_copy, (int(newright[0]), int(
        newright[1])), 2, (255, 255, 0), 2, 4)

    win_name = 'Warped'
    cv2.namedWindow(win_name)
    cv2.imshow(win_name, im_copy)

    while (1):
        kill = cv2.waitKey(1) & 0xFF
        if kill == 27 or kill == 13:
            break

    cv2.destroyWindow(win_name)
    return transformation_matrix


def manipulateTransformationPoints(imCal, calData):
    win_name = 'calibration'

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    cv2.createTrackbar('tx1', win_name, 10, 20, _noop)
    cv2.createTrackbar('ty1', win_name, 10, 20, _noop)
    cv2.createTrackbar('tx2', win_name, 10, 20, _noop)
    cv2.createTrackbar('ty2', win_name, 10, 20, _noop)
    cv2.createTrackbar('tx3', win_name, 10, 20, _noop)
    cv2.createTrackbar('ty3', win_name, 10, 20, _noop)
    cv2.createTrackbar('tx4', win_name, 10, 20, _noop)
    cv2.createTrackbar('ty4', win_name, 10, 20, _noop)
    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, win_name, 0, 1, _noop)
    imCal_copy = imCal.copy()
    while (1):
        cv2.imshow(win_name, imCal_copy)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # get current positions of four trackbars
        tx1 = cv2.getTrackbarPos('tx1', win_name) - 10
        ty1 = cv2.getTrackbarPos('ty1', win_name) - 10
        tx2 = cv2.getTrackbarPos('tx2', win_name) - 10
        ty2 = cv2.getTrackbarPos('ty2', win_name) - 10
        tx3 = cv2.getTrackbarPos('tx3', win_name) - 10
        ty3 = cv2.getTrackbarPos('ty3', win_name) - 10
        tx4 = cv2.getTrackbarPos('tx4', win_name) - 10
        ty4 = cv2.getTrackbarPos('ty4', win_name) - 10
        s = cv2.getTrackbarPos(switch, win_name)
        if s == 0:
            imCal_copy[:] = 0
        else:
            # transform the image to form a perfect circle
            transformation_matrix = transformation(
                imCal, calData, tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4)

    return transformation_matrix


def findEllipse(thresh2, img):

    Ellipse = EllipseDef()

    contours, _ = calibrateContours(thresh2, img)
    ellipse = calibrateEllipse(contours, img)

    if ellipse is not None:
        x, y = ellipse[0]
        a, b = ellipse[1]
        angle = ellipse[2]

        a = a / 2
        b = b / 2

        Ellipse.a = a
        Ellipse.b = b
        Ellipse.x = x
        Ellipse.y = y
        Ellipse.angle = angle

    return Ellipse


def _get_line(theta, rho, width):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + width * (-b))
    y1 = int(y0 + width * (a))
    x2 = int(x0 - width * (-b))
    y2 = int(y0 - width * (a))

    return (x1, y1), (x2, y2)


def findSectorLines(edged, img):
    win_name = 'findSectorLines'
    angle_min = 0
    angle_max = 180
    angle_1_min = 0
    angle_1_max = 180
    angle_2_min = 0
    angle_2_max = 180
    line_width = max(img.shape)
    lines_seg = []

    # fit line to find intersec point for dartboard center point
    lines = calibrateHoughLines(edged, img)

    cv2.namedWindow(win_name)
    cv2.createTrackbar('1. angle1_MIN', win_name, angle_min, angle_max, _noop)
    cv2.createTrackbar('2. angle1_MAX', win_name, angle_min, angle_max, _noop)
    cv2.createTrackbar('3. angle2_MIN', win_name, angle_min, angle_max, _noop)
    cv2.createTrackbar('4. angle2_MAX', win_name, angle_min, angle_max, _noop)

    while (1):
        kill = cv2.waitKey(1) & 0xFF
        if kill == 27 or kill == 13:
            break
        preview_copy = img.copy()
        angle_1_min = cv2.getTrackbarPos('1. angle1_MIN', win_name)
        angle_1_max = cv2.getTrackbarPos('2. angle1_MAX', win_name)
        angle_2_min = cv2.getTrackbarPos('3. angle2_MIN', win_name)
        angle_2_max = cv2.getTrackbarPos('4. angle2_MAX', win_name)

        theta_1 = np.pi / 180 * angle_1_min
        theta_2 = np.pi / 180 * angle_1_max
        theta_3 = np.pi / 180 * angle_2_min
        theta_4 = np.pi / 180 * angle_2_max

        for line in lines:
            for rho, theta in line:
                if theta > theta_1 and theta < theta_2:
                    line_1_start, line_1_end = _get_line(
                        theta, rho, line_width)

                    cv2.line(preview_copy, line_1_start,
                             line_1_end, (255, 0, 0), 1)

                if theta > theta_3 and theta < theta_4:
                    line_2_start, line_2_end = _get_line(
                        theta, rho, line_width)

                    cv2.line(preview_copy, line_2_start,
                             line_2_end, (255, 0, 0), 1)

                    x1, y1 = line_1_start
                    x2, y2 = line_1_end
                    x3, y3 = line_2_start
                    x4, y4 = line_2_end

                    if y1 == y2 and y3 == y4:  # Horizontal Lines
                        diff = abs(y1 - y3)
                    elif x1 == x2 and x3 == x4:  # Vertical Lines
                        diff = abs(x1 - x3)
                    else:
                        diff = 0

                    if diff < 200 and diff is not 0:
                        continue

                    intersectpx, intersectpy = intersectLines(line_1_start,
                                                              line_1_end,
                                                              line_2_start,
                                                              line_2_end)

                    # consider only intersection close to
                    # the center of the image
                    px_out_of_range = intersectpx < 200 or intersectpx > 900
                    py_out_of_range = intersectpy < 200 or intersectpy > 900

                    if px_out_of_range or py_out_of_range:
                        continue

                    lines_seg[0] = [line_1_start, line_1_end]
                    lines_seg[1] = [line_2_start, line_2_end]

        cv2.imshow(win_name, preview_copy)

    cv2.destroyWindow(win_name)
    return lines_seg


def ellipse2circle(Ellipse):
    angle = (Ellipse.angle) * math.pi / 180
    x = Ellipse.x
    y = Ellipse.y
    a = Ellipse.a
    b = Ellipse.b

    '''build transformation matrix
    http://math.stackexchange.com/questions/619037/circle-affine-transformation
    '''
    R1 = np.array([[math.cos(angle), math.sin(angle), 0],
                   [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    R2 = np.array([[math.cos(angle), -math.sin(angle), 0],
                   [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])

    T1 = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
    T2 = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

    D = np.array([[1, 0, 0], [0, a / b, 0], [0, 0, 1]])

    M = T2.dot(R2.dot(D.dot(R1.dot(T1))))

    return M


def getEllipseLineIntersection(Ellipse, M, lines_seg):
    center_ellipse = (Ellipse.x, Ellipse.y)
    circle_radius = Ellipse.a
    M_inv = np.linalg.inv(M)

    # find line circle intersection and use inverse transformation
    # matrix to transform it back to the ellipse
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


def getTransformationPoints(img, mount):

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    blurred = cv2.medianBlur(img_gray, 5)
    thresh = calibrateBinaryThresh(blurred)
    thresh2 = calibrateMorph(thresh)
    edged = calibrateCanny(thresh2)

    Ellipse = findEllipse(thresh2, img)

    '''
    find 2 sector lines ->
    horizontal and vertical sector line ->
    make angles accessible? with slider?
    '''
    lines_seg = findSectorLines(edged, img)

    '''
    ellipse 2 circle transformation to find intersection points ->
    source points for transformation
    '''
    M = ellipse2circle(Ellipse)
    intersectp_s = getEllipseLineIntersection(Ellipse, M, lines_seg)

    source_points = []

    try:
        new_intersect = np.mean(
            ([intersectp_s[0], intersectp_s[4]]), axis=0, dtype=np.float32)
        source_points.append(new_intersect)  # top
        new_intersect = np.mean(
            ([intersectp_s[1], intersectp_s[5]]), axis=0, dtype=np.float32)
        source_points.append(new_intersect)  # bottom
        new_intersect = np.mean(
            ([intersectp_s[2], intersectp_s[6]]), axis=0, dtype=np.float32)
        source_points.append(new_intersect)  # left
        new_intersect = np.mean(
            ([intersectp_s[3], intersectp_s[7]]), axis=0, dtype=np.float32)
        source_points.append(new_intersect)  # right
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

    circles_img = img.copy()
    cv2.circle(circles_img, (int(source_points[0][0]), int(
        source_points[0][1])), 3, (255, 0, 0), 2, 8)
    cv2.circle(circles_img, (int(source_points[1][0]), int(
        source_points[1][1])), 3, (255, 0, 0), 2, 8)
    cv2.circle(circles_img, (int(source_points[2][0]), int(
        source_points[2][1])), 3, (255, 0, 0), 2, 8)
    cv2.circle(circles_img, (int(source_points[3][0]), int(
        source_points[3][1])), 3, (255, 0, 0), 2, 8)

    cv2.namedWindow('Circles')
    cv2.imshow('Circles', circles_img)

    while (1):
        kill = cv2.waitKey(1) & 0xFF
        if kill == 27 or kill == 13:
            break

    cv2.destroyWindow('Circles')
    return source_points


def calibrate(img1, img2):

    imCal_R = img1.copy()
    imCal_L = img2.copy()

    global calibrationComplete
    calibrationComplete = False

    while calibrationComplete is False:
        # Read calibration file, if exists
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

                # copy image for old calibration data
                transformed_img_R = img1.copy()
                transformed_img_L = img2.copy()

                transformed_img_R = cv2.warpPerspective(
                    img1, calData_R.transformation_matrix, (800, 800))
                transformed_img_L = cv2.warpPerspective(
                    img2, calData_L.transformation_matrix, (800, 800))

                draw_R = Draw()
                draw_L = Draw()

                transformed_img_R = draw_R.drawBoard(
                    transformed_img_R, calData_R)
                transformed_img_L = draw_L.drawBoard(
                    transformed_img_L, calData_L)

                cv2.imshow("Right Cam", transformed_img_R)
                cv2.imshow("Left Cam", transformed_img_L)

                test = cv2.waitKey(0)
                if test == 13:
                    cv2.destroyAllWindows()
                    # we are good with the previous calibration data
                    calibrationComplete = True
                    return calData_R, calData_L
                else:
                    cv2.destroyAllWindows()
                    calibrationComplete = True
                    # delete the calibration file and start over
                    os.remove("calibrationData_R.pkl")
                    os.remove("calibrationData_L.pkl")
                    # restart calibration
                    calibrate(img1, img2)

            # corrupted file
            except EOFError as err:
                print(err)

        # start calibration if no calibration data exists
        else:

            calData_R = CalibrationData()
            calData_L = CalibrationData()

            calData_R.points = getTransformationPoints(imCal_R, 'left')
            # 13/6: 0 | 6/10: 1 | 10/15: 2 | 15/2: 3 | 2/17: 4 |
            # 17/3: 5 | 3/19: 6 | 19/7: 7 | 7/16: 8 | 16/8: 9 |
            # 8/11: 10 | 11/14: 11 | 14/9: 12 | 9/12: 13 | 12/5: 14 |
            # 5/20: 15 | 20/1: 16 | 1/18: 17 | 18/4: 18 | 4/13: 19
            # top, bottom, left, right
            # 12/9, 2/15, 8/16, 13/4
            calData_R.dstpoints = [13, 3, 8, 17]
            calData_R.transformation_matrix = manipulateTransformationPoints(
                img1, calData_R)

            calData_L.points = getTransformationPoints(imCal_L, "left")
            # 12/9, 2/15, 8/16, 13/4
            calData_L.dstpoints = [12, 2, 8, 18]
            calData_L.transformation_matrix = manipulateTransformationPoints(
                imCal_L, calData_L)

            cv2.destroyAllWindows()

            print("The dartboard image has now been normalized.")
            print("")

            cv2.imshow(winName4, imCal_R)
            test = cv2.waitKey(0)
            if test == 13:
                cv2.destroyWindow(winName4)
                cv2.destroyAllWindows()

            # write the calibration data to a file
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
    img1, img2 = loadReferenceImages()

    img1 = cv2.rotate(img1, cv2.ROTATE_180)
    img2 = cv2.rotate(img2, cv2.ROTATE_180)
    calibrate(img1, img2)
