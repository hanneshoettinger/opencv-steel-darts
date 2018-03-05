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
from utils.ReferenceImages import loadReferenceImages

dst_points = [(161, 157), (641, 157), (160, 640), (641, 642)]


def _noop(x):
    pass


def calculate_dst(center, radius=340):
    sector_angle = 2 * math.pi / 20

    i = 12  # 9/12 intersection
    radius = radius  # * 2
    new_top = [(center[0] + radius * _get_sec_cos(sector_angle, i)),
               (center[1] + radius * _get_sec_sin(sector_angle, i))]
    i = 2  # 15/2 intersection
    new_bottom = [(center[0] + radius * _get_sec_cos(sector_angle, i)),
                  (center[1] + radius * _get_sec_sin(sector_angle, i))]
    i = 7  # 7/16 intersection
    new_left = [(center[0] + radius * _get_sec_cos(sector_angle, i)),
                (center[1] + radius * _get_sec_sin(sector_angle, i))]
    i = 17  # 18/4 intersection
    new_right = [(center[0] + radius * _get_sec_cos(sector_angle, i)),
                 (center[1] + radius * _get_sec_sin(sector_angle, i))]

    return [new_top, new_bottom, new_left, new_right]


def _get_sec_sin(sector_angle, i):
    return math.sin((0.5 + i) * sector_angle)


def _get_sec_cos(sector_angle, i):
    return math.cos((0.5 + i) * sector_angle)


def getTransformPoints(img, cal_data):
    win1_name = 'Calibration'
    win2_name = 'Preview'
    board = Draw()
    im_copy = img.copy()
    dst_points = calculate_dst((400, 400))
    src_points = [[0, 0], [0, 0], [0, 0], [0, 0]]
    position = 0

    def _storeCoordinates(event, x, y, params, _):
        if event is cv2.EVENT_LBUTTONDOWN:
            src_points[position] = [x, y]

    def _finetune(value, position, points_copy, x):
        if x:
            points_copy[position][0] = points_copy[position][0] + value
        else:
            points_copy[position][1] = points_copy[position][1] + value

        return points_copy

    cv2.namedWindow(win1_name)
    cv2.createTrackbar('Position', win1_name, 0, 3, _noop)
    cv2.createTrackbar('X:', win1_name, 20, 40, _noop)
    cv2.createTrackbar('Y:', win1_name, 20, 40, _noop)
    cv2.setMouseCallback(win1_name, _storeCoordinates)
    cv2.imshow(win1_name, img)

    while (1):
        kill = cv2.waitKey(1) & 0xFF
        if kill == 13 or kill == 27:
            break

        points_copy = np.copy(src_points)
        new_position = cv2.getTrackbarPos('Position', win1_name)
        x = cv2.getTrackbarPos('X:', win1_name) - 20
        y = cv2.getTrackbarPos('Y:', win1_name) - 20
        points_copy = _finetune(x, position, points_copy, True)
        points_copy = _finetune(y, position, points_copy, False)
        if new_position != position:
            position = new_position
            src_points = np.copy(points_copy)
            cv2.setTrackbarPos('X:', win1_name, 20)
            cv2.setTrackbarPos('Y:', win1_name, 20)
        matrix = cv2.getPerspectiveTransform(
            np.array(points_copy, np.float32),
            np.array(dst_points, np.float32))
        im_copy = cv2.warpPerspective(img, matrix, (800, 800))
        board.drawBoard(im_copy, cal_data)
        cv2.imshow(win2_name, im_copy)

    im_copy = cv2.warpPerspective(img, matrix, (800, 800))
    cv2.destroyAllWindows()
    return points_copy, im_copy


def calibrate(img_l, img_r):

    cal_data = CalibrationData()

    im_copy1 = img_l.copy()
    im_copy2 = img_r.copy()

    data, im_transformed = getTransformPoints(im_copy2, cal_data)
    print(data)

    global calibrationComplete
    calibrationComplete = False

    board = Draw()

    cv2.namedWindow('board')

    im_transformed = board.drawBoard(im_transformed, cal_data)
    cv2.imshow('board', im_transformed)
    cv2.imshow('org', im_copy1)

    while (1):
        kill = cv2.waitKey(1) & 0xFF
        if kill == 27 or kill == 13:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_r, img_r = loadReferenceImages()

    img_r = cv2.rotate(img_r, cv2.ROTATE_180)
    img_r = cv2.rotate(img_r, cv2.ROTATE_180)
    calibrate(img_r, img_r)
