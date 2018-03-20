import math
import os
import pickle
import sys

import cv2
import numpy as np

from darts_ui.darts_recognition.Classes import CalibrationData, EllipseDef
from darts_ui.darts_recognition.Draw import Draw
from darts_ui.darts_recognition.utils.ReferenceImages import loadReferenceImages
from darts_ui.darts_recognition.utils.VideoCapture import VideoStream

cal_l_path = './calibration_data/cal_l.pkl'
cal_r_path = './calibration_data/cal_r.pkl'


def _noop(x):
    pass


def _calculate_dst(center, radius=340):
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


def _get_live_feed():
    cam_l = VideoStream(src=0).start()
    cam_r = VideoStream(src=1).start()

    try:
        _, img_l = cam_l.read()
        _, img_r = cam_r.read()
    except:
        print('Could not init cams')
        raise
    finally:
        cam_l.stop()
        cam_r.stop()

    return img_l, img_r


def _getTransformPoints(img, cal_data):
    win1_name = 'Calibration'
    win2_name = 'Preview'
    board = Draw()
    im_copy = img.copy()
    dst_points = _calculate_dst((400, 400))
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


def _calibrate(img_l=None, img_r=None):
    cal_data_l = CalibrationData()
    cal_data_r = CalibrationData()

    if img_l is None or img_r is None:
        img_l, img_r = _get_live_feed()

    im_copy_l = img_l.copy()
    im_copy_r = img_r.copy()

    tranform_data_l, _ = _getTransformPoints(
        im_copy_l, cal_data_l)
    tranform_data_r, _ = _getTransformPoints(
        im_copy_r, cal_data_r)

    cal_data_l.points = _calculate_dst((400, 400))
    cal_data_l.transformation_matrix = tranform_data_l
    calibration_file_l = open(cal_l_path, 'wb')
    pickle.dump(cal_data_l, calibration_file_l, 0)
    calibration_file_l.close()

    cal_data_r.points = _calculate_dst((400, 400))
    cal_data_r.transformation_matrix = tranform_data_r
    calibration_file_r = open(cal_r_path, 'wb')
    pickle.dump(cal_data_r, calibration_file_r, 0)
    calibration_file_r.close()

    return cal_data_l, cal_data_r


def getCalibrationData():
    cal_l_exists = os.path.isfile(cal_l_path)
    cal_r_exists = os.path.isfile(cal_r_path)

    cal_data_l = CalibrationData()
    cal_data_r = CalibrationData()

    if cal_l_exists and cal_r_exists:
        calibration_file_l = open(cal_l_path, 'rb')
        calibration_file_r = open(cal_r_path, 'rb')

        cal_data_l = pickle.load(calibration_file_l)
        cal_data_r = pickle.load(calibration_file_r)

        calibration_file_l.close()
        calibration_file_r.close()
    else:
        cal_data_l, cal_data_r = _calibrate()

    return cal_data_l, cal_data_r


if __name__ == '__main__':
    img_l = None
    img_r = None
    # img_l, img_r = loadReferenceImages()

    _calibrate(img_l, img_r)
