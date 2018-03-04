import cv2
import numpy as np

_RETR_MODES = (
    cv2.RETR_EXTERNAL,
    cv2.RETR_LIST,
    cv2.RETR_CCOMP,
    cv2.RETR_TREE,
    cv2.RETR_FLOODFILL
)

_MORPH_MODES = (
    cv2.MORPH_ERODE,
    cv2.MORPH_DILATE,
    cv2.MORPH_OPEN,
    cv2.MORPH_CLOSE,
    cv2.MORPH_GRADIENT,
    cv2.MORPH_TOPHAT,
    cv2.MORPH_BLACKHAT,
    cv2.MORPH_HITMISS
)

_APPROX_MODES = (
    cv2.CHAIN_APPROX_NONE,
    cv2.CHAIN_APPROX_SIMPLE,
    cv2.CHAIN_APPROX_TC89_L1,
    cv2.CHAIN_APPROX_TC89_KCOS
)


def _noop(x):
    pass


def calibrateBinaryThresh(img):
    win_name = 'calibrateThresh'
    min = 125
    max = 255
    im_copy = img.copy()

    cv2.namedWindow(win_name)
    cv2.createTrackbar('min', win_name, min, 255, _noop)
    cv2.createTrackbar('max', win_name, max, 255, _noop)

    while (1):
        cv2.imshow(win_name, im_copy)
        kill = cv2.waitKey(1) & 0xFF
        if kill == 27 or kill == 13:
            break

        min = cv2.getTrackbarPos('min', win_name)
        max = cv2.getTrackbarPos('max', win_name)

        _, im_copy = cv2.threshold(img, min, max, cv2.THRESH_BINARY)

    cv2.destroyWindow(win_name)

    return im_copy, min, max


def calibrateMorph(img):
    win_name = 'calibrateMorph'
    im_copy = img.copy()
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    cv2.namedWindow(win_name)
    cv2.createTrackbar('kernelSize', win_name, 5, 5, _noop)
    cv2.createTrackbar(
        'mode', win_name, _MORPH_MODES[0], len(_MORPH_MODES) - 1, _noop)

    while (1):
        cv2.imshow(win_name, im_copy)
        kill = cv2.waitKey(1) & 0xFF
        if kill == 27 or kill == 13:
            break

        kernel_size = cv2.getTrackbarPos('kernelSize', win_name)
        mode = cv2.getTrackbarPos('mode', win_name)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        im_copy = cv2.morphologyEx(img, _MORPH_MODES[mode], kernel)

    cv2.destroyWindow(win_name)
    return im_copy, kernel, _MORPH_MODES[mode]


def calibrateContours(img, preview):
    win_name = 'calibrateContours'
    preview_copy = preview.copy()

    cv2.namedWindow(win_name)
    cv2.createTrackbar('mode', win_name,
                       _RETR_MODES[1], len(_RETR_MODES), _noop)
    cv2.createTrackbar('method', win_name,
                       _APPROX_MODES[2], len(_APPROX_MODES), _noop)

    while (1):
        cv2.imshow(win_name, preview_copy)
        kill = cv2.waitKey(1) & 0xFF
        if kill == 27 or kill == 13:
            break

        mode = cv2.getTrackbarPos('mode', win_name)
        method = cv2.getTrackbarPos('method', win_name)

        _, contours, hierarchy = cv2.findContours(img, mode, method)
        cv2.drawContours(preview_copy, contours, -1, (255, 0, 0), 2)

    cv2.destroyWindow(win_name)
    return contours, hierarchy, preview_copy


def _drawEllipseWithThreshhold(contours, min_thresh, max_thresh, img):
    ellipse = None
    preview_copy = img.copy()

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if min_thresh < area < max_thresh:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(preview_copy, ellipse, (0, 255, 0), 2)
            cv2.drawContours(preview_copy, contours, idx, (255, 255, 255), 2)

    return ellipse, preview_copy


def calibrateEllipse(contours, preview, min_thresh=25000):
    win_name = 'calibrateEllipse'
    preview_copy = preview.copy()
    max_thresh = int(max(np.rint([cv2.contourArea(c)
                                  for c in contours]))) + min_thresh
    thresh_multiplier = 1000
    max_thresh_trackbar = int(np.rint(max_thresh / thresh_multiplier))
    max_thresh_value = int(np.rint(max_thresh_trackbar / 2))
    min_thresh_value = int(np.rint(min_thresh / thresh_multiplier))

    cv2.namedWindow(win_name)
    cv2.createTrackbar('minThresh', win_name,
                       min_thresh_value, max_thresh_trackbar, _noop)
    cv2.createTrackbar('maxThresh', win_name,
                       max_thresh_value, max_thresh_trackbar, _noop)

    while (1):
        cv2.imshow(win_name, preview_copy)
        kill = cv2.waitKey(1) & 0xFF
        if kill == 27 or kill == 13:
            break

        new_min_thresh_value = cv2.getTrackbarPos('minThresh', win_name)
        new_max_thresh_value = cv2.getTrackbarPos('maxThresh', win_name)

        same_min = new_min_thresh_value == min_thresh_value
        same_max = new_max_thresh_value == max_thresh_value

        if not same_min or not same_max:
            min_thresh = new_min_thresh_value * thresh_multiplier
            min_thresh_value = new_min_thresh_value
            max_thresh = new_max_thresh_value * thresh_multiplier
            max_thresh_value = new_max_thresh_value
            ellipse, preview_copy = _drawEllipseWithThreshhold(
                contours, min_thresh, max_thresh, preview)

    cv2.destroyWindow(win_name)
    return ellipse


def calibrateCanny(img):
    win_name = 'calibrateCanny'
    im_copy = img.copy()
    min_thresh = 0
    max_thresh = 255

    cv2.namedWindow(win_name)
    cv2.createTrackbar('minThresh', win_name, min_thresh, max_thresh, _noop)
    cv2.createTrackbar('maxThresh', win_name, max_thresh, max_thresh, _noop)

    while (1):
        cv2.imshow(win_name, im_copy)
        kill = cv2.waitKey(1) & 0xFF
        if kill == 27 or kill == 13:
            break

        min_thresh = cv2.getTrackbarPos('minThresh', win_name)
        max_thresh = cv2.getTrackbarPos('maxThresh', win_name)

        im_copy = cv2.Canny(img, min_thresh, max_thresh)

    cv2.destroyWindow(win_name)
    return im_copy


def calibrateHoughLines(img, preview):
    win_name = 'calibrateHoughLines'
    rho_value = 1
    theta_value = np.pi / 80
    threshold = 100
    max_lines = 100
    lines = None
    sample_line_dimension = max(img.shape)

    cv2.namedWindow(win_name)
    cv2.createTrackbar('threshold', win_name, threshold, 255, _noop)
    cv2.createTrackbar('lines', win_name, max_lines, 255, _noop)
    cv2.createTrackbar('theta', win_name, 80, 359, _noop)

    while (1):
        preview_copy = preview.copy()
        kill = cv2.waitKey(1) & 0xFF
        if kill == 27 or kill == 13:
            break

        threshold = cv2.getTrackbarPos('threshold', win_name)
        max_lines = cv2.getTrackbarPos('lines', win_name)
        theta_value = np.pi / (cv2.getTrackbarPos('theta', win_name) + 1)

        lines = cv2.HoughLines(
            img, rho_value, theta_value, threshold, max_lines)

        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + sample_line_dimension * (-b))
                    y1 = int(y0 + sample_line_dimension * (a))
                    x2 = int(x0 - sample_line_dimension * (-b))
                    y2 = int(y0 - sample_line_dimension * (a))

                    cv2.line(preview_copy, (x1, y1), (x2, y2),
                             (255, 0, 255), 1)
        cv2.imshow(win_name, preview_copy)

    cv2.destroyWindow(win_name)
    return lines
