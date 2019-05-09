__author__ = "Hannes Hoettinger"

# import numpy as np
import time
# import math
# import pickle
# from Classes import *
from cv2 import *
from MathFunctions import *
from DartsMapping import *
from Draw import *

DEBUG = True

winName = "test2"


def cam2gray(cam):
    success, image = cam.read()
    img_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return success, img_g


def get_threshold(cam, t):
    success, t_plus = cam2gray(cam)
    dimg = cv2.absdiff(t, t_plus)

    blur = cv2.GaussianBlur(dimg, (5, 5), 0)
    blur = cv2.bilateralFilter(blur, 9, 75, 75)
    _, thresh = cv2.threshold(blur, 60, 255, 0)

    return thresh


def diff2blur(cam, t):
    _, t_plus = cam2gray(cam)
    dimg = cv2.absdiff(t, t_plus)

    # kernel size important -> make accessible
    # filter noise from image distortions
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(dimg, -1, kernel)

    return t_plus, blur


def get_corners(img_in):
    # number of features to track is a distinctive feature
    # FeaturesToTrack important -> make accessible
    edges = cv2.goodFeaturesToTrack(img_in, 640, 0.0008, 1, mask=None, blockSize=3, useHarrisDetector=1, k=0.06)
    corners = np.int0(edges)

    return corners


def filter_corners(corners):
    cornerdata = []
    tt = 0
    mean_corners = np.mean(corners, axis=0)
    for i in corners:
        xl, yl = i.ravel()
        # filter noise to only get dart arrow
        # threshold important -> make accessible
        if abs(mean_corners[0][0] - xl) > 180:
            cornerdata.append(tt)
        if abs(mean_corners[0][1] - yl) > 120:
            cornerdata.append(tt)
        tt += 1

    corners_new = np.delete(corners, [cornerdata], axis=0)  # delete corners to form new array

    return corners_new


def filter_corners_line(corners, rows, cols):
    [vx, vy, x, y] = cv2.fitLine(corners, cv2.DIST_HUBER, 0, 0.1, 0.1)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)

    cornerdata = []
    tt = 0
    for i in corners:
        xl, yl = i.ravel()
        # check distance to fitted line, only draw corners within certain range
        distance = dist(0, lefty, cols - 1, righty, xl, yl)
        if distance > 40:  # threshold important -> make accessible
            cornerdata.append(tt)

        tt += 1

    corners_final = np.delete(corners, [cornerdata], axis=0)  # delete corners to form new array

    return corners_final


def get_real_location(corners_final, mount):
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
        if distance < 40:  # threshold important
            tt += 1
        else:
            cornerdata.append(tt)

    if tt < 3:
        corners_temp = cornerdata
        maxloc = np.argmax(corners_temp, axis=0)
        locationofdart = corners_temp[maxloc]
        print("### used different location due to noise!")

    return locationofdart


def get_darts(cam_r, cam_l, cal_data_r, cal_data_l, player_obj, gui):
    final_score = 0
    count = 0
    breaker = 0
    # threshold important -> make accessible
    min_thres = 2000 / 2
    max_thres = 15000 / 2

    # save score if score is below 1...
    old_score = player_obj.score

    # Read first image twice (issue somewhere) to start loop:
    _, _ = cam2gray(cam_r)
    _, _ = cam2gray(cam_l)
    # wait for camera
    time.sleep(0.1)
    success, t_r = cam2gray(cam_r)
    _, t_l = cam2gray(cam_l)

    while success:
        # wait for camera
        time.sleep(0.1)
        # check if dart hit the board
        thresh_r = get_threshold(cam_r, t_r)
        thresh_l = get_threshold(cam_l, t_l)

        print(cv2.countNonZero(thresh_r))
        # threshold important
        if (cv2.countNonZero(thresh_r) > min_thres and cv2.countNonZero(thresh_r) < max_thres) \
                or (cv2.countNonZero(thresh_l) > min_thres and cv2.countNonZero(thresh_l) < max_thres):
            # wait for camera vibrations
            time.sleep(0.2)
            # filter noise
            t_plus_r, blur_r = diff2blur(cam_r, t_r)
            t_plus_l, blur_l = diff2blur(cam_l, t_l)
            # get corners
            corners_r = get_corners(blur_r)
            corners_l = get_corners(blur_l)

            testimg = blur_r.copy()

            # dart outside?
            if corners_r.size < 40 and corners_l.size < 40:
                print("### dart not detected")
                continue

            # filter corners
            corners_f_r = filter_corners(corners_r)
            corners_f_l = filter_corners(corners_l)

            # dart outside?
            if corners_f_r.size < 30 and corners_f_l.size < 30:
                print("### dart not detected")
                continue

            # find left and rightmost corners#
            rows, cols = blur_r.shape[:2]
            corners_final_r = filter_corners_line(corners_f_r, rows, cols)
            corners_final_l = filter_corners_line(corners_f_l, rows, cols)

            _, thresh_r = cv2.threshold(blur_r, 60, 255, 0)
            _, thresh_l = cv2.threshold(blur_l, 60, 255, 0)

            # check if it was really a dart
            print(cv2.countNonZero(thresh_r))
            if cv2.countNonZero(thresh_r) > max_thres * 2 or cv2.countNonZero(thresh_l) > max_thres * 2:
                continue

            print("Dart detected")
            # dart was found -> increase counter
            breaker += 1

            # dart_info = DartDef()

            # get final darts location
            try:
                dart_info_r = DartDef()
                dart_info_l = DartDef()

                dart_info_r.corners = corners_final_r.size
                dart_info_l.corners = corners_final_l.size

                locationofdart_r = get_real_location(corners_final_r, "right")
                locationofdart_l = get_real_location(corners_final_l, "left")

                # check for the location of the dart with the calibration
                dartloc_r = get_transformed_location(locationofdart_r.item(0), locationofdart_r.item(1), cal_data_r)
                dartloc_l = get_transformed_location(locationofdart_l.item(0), locationofdart_l.item(1), cal_data_l)
                # detect region and score
                dart_info_r = get_dart_region(dartloc_r, cal_data_r)
                dart_info_l = get_dart_region(dartloc_l, cal_data_l)

                cv2.circle(testimg, (locationofdart_r.item(0), locationofdart_r.item(1)), 10, (255, 255, 255), 2, 8)
                cv2.circle(testimg, (locationofdart_r.item(0), locationofdart_r.item(1)), 2, (0, 255, 0), 2, 8)
            except:
                print("Something went wrong in finding the darts location!")
                breaker -= 1
                continue

            # "merge" scores
            if dart_info_r.base == dart_info_l.base and dart_info_r.multiplier == dart_info_l.multiplier:
                dart_info = dart_info_r
            # use the score of the image with more corners
            else:
                if dart_info_r.corners > dart_info_l.corners:
                    dart_info = dart_info_r
                else:
                    dart_info = dart_info_l

            print(dart_info.base, dart_info.multiplier)

            if breaker == 1:
                gui.dart1entry.insert(10, str(dart_info.base * dart_info.multiplier))
                dart = int(gui.dart1entry.get())
                cv2.imwrite("frame2.jpg", testimg)  # save dart1 frame
            elif breaker == 2:
                gui.dart2entry.insert(10, str(dart_info.base * dart_info.multiplier))
                dart = int(gui.dart2entry.get())
                cv2.imwrite("frame3.jpg", testimg)  # save dart2 frame
            elif breaker == 3:
                gui.dart3entry.insert(10, str(dart_info.base * dart_info.multiplier))
                dart = int(gui.dart3entry.get())
                cv2.imwrite("frame4.jpg", testimg)  # save dart3 frame

            player_obj.score -= dart

            if player_obj.score == 0 and dart_info.multiplier == 2:
                player_obj.score = 0
                breaker = 3
            elif player_obj.score <= 1:
                player_obj.score = old_score
                breaker = 3

            # save new diff img for next dart
            t_r = t_plus_r
            t_l = t_plus_l

            if player_obj.player == 1:
                gui.e1.delete(0, 'end')
                gui.e1.insert(10, player_obj.score)
            else:
                gui.e2.delete(0, 'end')
                gui.e2.insert(10, player_obj.score)

            final_score += (dart_info.base * dart_info.multiplier)

            if breaker == 3:
                break

            # cv2.imshow(winName, tnow)

        # missed dart
        elif cv2.countNonZero(thresh_r) < max_thres / 2 or cv2.countNonZero(thresh_l) < max_thres / 2:
            continue

        # if player enters zone - break loop
        elif cv2.countNonZero(thresh_r) > max_thres / 2 or cv2.countNonZero(thresh_l) > max_thres / 2:
            break

        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyWindow(winName)
            break

        count += 1

    gui.finalentry.delete(0, 'end')
    gui.finalentry.insert(10, final_score)

    print(final_score)


if __name__ == '__main__':
    print("Welcome to darts!")
    img = cv2.imread("frame1_R.jpg")
    img2 = cv2.imread("frame2_L.jpg")

    vidcap = cv2.VideoCapture("dummy_src/Darts_Testvideo_9_1.mp4")
    from_video = False

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
