import cv2
import numpy as np

from darts_ui.darts_recognition.Calibration import getCalibrationData
from darts_ui.darts_recognition.utils.VideoCapture import VideoStream

cal_data_l = None
cal_data_r = None


def show_corrected_live_stream():
    win_name_l = 'Preview Left'
    win_name_r = 'Preview Right'
    cam_l = VideoStream(src=0).start()
    cam_r = VideoStream(src=1).start()

    dst_points = np.array(cal_data_l.points, np.float32)
    src_points_l = np.array(cal_data_l.transformation_matrix, np.float32)
    src_points_r = np.array(cal_data_r.transformation_matrix, np.float32)

    cv2.namedWindow(win_name_l)
    cv2.namedWindow(win_name_r)

    fgbg = cv2.createBackgroundSubtractorMOG2()
    while (1):
        kill = cv2.waitKey(1) & 0xFF
        if kill == 13 or kill == 27:
            cam_l.stop()
            cam_r.stop()
            cv2.destroyAllWindows()
            break

        _, img_l = cam_l.read()
        _, img_r = cam_r.read()

        matrix_l = cv2.getPerspectiveTransform(src_points_l, dst_points)
        matrix_r = cv2.getPerspectiveTransform(src_points_r, dst_points)

        im_copy_l = cv2.warpPerspective(img_l, matrix_l, (800, 800))
        im_copy_r = cv2.warpPerspective(img_r, matrix_r, (800, 800))

        im_copy_l = fgbg.apply(im_copy_l)

        cv2.imshow(win_name_l, im_copy_l)
        cv2.imshow(win_name_r, im_copy_r)


def kickoff():
    global cal_data_l
    global cal_data_r

    cal_data_l, cal_data_r = getCalibrationData()

    show_corrected_live_stream()

if __name__ == '__main__':
    kickoff()
