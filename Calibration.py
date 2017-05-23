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
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from scipy.cluster import vq
# visual logging from https://github.com/dchaplinsky/visual-logging
#from logging import FileHandler
#from vlogging import VisualRecord
from numpy.linalg import inv
from VideoCapture import VideoStream

#import logging

DEBUG = True

#logger = logging.getLogger("demo")
#fh = FileHandler('test.html', mode="w")
#logger.setLevel(logging.DEBUG)
#logger.addHandler(fh)

points = []
newpoints = []
circle_radius = []
intersectp = []
rotated_rect = []
intersectp_s = []
center_ellipse = []
ellipse_vertices = []
center_dartboard = []
center_dartboard_new = []
ring_arr = []
winName3 = "hsv image colors?"
winName4 = "Calibration?"
winName5 = "Choose Ring"
#imCalRGB = cv2.imread("/Users/Hannes/Desktop/Darts/Dartboard_2.png")
try:
    cam = VideoStream(src=2).start()
    # frame = vs.read()
    # cam = cv2.VideoCapture(2)
    # cam.set(cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
    # cam.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
    success,imCalRGB = cam.read()
    imCalHSV = cv2.cvtColor(imCalRGB, cv2.COLOR_BGR2HSV)
except:
    #vidcap = cv2.VideoCapture("D:/Projekte/PycharmProjects/DartScore/Videos/dartscoreRaw_20170327_193108.avi")
    vidcap = cv2.VideoCapture("C:/Users/hanne/OneDrive/Projekte/GitHub/darts/Darts/Darts_Testvideo_9.mp4")
    success,imCalRGB = vidcap.read()
    #imCalRGB = cv2.imread("Image_kmeans_5clusters.png")
    imCalHSV = cv2.cvtColor(imCalRGB, cv2.COLOR_BGR2HSV)

    #logger.debug(VisualRecord("Hello from OpenCV", imCalHSV, "This is openCV image", fmt="png"))
    #logger.warning(VisualRecord("Hello from all", [imCalHSV, imCalRGB], fmt="png"))

calibrationComplete = False
new_image = imCalRGB.copy() # from camera = 480, 640  # from video 1080, 1920
image_proc_img = imCalRGB.copy()
imCalRGBorig = imCalRGB.copy()


def intersectLineCircle(center, radius, p1, p2):
    baX = p2[0] - p1[0]
    baY = p2[1] - p1[1]
    caX = center[0] - p1[0]
    caY = center[1] - p1[1]

    a = baX * baX + baY * baY
    bBy2 = baX * caX + baY * caY
    c = caX * caX + caY * caY - radius * radius

    pBy2 = bBy2 / a
    q = c / a

    disc = pBy2 * pBy2 - q
    if disc < 0:
        return False, None, False, None

    tmpSqrt = math.sqrt(disc)
    abScalingFactor1 = -pBy2 + tmpSqrt
    abScalingFactor2 = -pBy2 - tmpSqrt

    pint1 = p1[0] - baX * abScalingFactor1, p1[1] - baY * abScalingFactor1
    if disc == 0:
        return True, pint1, False, None

    pint2 = p1[0] - baX * abScalingFactor2, p1[1] - baY * abScalingFactor2
    return True, pint1, True, pint2

# line intersection
def intersectLines(pt1, pt2, ptA, ptB):
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE:
        return 0, 0

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    x = (x1 + r * dx1 + x + s * dx) / 2.0
    y = (y1 + r * dy1 + y + s * dy) / 2.0
    return x, y


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    x = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    y = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return x, y


def segment_intersection(p1, p2, p3, p4):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    x3 = p3[0]
    y3 = p3[1]
    x4 = p4[0]
    y4 = p4[1]
    d = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return px, py


def nothing(x):
    pass


def transformation(new_center, tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4):

    global new_image
    global center_dartboard
    global ellipse_vertices
    global newpoints
    global circle_radius
    global image_proc_img

    sectorangle = 2 * math.pi / 20
    # 12/9, 2/15, 8/16, 13/4
    #calData_R.dstpoints = [12, 2, 8, 18]
    ## sectors are sometimes different -> make accessible
    # 13/6: 0 | 6/10: 1 | 10/15: 2 | 15/2: 3 | 2/17: 4 | 17/3: 5 | 3/19: 6 | 19/7: 7 | 7/16: 8 | 16/8: 9 | 8/11: 10 |
    # 11/14: 11 | 14/9: 12 | 9/12: 13 | 12/5: 14 | 5/20: 15 | 20/1: 16 | 1/18: 17 | 18/4: 18 | 4/13: 19

    # used when line rectangle intersection at specific segment is used for transformation:
    i = 13  # 9/12 intersection
    newtop = [(new_center[0] + 170 * 2 * math.cos((0.5 + i) * sectorangle)),
              (new_center[1] + 170 * 2 * math.sin((0.5 + i) * sectorangle))]
    i = 3  # 15/2 intersection
    newbottom = [(new_center[0] + 170 * 2 * math.cos((0.5 + i) * sectorangle)),
                 (new_center[1] + 170 * 2 * math.sin((0.5 + i) * sectorangle))]
    i = 8 # 7/16 intersection
    newleft = [(new_center[0] + 170 * 2 * math.cos((0.5 + i) * sectorangle)),
               (new_center[1] + 170 * 2 * math.sin((0.5 + i) * sectorangle))]
    i = 18  # 18/4 intersection
    newright = [(new_center[0] + 170 * 2 * math.cos((0.5 + i) * sectorangle)),
                (new_center[1] + 170 * 2 * math.sin((0.5 + i) * sectorangle))]

    # get a fresh new image
    new_image = imCalRGB.copy()

    #
    src = np.array([(points[0][0]+tx1, points[0][1]+ty1), (points[1][0]+tx2, points[1][1]+ty2),
                    (points[2][0]+tx3, points[2][1]+ty3), (points[3][0]+tx4, points[3][1]+ty4)], np.float32)
    dst = np.array([newtop, newbottom, newleft, newright], np.float32)
    ret = cv2.getPerspectiveTransform(src, dst)
    # cv.GetPerspectiveTransform([points[0],points[1],points[2],points[3]],
    # [newtop, newbottom, newleft, newright],mapping)

    new_image = cv2.warpPerspective(new_image, ret, (800, 800))

    ## circle radius sometimes different? -> make accessible
    cv2.circle(new_image, (int(new_center[0]), int(new_center[1])), 170 * 2, (0, 255, 0), 1)  # outside double
    cv2.circle(new_image, (int(new_center[0]), int(new_center[1])), 160 * 2, (0, 255, 0), 1)  # inside double
    cv2.circle(new_image, (int(new_center[0]), int(new_center[1])), 107 * 2, (0, 255, 0), 1)  # outside treble
    cv2.circle(new_image, (int(new_center[0]), int(new_center[1])), 97 * 2, (0, 255, 0), 1)  # inside treble
    cv2.circle(new_image, (int(new_center[0]), int(new_center[1])), 16 * 2, (0, 255, 0), 1)  # 25
    cv2.circle(new_image, (int(new_center[0]), int(new_center[1])), 7 * 2, (0, 255, 0), 1)  # Bulls eye

    # 20 sectors...
    i = 0
    while (i < 20):
        cv2.line(new_image, (int(new_center[0]), int(new_center[1])), (
            int(new_center[0] + 170 * 2 * math.cos((0.5 + i) * sectorangle)),
            int(new_center[1] + 170 * 2 * math.sin((0.5 + i) * sectorangle))), (0, 255, 0), 1)
        i = i + 1

    cv2.circle(new_image, (int(newtop[0]), int(newtop[1])), 2, cv.CV_RGB(255, 255, 0), 2, 4)
    cv2.circle(new_image, (int(newbottom[0]), int(newbottom[1])), 2, cv.CV_RGB(255, 255, 0), 2, 4)
    cv2.circle(new_image, (int(newleft[0]), int(newleft[1])), 2, cv.CV_RGB(255, 255, 0), 2, 4)
    cv2.circle(new_image, (int(newright[0]), int(newright[1])), 2, cv.CV_RGB(255, 255, 0), 2, 4)

    cv2.imshow('manipulation', new_image)

    return ret


def calibrate():

    #cam = cv2.VideoCapture(1)

    global imCalRGB
    global new_image
    global image_proc_img
    global imCalRGBorig
    global intersectp
    global center_dartboard
    global points

    #imCalRGB = cv2.imread("/Users/Hannes/Desktop/Darts/Dartboard_2.png")
    #imCalRGB = cv2.imread("frame1.jpg")
    #success,imCalRGB = cam.read() #cam
    cv2.imwrite("frame1.jpg", imCalRGB)     # save calibration frame

    global calibrationComplete
    calibrationComplete = False

    while calibrationComplete == False:
        #Read calibration file, if exists
        if os.path.isfile("calibrationData.pkl"):
            try:
                # ToDo: adapt system to automatic calibration data
                #start a fresh set of points
                points = []

                calFile = open('calibrationData.pkl', 'rb')
                calData = CalibrationData()
                calData = pickle.load(calFile)
                #load the data into the global variables
                transformation_matrix = calData.transformationMatrix
                center_dartboard = calData.center_dartboard
                ring_radius = []
                ring_radius.append(calData.ring_radius[0])
                ring_radius.append(calData.ring_radius[1])
                ring_radius.append(calData.ring_radius[2])
                ring_radius.append(calData.ring_radius[3])
                ring_radius.append(calData.ring_radius[4])
                ring_radius.append(calData.ring_radius[5])      #append the 6 ring radii
                #close the file once we are done reading the data
                calFile.close()

                #copy image for old calibration data
                new_image = imCalRGB.copy()

                #now draw them out:
                height, width = imCalRGB.shape[:2]

                # get a fresh new image
                new_image = imCalRGB.copy()

                heightnew, widthnew = imCalRGB.shape[:2]

                new_image = cv2.warpPerspective(imCalRGBorig,transformation_matrix,(800,800))
                # cv.WarpPerspective(imCalRGB,new_image,mapping)
                cv2.imshow(winName4, new_image)

                cv2.circle(new_image, (int(center_dartboard[0]), int(center_dartboard[1])), ring_radius[0], (0, 255, 0),
                           1)  # outside double
                cv2.circle(new_image, (int(center_dartboard[0]), int(center_dartboard[1])), ring_radius[1], (0, 255, 0),
                           1)  # inside double
                cv2.circle(new_image, (int(center_dartboard[0]), int(center_dartboard[1])), ring_radius[2], (0, 255, 0),
                           1)  # outside treble
                cv2.circle(new_image, (int(center_dartboard[0]), int(center_dartboard[1])), ring_radius[3], (0, 255, 0), 1)  # inside treble
                cv2.circle(new_image, (int(center_dartboard[0]), int(center_dartboard[1])), ring_radius[4], (0, 255, 0), 1)  # 25
                cv2.circle(new_image, (int(center_dartboard[0]), int(center_dartboard[1])), ring_radius[5], (0, 255, 0), 1)  # Bulls eye

                # 20 sectors...
                sectorangle = 2 * math.pi / 20
                i = 0
                while (i < 20):
                    cv2.line(new_image, (int(center_dartboard[0]), int(center_dartboard[1])), (
                        int(center_dartboard[0] + 170 * 2 * math.cos((0.5 + i) * sectorangle)),
                        int(center_dartboard[1] + 170 * 2 * math.sin((0.5 + i) * sectorangle))), (0, 255, 0), 1)
                    i = i + 1

                cv2.imshow(winName4, new_image)

                test = cv2.waitKey(0)
                if test == 13:
                    cv2.destroyAllWindows()
                    #we are good with the previous calibration data
                    calibrationComplete = True
                else:
                    cv2.destroyAllWindows()
                    calibrationComplete = True
                    #delete the calibration file and start over
                    os.remove("calibrationData.pkl")

            #corrupted file
            except EOFError as err:
                print err

        else:
            # ToDo: remove manual calibration and adapt system to automatic calibration data
            # create new image for imageprocessing
            # image_proc_img = new_image.copy()
            image_proc_img = imCalRGB.copy()
            # call image processing function
            imagproccalib()

            height, width = imCalRGB.shape[:2]

            new_center = (400, 400)

            # raw_loc_mat = np.zeros((height, width))
            if DEBUG:
            #cv2.namedWindow('image')
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                # create trackbars for color change
                cv2.createTrackbar('cx', 'image', 0, 20, nothing)
                cv2.createTrackbar('cy', 'image', 0, 20, nothing)

                cv2.createTrackbar('tx1', 'image', 0, 20, nothing)
                cv2.createTrackbar('ty1', 'image', 0, 20, nothing)

                cv2.createTrackbar('tx2', 'image', 0, 20, nothing)
                cv2.createTrackbar('ty2', 'image', 0, 20, nothing)

                cv2.createTrackbar('tx3', 'image', 0, 20, nothing)
                cv2.createTrackbar('ty3', 'image', 0, 20, nothing)

                cv2.createTrackbar('tx4', 'image', 0, 20, nothing)
                cv2.createTrackbar('ty4', 'image', 0, 20, nothing)

                cv2.setTrackbarPos('cx', 'image', 10)
                cv2.setTrackbarPos('cy', 'image', 10)

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

                while (1):
                    cv2.imshow('image', new_image)
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27:
                        break

                    # get current positions of four trackbars
                    cx = cv2.getTrackbarPos('cx', 'image') - 10
                    cy = cv2.getTrackbarPos('cy', 'image') - 10
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
                        new_image[:] = 0
                    else:
                        # transform the image to form a perfect circle
                        transformation_matrix = transformation(new_center, tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4)

            else:
                transformation_matrix = transformation(new_center, 3, -1, 4, -3, 0, 0, 1, 5)

            cv2.destroyAllWindows()

            print "The dartboard image has now been normalized."
            print ""

            cv2.imshow(winName4, new_image)
            cv2.setMouseCallback(winName4, on_mouse_new)
            test = cv2.waitKey(0)
            if test == 13:
                cv2.destroyWindow(winName4)
                cv2.destroyAllWindows()

            ## sectors are sometimes different -> make accessible
            ring_radius = [7 * 2, 16 * 2, 97 * 2, 107 * 2, 160 * 2, 170 * 2]

            # time.sleep(5)
            # cv2.destroyWindow(winName)
            #save valuable calibration data into a structure
            calData = CalibrationData()
            calData.transformationMatrix = transformation_matrix
            calData.center_dartboard = new_center
            calData.ring_radius = ring_radius

            #write the calibration data to a file
            calFile = open("calibrationData.pkl", "wb")
            pickle.dump(calData, calFile, 0)
            calFile.close()

            calibrationComplete = True


    cv2.destroyAllWindows()



def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # events
            global points

            # append user clicked points
            points.append((x, y))
            print points
            cv2.circle(imCalRGB, (x, y), 3,(255, 0, 0),2, 8)
            cv2.imshow(winName3, imCalRGB)

def on_mouse_new(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # events
            global points

            # append user clicked points
            points.append((x, y))
            print points
            cv2.circle(new_image, (x, y), 3,(255, 0, 0),2, 8)
            cv2.imshow(winName4, new_image)
        # key.set()

def on_mouse_rings(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # events
            global points

            # append user clicked points
            points.append((x, y))
            print points
            cv2.circle(new_image, (x, y), 3,(255, 0, 0),2, 8)
            cv2.imshow(winName5, new_image)
        # key.set()


def imagproccalib():

    global intersectp
    global intersectp_s
    global center_ellipse
    global ellipse_vertices
    global newpoints
    global circle_radius


    # imCalRGB = cv2.cvtColor(imCal, cv2.COLOR_RGB2GRAY)
    imCalHSV = cv2.cvtColor(image_proc_img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(imCalHSV, -1, kernel)
    h, s, imCal = cv2.split(blur)

    ## threshold important -> make accessible
    ret, thresh2 = cv2.threshold(imCal, 128, 255, cv2.THRESH_BINARY_INV) # using a video
    #ret, thresh2 = cv2.threshold(imCal, 140, 255, cv2.THRESH_BINARY_INV)
    #ret, thresh2 = cv2.threshold(imCal, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ## kernel size important -> make accessible
    # very important -> removes lines outside the outer ellipse -> find ellipse
    kernel = np.ones((3, 3), np.uint8)
    thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("thresh2", thresh2)

    edged = cv2.Canny(thresh2, 250, 255)

    # return the edged image
    cv2.imshow("test", edged)

    # find enclosing ellipse
    contours, hierarchy = cv2.findContours(thresh2, 1, 2)
    #cv2.drawContours(image_proc_img, contours, -1, (0, 255, 0), 3)

    ## contourArea threshold important -> make accessible
    for cnt in contours:
        try: #threshold critical, change on demand?
            if 200000/4 < cv2.contourArea(cnt) < 1000000/4:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(image_proc_img, ellipse, (0, 255, 0), 2)

                x, y = ellipse[0]
                a, b = ellipse[1]
                angle = ellipse[2]

                center_ellipse = (x, y)

                a = a/2
                b = b/2

                cv2.ellipse(image_proc_img, (int(x), int(y)), (int(a), int(b)), int(angle), 0.0, 360.0, cv.CV_RGB(255, 0, 0))

                #cv2.circle(image_proc_img, (int(x), int(y-b/2)), 3, cv.CV_RGB(0, 255, 0), 2, 8)

                # vertex calculation
                xb = b * math.cos(angle)
                yb = b * math.sin(angle)

                xa = a * math.sin(angle)
                ya = a * math.cos(angle)

                rect = cv2.minAreaRect(cnt)
                box = cv2.cv.BoxPoints(rect)
                box = np.int0(box)
                #cv2.drawContours(image_proc_img, [box], 0, (0, 0, 255), 2)

        # corrupted file
        except:
            print "error"

    cv2.imshow("test4", image_proc_img)

    circle_radius = a

    anglezone1 = (angle - 5, angle + 5)
    anglezone2 = (angle - 100, angle - 80)

    # transform ellipse to a perfect circle?
    height, width = image_proc_img.shape[:2]

    angle = (angle) * math.pi / 180

    # build transformation matrix http://math.stackexchange.com/questions/619037/circle-affine-transformation
    R1 = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    R2 = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])

    T1 = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
    T2 = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

    D = np.array([[1, 0, 0], [0, a / b, 0], [0, 0, 1]])

    M = T2.dot(R2.dot(D.dot(R1.dot(T1))))

    M_inv = np.linalg.inv(M)

    # fit line to find intersec point for dartboard center point
    # change houghline parameter of angle
    lines = cv2.HoughLines(edged, 1, np.pi / 70, 100, 100)

    p = []
    lines_seg = []
    counter = 0

    ## sector angles important -> make accessible
    for rho, theta in lines[0]:
        # split between horizontal and vertical lines (take only lines in certain range)
        if theta > np.pi / 180 * anglezone1[0] and theta < np.pi / 180 * anglezone1[1]:

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 3000 * (-b))
            y1 = int(y0 + 3000 * (a))
            x2 = int(x0 - 3000 * (-b))
            y2 = int(y0 - 3000 * (a))

            for rho1, theta1 in lines[0]:

                if theta1 > np.pi / 180 * anglezone2[0] and theta1 < np.pi / 180 * anglezone2[1]:

                    a = np.cos(theta1)
                    b = np.sin(theta1)
                    x0 = a * rho1
                    y0 = b * rho1
                    x3 = int(x0 + 3000 * (-b))
                    y3 = int(y0 + 3000 * (a))
                    x4 = int(x0 - 3000 * (-b))
                    y4 = int(y0 - 3000 * (a))

                    if y1 == y2 and y3 == y4:  # Horizontal Lines
                        diff = abs(y1 - y3)
                    elif x1 == x2 and x3 == x4:  # Vertical Lines
                        diff = abs(x1 - x3)
                    else:
                        diff = 0

                    if diff < 200 and diff is not 0:
                        continue

                    #cv2.line(image_proc_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    #cv2.line(image_proc_img, (x3, y3), (x4, y4), (255, 0, 0), 1)

                    p.append((x1, y1))
                    p.append((x2, y2))
                    p.append((x3, y3))
                    p.append((x4, y4))

                    intersectpx, intersectpy = intersectLines(p[counter], p[counter + 1], p[counter + 2],
                                                              p[counter + 3])

                    # consider only intersection close to the center of the image
                    if (intersectpx < 100 or intersectpx > 800) or (intersectpy < 100 or intersectpy > 800):
                        continue

                    intersectp.append((intersectpx, intersectpy))

                    lines_seg.append([(x1, y1), (x2, y2)])
                    lines_seg.append([(x3, y3), (x4, y4)])

                    cv2.line(image_proc_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.line(image_proc_img, (x3, y3), (x4, y4), (255, 0, 0), 1)

                    # point offset
                    counter = counter + 4

    ellipse_vertices.append([(box[1][0] + box[2][0]) / 2, (box[1][1] + box[2][1]) / 2])
    ellipse_vertices.append([(box[2][0] + box[3][0]) / 2, (box[2][1] + box[3][1]) / 2])
    ellipse_vertices.append([(box[0][0] + box[3][0]) / 2, (box[0][1] + box[3][1]) / 2])
    ellipse_vertices.append([(box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2])

    testpoint1 = M.dot(np.transpose(np.hstack([center_ellipse, 1])))
    testpoint2 = M.dot(np.transpose(np.hstack([ellipse_vertices[0], 1])))
    testpoint3 = M.dot(np.transpose(np.hstack([ellipse_vertices[1], 1])))
    testpoint4 = M.dot(np.transpose(np.hstack([ellipse_vertices[2], 1])))
    testpoint5 = M.dot(np.transpose(np.hstack([ellipse_vertices[3], 1])))

    newpoints.append([testpoint2[0], testpoint2[1]])
    newpoints.append([testpoint3[0], testpoint3[1]])
    newpoints.append([testpoint4[0], testpoint4[1]])
    newpoints.append([testpoint5[0], testpoint5[1]])
    newpoints.append([testpoint1[0], testpoint1[1]])

    for lin in lines_seg:
        line_p1 = M.dot(np.transpose(np.hstack([lin[0], 1])))
        line_p2 = M.dot(np.transpose(np.hstack([lin[1], 1])))
        inter1, inter_p1, inter2, inter_p2 = intersectLineCircle(np.asarray(center_ellipse), circle_radius, np.asarray(line_p1), np.asarray(line_p2))
        #cv2.line(image_proc_img, (int(line_p1[0]), int(line_p1[1])), (int(line_p2[0]), int(line_p2[1])), cv.CV_RGB(255, 0, 0), 2, 8)
        if inter1:
            #cv2.circle(image_proc_img, (int(inter_p1[0]), int(inter_p1[1])), 3, cv.CV_RGB(0, 0, 255), 2, 8)
            inter_p1 = M_inv.dot(np.transpose(np.hstack([inter_p1, 1])))
            #cv2.circle(image_proc_img, (int(inter_p1[0]), int(inter_p1[1])), 3, cv.CV_RGB(0, 0, 255), 2, 8)
            if inter2:
                #cv2.circle(image_proc_img, (int(inter_p1[0]), int(inter_p1[1])), 3, cv.CV_RGB(0, 0, 255), 2, 8)
                inter_p2 = M_inv.dot(np.transpose(np.hstack([inter_p2, 1])))
                #cv2.circle(image_proc_img, (int(inter_p2[0]), int(inter_p2[1])), 3, cv.CV_RGB(0, 0, 255), 2, 8)
                intersectp_s.append(inter_p1)
                intersectp_s.append(inter_p2)

    try:
        # calculate mean val between: 0,4;1,5;2,6;3,7
        new_intersect = np.mean(([intersectp_s[0],intersectp_s[4]]), axis=0, dtype=np.float32)
        points.append(new_intersect) # top
        new_intersect = np.mean(([intersectp_s[1], intersectp_s[5]]), axis=0, dtype=np.float32)
        points.append(new_intersect) # bottom
        new_intersect = np.mean(([intersectp_s[2], intersectp_s[6]]), axis=0, dtype=np.float32)
        points.append(new_intersect) # left
        new_intersect = np.mean(([intersectp_s[3], intersectp_s[7]]), axis=0, dtype=np.float32)
        points.append(new_intersect) # right
    except:
        pointarray = np.array(intersectp_s[:4]) # take only first 4 arguments
        top_idx = [np.argmin(pointarray[:, 1])][0]
        pointarray_1 = np.delete(pointarray, [top_idx], axis=0)
        bot_idx = [np.argmax(pointarray_1[:, 1])][0] + 1
        pointarray_2 = np.delete(pointarray_1, [bot_idx], axis=0)
        left_idx = [np.argmin(pointarray_2[:, 0])][0] + 2
        right_idx = [np.argmax(pointarray_2[:, 0])][0] + 2

        points.append(intersectp_s[top_idx])  # top
        points.append(intersectp_s[bot_idx])  # bottom
        points.append(intersectp_s[left_idx])  # left
        points.append(intersectp_s[right_idx])  # right

    #points.append(intersectp_s[0])  # top
    #points.append(intersectp_s[1])  # bottom
    #points.append(intersectp_s[2])  # left
    #points.append(intersectp_s[3])  # right

    cv2.circle(image_proc_img, (int(points[0][0]), int(points[0][1])), 3, cv.CV_RGB(255, 0, 0), 2, 8)
    cv2.circle(image_proc_img, (int(points[1][0]), int(points[1][1])), 3, cv.CV_RGB(255, 0, 0), 2, 8)
    cv2.circle(image_proc_img, (int(points[2][0]), int(points[2][1])), 3, cv.CV_RGB(255, 0, 0), 2, 8)
    cv2.circle(image_proc_img, (int(points[3][0]), int(points[3][1])), 3, cv.CV_RGB(255, 0, 0), 2, 8)

    ## ellipse vertices
    #cv2.circle(image_proc_img, (int(ellipse_vertices[0][0]), int(ellipse_vertices[0][1])), 3, cv.CV_RGB(255, 0, 255), 2, 8)
    #cv2.circle(image_proc_img, (int(ellipse_vertices[1][0]), int(ellipse_vertices[1][1])), 3, cv.CV_RGB(255, 0, 255), 2, 8)
    #cv2.circle(image_proc_img, (int(ellipse_vertices[2][0]), int(ellipse_vertices[2][1])), 3, cv.CV_RGB(255, 0, 255), 2, 8)
    #cv2.circle(image_proc_img, (int(ellipse_vertices[3][0]), int(ellipse_vertices[3][1])), 3, cv.CV_RGB(255, 0, 255), 2, 8)

    rotated_rect.append((box[1], box[2]))
    rotated_rect.append((box[2], box[3]))
    rotated_rect.append((box[0], box[3]))
    rotated_rect.append((box[0], box[1]))

    winName2 = "th circles?"
    cv2.namedWindow(winName2, cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow(winName2, image_proc_img)

    #winName2 = "th test?"
    #cv2.namedWindow(winName2, cv2.CV_WINDOW_AUTOSIZE)
    #cv2.imshow(winName2, dst)

    end = cv2.waitKey(0)
    if end == 13:
        cv2.destroyAllWindows()


#For file IO
class CalibrationData:
    def __init__(self):
        #for perspective transform
        self.transformationMatrix = []
        #for calculating the first angle
        self.center_dartboard = []
        #radii of the rings, there are 6 in total
        self.ring_radius = []


if __name__ == '__main__':
    print "Welcome to darts!"
    #getTransformation()
    calibrate()
