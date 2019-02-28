__author__ = "Hannes Hoettinger"

import numpy as np
import cv2
import time
import cv2.cv as cv
import math
import pickle

img = cv2.imread("D:\Projekte\PycharmProjects\DartsScorer\Darts\Dartboard_2.png")
img2 = cv2.imread("D:\Projekte\PycharmProjects\DartsScorer\Darts\Dartboard_3.png")

vidcap = cv2.VideoCapture("C:\Users\hanne\OneDrive\Projekte\GitHub\darts\Darts\Darts_Testvideo_9_1.mp4")
from_video = True

DEBUG = True

winName = "test2"

center_dartboard = []
ring_radius = []
transformation_matrix = []

class dartThrow:
    def __init__(self):
        self.base = -1
        self.multiplier = -1
        self.magnitude = -1
        self.angle = -1


#For file IO
class CalibrationData:
    def __init__(self):
        #for perspective transform
        self.top = []
        self.bottom = []
        self.left = []
        self.right = []
        #for calculating the first angle
        self.init_point_arr = []
        self.center_dartboard = []
        #initial angle of the 20 / 1 points divider
        self.ref_angle = []
        #radii of the rings, there are 6 in total
        self.ring_radius = []
        self.transformationMatrix = []


## improve and make circle radius accessible
def drawBoard():
    raw_loc_mat = np.zeros((800, 800, 3))

    # draw board
    cv2.circle(raw_loc_mat, (400, 400), 170 * 2, (255, 255, 255), 1)  # outside double
    cv2.circle(raw_loc_mat, (400, 400), 160 * 2, (255, 255, 255), 1)  # inside double
    cv2.circle(raw_loc_mat, (400, 400), 107 * 2, (255, 255, 255), 1)  # outside treble
    cv2.circle(raw_loc_mat, (400, 400), 97 * 2, (255, 255, 255), 1)  # inside treble
    cv2.circle(raw_loc_mat, (400, 400), 16 * 2, (255, 255, 255), 1)  # 25
    cv2.circle(raw_loc_mat, (400, 400), 7 * 2, (255, 255, 255), 1)  # Bulls eye

    # 20 sectors...
    sectorangle = 2 * math.pi / 20
    i = 0
    while (i < 20):
        cv2.line(raw_loc_mat, (400, 400), (
            int(400 + 170 * 2 * math.cos((0.5 + i) * sectorangle)),
            int(400 + 170 * 2 * math.sin((0.5 + i) * sectorangle))), (255, 255, 255), 1)
        i = i + 1

    return raw_loc_mat

def dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1
    py = y2-y1

    something = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = math.sqrt(dx*dx + dy*dy)

    return dist

def DartLocation(x_coord,y_coord):
    try:

            #start a fresh set of points
            points = []

            calFile = open('calibrationData.pkl', 'rb')
            calData = CalibrationData()
            calData = pickle.load(calFile)
            #load the data into the global variables
            global transformation_matrix
            transformation_matrix = calData.transformationMatrix
            global ring_radius
            ring_radius.append(calData.ring_radius[0])
            ring_radius.append(calData.ring_radius[1])
            ring_radius.append(calData.ring_radius[2])
            ring_radius.append(calData.ring_radius[3])
            ring_radius.append(calData.ring_radius[4])
            ring_radius.append(calData.ring_radius[5])  # append the 6 ring radii
            global center_dartboard
            center_dartboard = calData.center_dartboard

            #close the file once we are done reading the data
            calFile.close()
            #print "Raw dart location:"
            #print x_coord,y_coord

            # transform only the hit point with the saved transformation matrix
            dart_loc_temp = np.array([[x_coord, y_coord]], dtype="float32")
            dart_loc_temp = np.array([dart_loc_temp])
            dart_loc = cv2.perspectiveTransform(dart_loc_temp, transformation_matrix)
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
def DartRegion(dart_loc):
    try:
            height = 800
            width = 800

            global dartInfo

            dartInfo = dartThrow()

            #find the magnitude and angle of the dart
            vx = (dart_loc[0] - center_dartboard[0])
            vy = (center_dartboard[1] - dart_loc[1])

            # reference angle for atan2 conversion
            ref_angle = 81

            dart_magnitude = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
            dart_angle = math.fmod(((math.atan2(vy,vx) * 180/math.pi) + 360 - ref_angle), 360)

            dartInfo.magnitude = dart_magnitude
            dartInfo.angle = dart_angle

            angleDiffMul = int((dart_angle) / 18.0)

            print vx, vy, dart_angle

            #starting from the 20 points
            if angleDiffMul == 19:
                dartInfo.base = 20
            elif angleDiffMul == 0:
                dartInfo.base = 5
            elif angleDiffMul == 1:
                dartInfo.base = 12
            elif angleDiffMul == 2:
                dartInfo.base = 9
            elif angleDiffMul == 3:
                dartInfo.base = 14
            elif angleDiffMul == 4:
                dartInfo.base = 11
            elif angleDiffMul == 5:
                dartInfo.base = 8
            elif angleDiffMul == 6:
                dartInfo.base = 16
            elif angleDiffMul == 7:
                dartInfo.base = 7
            elif angleDiffMul == 8:
                dartInfo.base = 19
            elif angleDiffMul == 9:
                dartInfo.base = 3
            elif angleDiffMul == 10:
                dartInfo.base = 17
            elif angleDiffMul == 11:
                dartInfo.base = 2
            elif angleDiffMul == 12:
                dartInfo.base = 15
            elif angleDiffMul == 13:
                dartInfo.base = 10
            elif angleDiffMul == 14:
                dartInfo.base = 6
            elif angleDiffMul == 15:
                dartInfo.base = 13
            elif angleDiffMul == 16:
                dartInfo.base = 4
            elif angleDiffMul == 17:
                dartInfo.base = 18
            elif angleDiffMul == 18:
                dartInfo.base = 1
            else:
                #something went wrong
                dartInfo.base = -300

            #Calculating multiplier (and special cases for Bull's Eye):
            for i in range(0, len(ring_radius)):
                #Find the ring that encloses the dart
                if dartInfo.magnitude <= ring_radius[i]:
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
            if dartInfo.magnitude > ring_radius[5]:
                dartInfo.base = 0
                dartInfo.multiplier = 0

            return dartInfo


    #system not calibrated
    except AttributeError as err1:
        print err1
        dartInfo = dartThrow()
        return dartInfo

    except NameError as err2:
        #not calibrated error
        print err2
        dartInfo = dartThrow()
        return dartInfo

        #if breaker == 3:
        #    break

def getDart():
    global finalScore
    global transformation_matrix

    debug_img = drawBoard()

    finalScore = 0
    count = 0
    breaker = 0
    success = 1
    ## threshold important -> make accessible
    x = 3000
    # Read first image twice (issue somewhere) to start loop:
    t = cv2.cvtColor(vidcap.read()[1], cv2.COLOR_RGB2GRAY)
    # wait for camera
    time.sleep(0.1)
    t = cv2.cvtColor(vidcap.read()[1], cv2.COLOR_RGB2GRAY)

    while success:
        time.sleep(0.1)
        success,image = vidcap.read()
        t_plus = cv2.cvtColor(vidcap.read()[1], cv2.COLOR_RGB2GRAY)
        dimg = cv2.absdiff(t, t_plus)
        # cv2.imshow(winName, edges(t_minus, t, t_plus))
        blur = cv2.GaussianBlur(dimg,(5,5),0)
        blur = cv2.bilateralFilter(blur,9,75,75)
        ret, thresh = cv2.threshold(blur, 60, 255, 0)
        if cv2.countNonZero(thresh) > x and cv2.countNonZero(thresh) < 15000: ## threshold important -> make accessible

            if from_video:
                t_plus = cv2.cvtColor(vidcap.read()[1], cv2.COLOR_RGB2GRAY)
                t_plus = cv2.cvtColor(vidcap.read()[1], cv2.COLOR_RGB2GRAY)
                t_plus = cv2.cvtColor(vidcap.read()[1], cv2.COLOR_RGB2GRAY)
                t_plus = cv2.cvtColor(vidcap.read()[1], cv2.COLOR_RGB2GRAY)
                t_plus = cv2.cvtColor(vidcap.read()[1], cv2.COLOR_RGB2GRAY)
            else:
                time.sleep(0.2)

            t_plus = cv2.cvtColor(vidcap.read()[1], cv2.COLOR_RGB2GRAY)

            cv2.imshow(winName, t_plus)
            dimg = cv2.absdiff(t, t_plus)

            ## kernel size important -> make accessible
            # filter noise from image distortions
            kernel = np.ones((8, 8), np.float32) / 40
            blur = cv2.filter2D(dimg, -1, kernel)
            #blur = cv2.GaussianBlur(dimg,(3,3),1)
            #blur = cv2.bilateralFilter(blur,3,10,70) # 10,70

            # remove image distortions
            #kernel = np.ones((1, 1), np.uint8)
            #blur = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
            #kernel = np.ones((1, 1), np.uint8)
            # blur = cv2.dilate(blur, kernel, iterations=2)
            #blur = cv2.erode(blur, kernel, iterations=1)

            # number of features to track is a distinctive feature
            #edges = cv2.goodFeaturesToTrack(blur,200,0.01,0,mask=None, blockSize=2, useHarrisDetector=1, k=0.001)
            ## FeaturesToTrack important -> make accessible
            edges = cv2.goodFeaturesToTrack(blur,640,0.0008,3,mask=None, blockSize=3, useHarrisDetector=1, k=0.06) # k=0.08
            corners = np.int0(edges)
            testimg = blur.copy()
            t_plus_copy = t_plus.copy()

            # filter corners
            cornerdata = []
            tt = 0
            mean_corners = np.mean(corners, axis=0)
            for i in corners:
                xl, yl = i.ravel()
                ## threshold important -> make accessible
                # filter noise to only get dart arrow
                if abs(mean_corners[0][0] - xl) > 180:
                    cornerdata.append(tt)
                if abs(mean_corners[0][1] - yl) > 120:
                    cornerdata.append(tt)
                tt += 1

            corners_new = np.delete(corners, [cornerdata], axis=0)  # delete corners to form new array

            # find left and rightmost corners
            rows,cols = dimg.shape[:2]
            [vx,vy,x,y] = cv2.fitLine(corners_new,cv.CV_DIST_HUBER, 0,0.1,0.1)
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)

            cornerdata = []
            tt = 0
            for i in corners_new:
                xl,yl = i.ravel()
                # check distance to fitted line, only draw corners within certain range
                distance = dist(0,lefty, cols-1,righty, xl,yl)
                if distance < 40:        ## threshold important -> make accessible
                    cv2.circle(testimg,(xl,yl),3,255,-1)
                else:  # only save corners within certain range
                    cornerdata.append(tt)
                tt += 1

            corners_final = np.delete(corners_new, [cornerdata], axis=0)  # delete corners to form new array

            ret, thresh = cv2.threshold(blur, 60, 255, 0)
            ## threshold important -> make accessible
            if cv2.countNonZero(thresh) > 15000:
                continue

            x,y,w,h = cv2.boundingRect(corners_final)

            cv2.rectangle(t_plus_copy,(x,y),(x+w,y+h),(0,255,0),1)

            breaker += 1

            # find maximum x distance to dart tip, if camera is mounted on top

            maxloc = np.argmax(corners_final, axis=0)  # check max pos!!!, write image with circle??!!!

            locationofdart = corners_final[maxloc]

            try:
                # check if dart location has neighbouring corners (if not -> continue)
                cornerdata = []
                tt = 0
                for i in corners_final:
                    xl, yl = i.ravel()
                    distance = abs(locationofdart.item(0) - xl) + abs(locationofdart.item(1) - yl)
                    if distance < 40: ## threshold important -> make accessible
                        tt += 1
                    else:
                        cornerdata.append(tt)

                if tt < 3:
                    corners_temp = cornerdata
                    maxloc = np.argmax(corners_temp, axis=0)
                    locationofdart = corners_temp[maxloc]
                    print "### used different location due to noise!"

                cv2.circle(t_plus_copy, (locationofdart.item(0),locationofdart.item(1)), 10,(0, 0, 0),2, 8)
                cv2.circle(t_plus_copy, (locationofdart.item(0), locationofdart.item(1)), 2, (0, 0, 0), 2, 8)

                # check for the location of the dart with the calibration

                dartloc = DartLocation(locationofdart.item(0), locationofdart.item(1))
                dartInfo = DartRegion(dartloc) #cal_image

            except:
                print "Something went wrong in finding the darts location!"
                continue

            # check for the location of the dart with the calibration

            print dartInfo.base, dartInfo.multiplier

            if breaker == 1:
                cv2.imwrite("frame2.jpg", testimg)  # save dart1 frame
            elif breaker == 2:
                cv2.imwrite("frame3.jpg", testimg)  # save dart2 frame
            elif breaker == 3:
                cv2.imwrite("frame4.jpg", testimg)  # save dart3 frame

            # save new diff img for next dart
            t = t_plus

            finalScore += (dartInfo.base * dartInfo.multiplier)

            if DEBUG:
                loc_x = dartloc[0] #400 + dartInfo.magnitude * math.tan(dartInfo.angle * math.pi/180)
                loc_y = dartloc[1] #400 + dartInfo.magnitude * math.tan(dartInfo.angle * math.pi/180)
                cv2.circle(debug_img, (int(loc_x), int(loc_y)), 2, (0, 255, 0), 2, 8)
                cv2.circle(debug_img, (int(loc_x), int(loc_y)), 6, (0, 255, 0), 1, 8)
                string = "" + str(dartInfo.base) + "x" + str(dartInfo.multiplier)
                # add text (before clear with rectangle)
                cv2.rectangle(debug_img, (600, 700), (800, 800), (0, 0, 0), -1)
                cv2.putText(debug_img, string, (600, 750), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, 8)
                cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
                cv2.namedWindow("raw", cv2.WINDOW_NORMAL)
                cv2.namedWindow("test", cv2.WINDOW_NORMAL)
                cv2.imshow(winName, debug_img)
                cv2.imshow("raw", t_plus_copy)
                cv2.imshow("test", testimg)
            else:
                cv2.imshow(winName, testimg)

            #if breaker == 3:
            #    break

        # missed dart
        elif cv2.countNonZero(thresh) < 35000:
            continue

            # if player enters zone - break loop
        elif cv2.countNonZero(thresh) > 35000:
            break

        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyWindow(winName)
            break

        count += 1


dartInfo = dartThrow()


if __name__ == '__main__':
    print "Welcome to darts!"
    getDart()
    #getTransformation()
