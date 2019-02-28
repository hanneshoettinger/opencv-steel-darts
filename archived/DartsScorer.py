__author__ = "Hannes Hoettinger"

from Tkinter import *
from Calibration import *
from GetDart import *
from thread import *

import cv2
import time

#cam = cv2.VideoCapture(0)
finalScore = 0
curr_player = 1
scoreplayer1 = 501
scoreplayer2 = 501

#cam = cv2.VideoCapture(2)
#cam.set(cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
#cam.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
cam = VideoStream(src=2).start()

points = []

class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        master.minsize(width=800, height=600)
        self.pack()

def GameOn():
    global cal_image
    success,cal_image = cam.read()
    cv2.imwrite("frame1.jpg", cal_image)     # save calibration frame
    scoreplayer1 = 501
    scoreplayer2 = 501
    global curr_player
    curr_player = 1

    e1.configure(bg='light green')

    global finalScore
    finalScore = 0
    e1.delete(0,'end')
    e2.delete(0,'end')
    e1.insert(10,scoreplayer1)
    e2.insert(10,scoreplayer2)
    finalentry.delete(0, 'end')
    dart1entry.delete(0, 'end')
    dart2entry.delete(0, 'end')
    dart3entry.delete(0, 'end')
     # start getDart thread!!!
    t = Thread(target=start_imag_proc)
    t.start()

def printin(event):
    test = str(eval(e1.get()))
    print test


# correct dart score with binding -> press return to change
def dartcorr(event):
    # check if empty, on error write 0 to value of dart
    try:
        dart1 = int(eval(dart1entry.get()))
    except:
        dart1 = 0
    try:
        dart2 = int(eval(dart2entry.get()))
    except:
        dart2 = 0
    try:
        dart3 = int(eval(dart3entry.get()))
    except:
        dart3 = 0

    dartscore = dart1 + dart2 + dart3

    # check which player
    if curr_player == 1:
        new_score = scoreplayer1 - dartscore
        e1.delete(0,'end')
        e1.insert(10, new_score)
    else:
        new_score = scoreplayer2 - dartscore
        e2.delete(0,'end')
        e2.insert(10, new_score)
    finalentry.delete(0,'end')
    finalentry.insert(10,dartscore)

# start motion processing in different thread, init scores
def dartscores():
    global scoreplayer1
    global scoreplayer2
    global curr_player
    if curr_player == 1:
        curr_player = 2
        e2.configure(bg='light green')
        e1.configure(bg='white')
        score = int(e2.get())
    else:
        curr_player = 1
        e1.configure(bg='light green')
        e2.configure(bg='white')
        score = int(e1.get())

    # clear dartscores
    finalentry.delete(0, 'end')
    dart1entry.delete(0, 'end')
    dart2entry.delete(0, 'end')
    dart3entry.delete(0, 'end')
    scoreplayer1 = int(e1.get())
    scoreplayer2 = int(e2.get())
    # start getDart thread!!!
    t = Thread(target=start_imag_proc)
    t.start()

def start_imag_proc():
    global finalScore
    global curr_player

    finalScore = 0
    count = 0
    breaker = 0
    success = 1
    ## threshold important -> make accessible
    x = 1000

    #check which player
    if curr_player == 1:
        print e1.get()
        score = int(e1.get())
    else:
        print e2.get()
        score = int(e2.get())

    # save score if score is below 1...
    old_score = score

    # Read first image twice (issue somewhere) to start loop:
    success, image = cam.read()
    t = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # wait for camera
    time.sleep(0.1)
    success, image = cam.read()
    t = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    while success:
        # wait for camera
        success,image = cam.read()
        t_plus = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dimg = cv2.absdiff(t, t_plus)
        time.sleep(0.1)
        # cv2.imshow(winName, edges(t_minus, t, t_plus))
        blur = cv2.GaussianBlur(dimg,(5,5),0)
        blur = cv2.bilateralFilter(blur,9,75,75)
        ret, thresh = cv2.threshold(blur, 60, 255, 0)
        if cv2.countNonZero(thresh) > x and cv2.countNonZero(thresh) < 8000: ## threshold important -> make accessible
            # wait for camera vibrations
            time.sleep(0.2)
            t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
            dimg = cv2.absdiff(t, t_plus)

            ## kernel size important -> make accessible
            # filter noise from image distortions
            kernel = np.ones((8, 8), np.float32) / 40
            blur = cv2.filter2D(dimg, -1, kernel)

            # dilate and erode?

            # number of features to track is a distinctive feature
            # edges = cv2.goodFeaturesToTrack(blur,200,0.01,0,mask=None, blockSize=2, useHarrisDetector=1, k=0.001)
            ## FeaturesToTrack important -> make accessible
            edges = cv2.goodFeaturesToTrack(blur,640,0.0008,1,mask=None, blockSize=3, useHarrisDetector=1, k=0.06) # k=0.08
            corners = np.int0(edges)
            testimg = blur.copy()

            # dart outside?
            if corners.size < 40:
                print "### dart not detected"
                continue

            # filter corners
            cornerdata = []
            tt = 0
            mean_corners = np.mean(corners, axis=0)
            for i in corners:
                xl, yl = i.ravel()
                # filter noise to only get dart arrow
                ## threshold important -> make accessible
                if abs(mean_corners[0][0] - xl) > 280:
                    cornerdata.append(tt)
                if abs(mean_corners[0][1] - yl) > 220:
                    cornerdata.append(tt)
                tt += 1

            corners_new = np.delete(corners, [cornerdata], axis=0)  # delete corners to form new array

            # dart outside?
            if corners_new.size < 30:
                print "### dart not detected"
                continue

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
                 else:  # save corners out of range to delete afterwards
                    cornerdata.append(tt)
                 tt += 1

            corners_final = np.delete(corners_new, [cornerdata], axis=0)  # delete corners to form new array

            t_plus_new = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
            dimg_new = cv2.absdiff(t_plus, t_plus_new)
            # filter noise from image distortions
            kernel = np.ones((8, 8), np.float32) / 40
            blur_new = cv2.filter2D(dimg_new, -1, kernel)

            ret, thresh_new = cv2.threshold(blur_new, 60, 255, 0)
            ## threshold important -> make accessible
            ### check for bouncer????????
            if cv2.countNonZero(thresh_new) > 400:
                continue

            x,y,w,h = cv2.boundingRect(corners_final)

            cv2.rectangle(testimg,(x,y),(x+w,y+h),(255,255,255),1)

            breaker += 1
            ###########################
            # find maximum x distance to dart tip, if camera is mounted on top

            maxloc = np.argmax(corners_final, axis=0)  # check max pos!!!, write image with circle??!!!
            ###########################
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

                cv2.circle(testimg, (locationofdart.item(0),locationofdart.item(1)), 10,(255, 255, 255),2, 8)
                cv2.circle(testimg, (locationofdart.item(0), locationofdart.item(1)), 2, (0, 255, 0), 2, 8)

                # check for the location of the dart with the calibration

                dartloc = DartLocation(locationofdart.item(0), locationofdart.item(1))
                dartInfo = DartRegion(dartloc) #cal_image

            except:
                print "Something went wrong in finding the darts location!"
                breaker -= 1
                continue

            print dartInfo.base, dartInfo.multiplier

            if breaker == 1:
                dart1entry.insert(10,str(dartInfo.base * dartInfo.multiplier))
                dart = int(dart1entry.get())
                cv2.imwrite("frame2.jpg", testimg)     # save dart1 frame
            elif breaker == 2:
                dart2entry.insert(10,str(dartInfo.base * dartInfo.multiplier))
                dart = int(dart2entry.get())
                cv2.imwrite("frame3.jpg", testimg)     # save dart2 frame
            elif breaker == 3:
                dart3entry.insert(10,str(dartInfo.base * dartInfo.multiplier))
                dart = int(dart3entry.get())
                cv2.imwrite("frame4.jpg", testimg)     # save dart3 frame

            score -= dart

            if score == 0 and dartInfo.multiplier == 2:
                score = 0
                breaker = 3
            elif score <= 1:
                score = old_score
                breaker = 3

            # save new diff img for next dart
            t = t_plus

            if curr_player == 1:
                e1.delete(0,'end')
                e1.insert(10,score)
            else:
                e2.delete(0,'end')
                e2.insert(10,score)

            finalScore += (dartInfo.base * dartInfo.multiplier)

            if breaker == 3:
                break

            #cv2.imshow(winName, tnow)

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

    finalentry.insert(10,finalScore)

    print finalScore

root = Tk()

# Background Image
back_gnd = Canvas(root)
back_gnd.pack(expand=True, fill='both')

back_gnd_image = PhotoImage(file="C:\Users\hanne\OneDrive\Projekte\GitHub\darts\Dartboard.gif")
back_gnd.create_image(0, 0, anchor='nw', image=back_gnd_image)

# Create Buttons

ImagCalib = Button(None, text="Calibrate", fg="black", font = "Helvetica 26 bold", command=calibrate)
back_gnd.create_window(20,200, window=ImagCalib, anchor='nw')

newgame = Button(None, text="Game On!", fg="black", font = "Helvetica 26 bold", command=GameOn)
back_gnd.create_window(20,20, window=newgame, anchor='nw')

QUIT = Button(None, text="QUIT", fg="black", font = "Helvetica 26 bold", command=root.quit)
back_gnd.create_window(20,300, window=QUIT, anchor='nw')

nextplayer = Button(None, text="Next Player", fg="black", font = "Helvetica 26 bold", command=dartscores)
back_gnd.create_window(460,400, window=nextplayer, anchor='nw')


# player labels and entry for total score

#player1 = Label(None, text="Player 1", font = "Helvetica 32 bold")
player1 = Entry(root, font = "Helvetica 32 bold",width=7)
back_gnd.create_window(250,20, window=player1, anchor='nw')
player1.insert(10,"Player 1")

#player2 = Label(None, text="Player 2", font = "Helvetica 32 bold")
player2 = Entry(root, font = "Helvetica 32 bold",width=7)
back_gnd.create_window(400,20, window=player2, anchor='nw')
player2.insert(10,"Player 2")

e1 = Entry(root,font = "Helvetica 44 bold",width=4)
e1.bind("<Return>", printin)
back_gnd.create_window(250,80, window=e1, anchor='nw')
e2 = Entry(root,font = "Helvetica 44 bold",width=4)
back_gnd.create_window(400,80, window=e2, anchor='nw')
e1.insert(10,"501")
e2.insert(10,"501")

#e1.pack()

# dart throw scores
dart1label = Label(None, text="1.: ", font = "Helvetica 20 bold")
back_gnd.create_window(300,160, window=dart1label, anchor='nw')

dart1entry = Entry(root,font = "Helvetica 20 bold",width=3)
dart1entry.bind("<Return>", dartcorr)
back_gnd.create_window(350,160, window=dart1entry, anchor='nw')

dart2label = Label(None, text="2.: ", font = "Helvetica 20 bold")
back_gnd.create_window(300,210, window=dart2label, anchor='nw')

dart2entry = Entry(root,font = "Helvetica 20 bold",width=3)
dart2entry.bind("<Return>", dartcorr)
back_gnd.create_window(350,210, window=dart2entry, anchor='nw')

dart3label = Label(None, text="3.: ", font = "Helvetica 20 bold")
back_gnd.create_window(300,260, window=dart3label, anchor='nw')

dart3entry = Entry(root,font = "Helvetica 20 bold",width=3)
dart3entry.bind("<Return>", dartcorr)
back_gnd.create_window(350,260, window=dart3entry, anchor='nw')

finallabel = Label(None, text=" = ", font = "Helvetica 20 bold")
back_gnd.create_window(300,310, window=finallabel, anchor='nw')

finalentry = Entry(root,font = "Helvetica 20 bold",width=3)
back_gnd.create_window(350,310, window=finalentry, anchor='nw')

app = Application(master=root)
app.mainloop()
cam.stop()
root.destroy()
