__author__ = "Hannes Hoettinger"

from Tkinter import *
from Calibration_1 import *
from DartsRecognition import *
from thread import *
from Classes import *

import cv2
import time

#cam = cv2.VideoCapture(0)
finalScore = 0
curr_player = 1
scoreplayer1 = 501
scoreplayer2 = 501

points = []

cam_R = VideoStream(src=2).start()
cam_L = VideoStream(src=3).start()

calData_R = CalibrationData()
calData_L = CalibrationData()


class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        master.minsize(width=800, height=600)
        self.pack()


def GameOn():
    global calData_R
    global calData_L
    global cal_image
    success,cal_image = cam_R.read()
    cv2.imwrite("frame1.jpg", cal_image)     # save calibration frame
    scoreplayer1 = 501
    scoreplayer2 = 501
    global curr_player
    curr_player = 1

    GUI.e1.configure(bg='light green')

    global finalScore
    finalScore = 0
    GUI.e1.delete(0,'end')
    GUI.e2.delete(0,'end')
    GUI.e1.insert(10,scoreplayer1)
    GUI.e2.insert(10,scoreplayer2)
    GUI.finalentry.delete(0, 'end')
    GUI.dart1entry.delete(0, 'end')
    GUI.dart2entry.delete(0, 'end')
    GUI.dart3entry.delete(0, 'end')

    player.player = curr_player
    player.score = 501
    # start getDart thread!!!
    t = Thread(target=getDarts, args=(cam_R, cam_L, calData_R, calData_L, player, GUI))
    t.start()


def printin(event):
    test = str(eval(GUI.e1.get()))
    print test


def calibrateGUI():
    global calData_R
    global calData_L
    calData_R, calData_L = calibrate(cam_R, cam_L)


# correct dart score with binding -> press return to change
def dartcorr(event):
    # check if empty, on error write 0 to value of dart
    try:
        dart1 = int(eval(GUI.dart1entry.get()))
    except:
        dart1 = 0
    try:
        dart2 = int(eval(GUI.dart2entry.get()))
    except:
        dart2 = 0
    try:
        dart3 = int(eval(GUI.dart3entry.get()))
    except:
        dart3 = 0

    dartscore = dart1 + dart2 + dart3

    # check which player
    if curr_player == 1:
        new_score = scoreplayer1 - dartscore
        GUI.e1.delete(0,'end')
        GUI.e1.insert(10, new_score)
    else:
        new_score = scoreplayer2 - dartscore
        GUI.e2.delete(0,'end')
        GUI.e2.insert(10, new_score)
    GUI.finalentry.delete(0,'end')
    GUI.finalentry.insert(10,dartscore)

# start motion processing in different thread, init scores
def dartscores():
    global scoreplayer1
    global scoreplayer2
    global calData_R
    global calData_L
    global curr_player
    if curr_player == 1:
        curr_player = 2
        GUI.e2.configure(bg='light green')
        GUI.e1.configure(bg='white')
        score = int(GUI.e2.get())
        player.player = curr_player
        player.score = score
    else:
        curr_player = 1
        GUI.e1.configure(bg='light green')
        GUI.e2.configure(bg='white')
        score = int(GUI.e1.get())
        player.player = curr_player
        player.score = score

    # clear dartscores
    GUI.finalentry.delete(0, 'end')
    GUI.dart1entry.delete(0, 'end')
    GUI.dart2entry.delete(0, 'end')
    GUI.dart3entry.delete(0, 'end')
    scoreplayer1 = int(GUI.e1.get())
    scoreplayer2 = int(GUI.e2.get())

    # start getDart thread!!!
    t = Thread(target=getDarts, args=(cam_R, cam_L, calData_R, calData_L, player, GUI))
    t.start()


root = Tk()

GUI = GUIDef()

player = Player()

# Background Image
back_gnd = Canvas(root)
back_gnd.pack(expand=True, fill='both')

back_gnd_image = PhotoImage(file="Dartboard.gif")
back_gnd.create_image(0, 0, anchor='nw', image=back_gnd_image)

# Create Buttons
ImagCalib = Button(None, text="Calibrate", fg="black", font = "Helvetica 26 bold", command=calibrateGUI)
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

GUI.e1 = Entry(root,font = "Helvetica 44 bold",width=4)
GUI.e1.bind("<Return>", printin)
back_gnd.create_window(250,80, window=GUI.e1, anchor='nw')
GUI.e2 = Entry(root,font = "Helvetica 44 bold",width=4)
back_gnd.create_window(400,80, window=GUI.e2, anchor='nw')
GUI.e1.insert(10,"501")
GUI.e2.insert(10,"501")

#e1.pack()

# dart throw scores
dart1label = Label(None, text="1.: ", font = "Helvetica 20 bold")
back_gnd.create_window(300,160, window=dart1label, anchor='nw')

GUI.dart1entry = Entry(root,font = "Helvetica 20 bold",width=3)
GUI.dart1entry.bind("<Return>", dartcorr)
back_gnd.create_window(350,160, window=GUI.dart1entry, anchor='nw')

dart2label = Label(None, text="2.: ", font = "Helvetica 20 bold")
back_gnd.create_window(300,210, window=dart2label, anchor='nw')

GUI.dart2entry = Entry(root,font = "Helvetica 20 bold",width=3)
GUI.dart2entry.bind("<Return>", dartcorr)
back_gnd.create_window(350,210, window=GUI.dart2entry, anchor='nw')

dart3label = Label(None, text="3.: ", font = "Helvetica 20 bold")
back_gnd.create_window(300,260, window=dart3label, anchor='nw')

GUI.dart3entry = Entry(root,font = "Helvetica 20 bold",width=3)
GUI.dart3entry.bind("<Return>", dartcorr)
back_gnd.create_window(350,260, window=GUI.dart3entry, anchor='nw')

finallabel = Label(None, text=" = ", font = "Helvetica 20 bold")
back_gnd.create_window(300,310, window=finallabel, anchor='nw')

GUI.finalentry = Entry(root,font = "Helvetica 20 bold",width=3)
back_gnd.create_window(350,310, window=GUI.finalentry, anchor='nw')

app = Application(master=root)
app.mainloop()
root.destroy()
