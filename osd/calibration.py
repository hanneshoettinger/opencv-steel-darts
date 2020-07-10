import os
import cPickle as pickle


class Calibration():
    """
    This Class will be used for Calibration
    It will be used per camera
    """
    def def __init__(self, completed=False, cam):
        self.completed = completed
        self.cam = cam

    def calibrate(self):
        while not self.completed:
            if os.path.isfile(f"calData_{self.cam.id}.pkl"):
                print("Load Calibration data here")
            else:
                print("Do Calibration here")
