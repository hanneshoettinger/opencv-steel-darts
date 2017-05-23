__author__ = "Hannes Hoettinger"

import math
import cv2
import cv2.cv as cv

DEBUG = True


class Player:
    def __init__(self):
        self.player = -1
        self.score = -1
        self.darts = -1


class GUIDef:
    def __init__(self):
        self.e1 = []
        self.e2 = []
        self.dart1entry = []
        self.dart2entry = []
        self.dart3entry = []
        self.finalentry = []


class DartDef:
    def __init__(self):
        self.base = -1
        self.multiplier = -1
        self.magnitude = -1
        self.angle = -1
        self.corners = -1


class EllipseDef:
    def __init__(self):
        self.a = -1
        self.b = -1
        self.x = -1
        self.y = -1
        self.angle = -1


#For file IO
class CalibrationData:
    def __init__(self):
        #for perspective transform
        self.top = []
        self.bottom = []
        self.left = []
        self.right = []
        self.points = []
        #radii of the rings, there are 6 in total
        self.ring_radius = [14, 32, 194, 214, 320, 340]
        self.center_dartboard = (400, 400)
        self.sectorangle = 2 * math.pi / 20
        self.dstpoints = []
        self.transformation_matrix = []

