__author__ = "Hannes Hoettinger"

# import numpy as np
# import cv2
# import math
# import pickle
from Classes import *

DEBUG = True


# distance point to line
def dist(x1,y1, x2,y2, x3,y3):  # x3,y3 is the point
    px = x2-x1
    py = y2-y1

    something = px*px + py*py

    u = ((x3 - x1) * px + (y3 - y1) * py) / float(something)

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


def intersect_line_circle(center, radius, p1, p2):
    ba_x = p2[0] - p1[0]
    ba_y = p2[1] - p1[1]
    ca_x = center[0] - p1[0]
    ca_y = center[1] - p1[1]

    a = ba_x * ba_x + ba_y * ba_y
    b_by2 = ba_x * ca_x + ba_y * ca_y
    c = ca_x * ca_x + ca_y * ca_y - radius * radius

    p_by2 = b_by2 / a
    q = c / a

    disc = p_by2 * p_by2 - q
    if disc < 0:
        return False, None, False, None

    tmp_sqrt = math.sqrt(disc)
    ab_scaling_factor1 = -p_by2 + tmp_sqrt
    ab_scaling_factor2 = -p_by2 - tmp_sqrt

    pint1 = p1[0] - ba_x * ab_scaling_factor1, p1[1] - ba_y * ab_scaling_factor1
    if disc == 0:
        return True, pint1, False, None

    pint2 = p1[0] - ba_x * ab_scaling_factor2, p1[1] - ba_y * ab_scaling_factor2
    return True, pint1, True, pint2


# line intersection
def intersect_lines(pt1, pt2, pt_a, pt_b):
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    det_tolerance = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = pt_a
    x_b, y_b = pt_b
    dx = x_b - x
    dy = y_b - y

    det = (-dx1 * dy + dy1 * dx)

    if math.fabs(det) < det_tolerance:
        return 0, 0

    # now, the determinant should be OK
    de_tinv = 1.0 / det

    # find the scalar amount along the "self" segment
    r = de_tinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = de_tinv * (-dy1 * (x - x1) + dx1 * (y - y1))

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
