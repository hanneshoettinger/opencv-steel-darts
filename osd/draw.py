import math
import cv2


"""
This parameter will be used to draw the board and can be changed here if needed
"""
board_size = (400, 400) # size of drawing px X px
line_color = (255,255,255) # white
line_thickness = 1 # 1 Pixel


class Draw:
    """
    This Class is responsible for drawing a dart board picture.
    """
    def __init__(self):
        """
        Radius is pi times 2
        Sectorangle will be radius divided by sector number
        There are 20 sectors on a dartboard
        """
        self.sectorangle = 2 * math.pi / 20

    def drawBoard(self, img, calData):
        """
        This method will draw the actual board
        """

        cv2.circle(img, board_size, calData.ring_radius[0], line_color, line_thickness) # outside double
        cv2.circle(img, board_size, calData.ring_radius[1], line_color, line_thickness) # inside double
        cv2.circle(img, board_size, calData.ring_radius[2], line_color, line_thickness) # outside triple
        cv2.circle(img, board_size, calData.ring_radius[3], line_color, line_thickness) # inside triple
        cv2.circle(img, board_size, calData.ring_radius[4], line_color, line_thickness) # 25
        cv2.circle(img, board_size, calData.ring_radius[5], line_color, line_thickness) # Bulls eye

        # TODO
        # Add sector lines here