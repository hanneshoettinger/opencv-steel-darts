import cv2
import numpy as numpy

from utils.ReferenceImages import loadReferenceImages, loadReferenceImagesWithDarts


def _findDarts(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    im_diff = cv2.absdiff(img1, img2)
    #im_thresh = cv2.threshold(im_diff, 80, 255, cv2.THRESH_BINARY)
    #im_erode = cv2.erode(im_thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    cv2.imshow('Diff', im_diff)
    #cv2.imshow('Thresh', im_thresh)
    #cv2.imshow('Erode', im_erode)

    while (1):
        kill = cv2.waitKey(1) & 0xFF
        if kill == 13 or kill == 27:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    img1, img2 = loadReferenceImages()
    img1_darts, img2_darts = loadReferenceImagesWithDarts()

    _findDarts(img1, img1_darts)
