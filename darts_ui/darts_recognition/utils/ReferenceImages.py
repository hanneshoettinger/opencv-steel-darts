from os import path

import cv2

from darts_ui.darts_recognition.utils.VideoCapture import VideoStream

current_path = path.abspath(path.dirname(__file__))
img_path = path.abspath(path.join(current_path, '../test_images'))
img1_path = path.abspath(path.join(img_path, 'cam1.jpg'))
img2_path = path.abspath(path.join(img_path, 'cam2.jpg'))


def loadReferenceImages():
    img1_exists = path.exists(img1_path)
    img2_exists = path.exists(img2_path)

    if not img1_exists or not img2_exists:
        print('Some reference images seems not to exist.')
        return None, None

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    return img1, img2


def takeReferenceImages():
    try:
        _, img1 = cam1.read()
        _, img2 = cam2.read()
    except:
        print('Could not initiate cams')
        return

    cv2.imwrite(img1_path, img1)
    cv2.imwrite(img2_path, img2)
    cam1.stop()
    cam2.stop()


if __name__ == '__main__':
    global cam1
    global cam2

    cam1 = VideoStream(src=0).start()
    cam2 = VideoStream(src=1).start()

    takeReferenceImages()
