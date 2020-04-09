import cv2
import numpy
import os
import string


# This software is designed to generate training data to be used in a image detection algorithm. It collects the values
# of the parameters of interest of specific objects, in order to draw decision boundaries.

def segment(image):
    grayImg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, threshold = cv2.threshold(grayImg, 120, 255, cv2.THRESH_BINARY)  # 190, 255
    k = 5  # int(image.shape[0] / 100)  # 100 for first photos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    open_img = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    close_img = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, kernel)
    return close_img


def representation(image, original):
    newImg = original.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    goodContours = []
    good = False
    for index in range(len(contours)):
        contour = contours[index]
        boundingBox = cv2.minAreaRect(contour)
        box = cv2.boxPoints(boundingBox)
        newbox = numpy.int0(box)
        areaRatio = (boundingBox[1][0] * boundingBox[1][1]) / (image.shape[0] * image.shape[1])
        if areaRatio < .9:
            if areaRatio > .1:  # cv2.contourArea(contour):
                goodContours.append(contour)
                cv2.drawContours(newImg, [newbox], 0, (0, 0, 255), 10)
                good = True
            elif areaRatio > .02:
                goodContours.append(contour)
                cv2.drawContours(newImg, [newbox], 0, (0, 255, 0), 10)
    if good and len(goodContours) <= 2:
        return goodContours, newImg
    return False, False


# This function saves the data to an external file
def logData(contours, path='database.db'):
    database = open(path, 'a')
    message = "\n"
    for contour in contours:
        message += str(cv2.minAreaRect(contour)) + "A"
    database.write(message)
    database.close()


def loadData(path='database.db'):
    database = open(path, 'r')
    m = True
    while m:
        m = database.readline()
        print(m)
    database.close()


# This function displays an image
def show(image, width=500, time=0):
    # The "width" is the value in pixels of the width a proportional resized version of the image to be displayed.
    # This is done so that the image can fit the screen, before it is displayed
    height = int(width * image.shape[0] / image.shape[1])
    resize = cv2.resize(image, (width, height))
    cv2.imshow('final image', resize)
    # "time" is the waiting period, defined in milliseconds.
    # If time is set to 0 it will wait until a key is pressed,
    # before closing the window with the displayed image
    cv2.waitKey(time)
    cv2.destroyAllWindows()


# turns a video into a sequence of good images
def vidtosec(path, time=300, width=500):
    video = cv2.VideoCapture(path)
    ret, frame = video.read()
    height = int(width * frame.shape[0] / frame.shape[1])
    count = 1
    while ret:
        ret, frame = video.read()
        segImg = segment(frame)
        rep, repImg = representation(segImg, frame)
        if rep:
            resize = cv2.resize(repImg, (width, height))
            cv2.imshow('final image', resize)
            if cv2.waitKey(time) & 0xFF == ord('s'):
                logData(rep)
                file = 'media_samples/video_sequence/vs_%d.jpg' % count
                saved = cv2.imwrite(file, frame)
                print(str(count) + " saved: ", saved)
                count += 1
    cv2.destroyAllWindows()


def makevideo():
    path = "media_samples/video_sequence/"
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter('output.avi', fourcc, 2.0, (640, 480))
    files = os.listdir(path + '.')
    for file in files:
        frame = cv2.imread(path + file, 1)
        out.write(frame)
    cap.release()
    out.release()


# Here is where the program starts
if __name__ == '__main__':
    vidtosec("media_samples/cocio.mp4", 0)
    # makevideo()
    """for i in range(4, 13):
        file = 'media_samples/video_sequence/vs_%d.jpg' % i
        rawImg = cv2.imread(file, 1)
        segImg = segment(rawImg)
        rep = representation(segImg, rawImg)
        if rep:
            logData(rep)
            loadData()
            show(rawImg)
"""
