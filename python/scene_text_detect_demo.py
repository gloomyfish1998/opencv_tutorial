import cv2 as cv
import numpy as np


def cascade_classfier_text_detect():
    img = cv.imread("D:/images/text_01.jpg")
    vis = img.copy()

    # Extract channels to be processed individually
    channels = cv.text.computeNMChannels(img)
    cn = len(channels)-1
    for c in range(0,cn):
      channels.append((255-channels[c]))

    # Apply the default cascade classifier to each independent channel (could be done in parallel)
    print("Extracting Class Specific Extremal Regions from "+str(len(channels))+" channels ...")
    print("    (...) this may take a while (...)")
    for channel in channels:

      erc1 = cv.text.loadClassifierNM1('trained_classifierNM1.xml')
      er1 = cv.text.createERFilterNM1(erc1,16,0.00015,0.13,0.2,True,0.1)

      erc2 = cv.text.loadClassifierNM2('trained_classifierNM2.xml')
      er2 = cv.text.createERFilterNM2(erc2,0.5)

      regions = cv.text.detectRegions(channel,er1,er2)

      rects = cv.text.erGrouping(img,channel,[r.tolist() for r in regions])

      #Visualization
      for r in range(0,np.shape(rects)[0]):
        rect = rects[r]
        cv.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 0, 0), 2)
        cv.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 255), 1)


    #Visualization
    cv.imshow("Text detection result", vis)
    cv.imwrite("D:/test_detection_demo_02.png", vis)
    cv.waitKey(0)


def cnn_text_detect():
    image = cv.imread("D:/images/text_01.jpg")
    cv.imshow("input", image)
    result = image.copy()
    detector = cv.text.TextDetectorCNN_create("textbox.prototxt", "TextBoxes_icdar13.caffemodel")
    boxes, scores = detector.detect(image);
    threshold = 0.5
    for r in range(np.shape(boxes)[0]):
        if scores[r] > threshold:
            rect = boxes[r]
            cv.rectangle(result, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)

    cv.imshow("Text detection result", result)
    cv.waitKey()

    cv.waitKey(0)
    cv.imwrite("D:/text_cnn_01.png", result)
    cv.destroyAllWindows()


if __name__ == '__main__':
    cnn_text_detect()