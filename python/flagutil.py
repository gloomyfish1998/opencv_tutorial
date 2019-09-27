import cv2 as cv
import numpy as np
model_bin = "D:/projects/opencv_tutorial/data/models/face_detector/opencv_face_detector_uint8.pb";
config_text = "D:/projects/opencv_tutorial/data/models/face_detector/opencv_face_detector.pbtxt";


def get_face(image, detect=True):
    if detect is not True:
        return image

    # 定义人脸ROI
    x = 0
    y = 0
    width = 0
    height = 0

    # 加载网络
    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    h = image.shape[0]
    w = image.shape[1]

    # 人脸检测
    blobImage = cv.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False);
    net.setInput(blobImage)
    cvOut = net.forward()

    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    # 绘制检测矩形
    count = 0
    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.5:
            left = detection[3] * w
            top = detection[4] * h
            right = detection[5] * w
            bottom = detection[6] * h
            count += 1
            x = np.int32(left - 100)
            y = np.int32(top - 100)
            height = np.int32((bottom - top) + 200)
            width = np.int32((right - left) + 200)

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x+width > w:
        width = w - x
    if y+height > h:
        height = h - y

    if count == 1:
        return image[y:y + height, x:x + width, :]
    else:
        return image


def generate_new_profile(flag_icon, avatar):
    mask = cv.inRange(icon, (210, 210, 210), (225, 225, 225))
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, se)
    cv.imwrite("D:/mask1.png", mask)
    # mask with gaussian
    mask = cv.GaussianBlur(mask, (5, 5), 0)
    cv.imshow("mask", mask)
    cv.imwrite("D:/mask2.png", mask)

    # blend
    h, w = mask.shape[:2]
    avatar = cv.resize(avatar, (w, h), interpolation=cv.INTER_CUBIC)
    cv.imshow("profile", avatar)
    result = np.zeros_like(avatar)
    for row in range(h):
        for col in range(w):
            pv = mask[row, col]
            w1 = pv / 255.0
            w2 = 1.0 - w1
            b1, g1, r1 = avatar[row, col]
            b2, g2, r2 = icon[row, col]
            b1 = b1 * w1 + b2 * w2
            g1 = g1 * w1 + g2 * w2
            r1 = r1 * w1 + r2 * w2
            result[row, col] = [np.int32(b1), np.int32(g1), np.int32(r1)]
    return result


if __name__ == "__main__":
    icon = cv.imread("D:/images/flag.png")
    src = cv.imread("D:/images/zhigang.png")
    cv.imshow("input", icon)
    cv.imshow("profile", src)
    avatar = get_face(src, False)
    result = generate_new_profile(icon, avatar)
    cv.imshow("result", result)
    cv.imwrite("D:/result.png", result)
    cv.waitKey(0)
    cv.destroyAllWindows()