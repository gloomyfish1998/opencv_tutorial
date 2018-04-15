####################################################
#   作者:zhigang,
#   微博:https://weibo.com/u/3181256271
####################################################
import cv2 as cv
import numpy as np


class LineDetector:
    def __init__(self):
        self.lines = []

    def find_lines(self, frame):
        h, w, ch = frame.shape
        # 二值化图像
        print("start to detect lines...\n")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        cv.imshow("binary image", binary)
        cv.imwrite("D:/binary.png", binary)

        dist = cv.distanceTransform(binary, cv.DIST_L1, cv.DIST_MASK_PRECISE)
        cv.imshow("distance", dist/15)
        dist = dist / 15
        result = np.zeros((h, w), dtype=np.uint8)
        ypts = []
        for row in range(h):
            cx = 0
            cy = 0
            max_d = 0
            for col in range(w):
                d = dist[row][col]
                if d > max_d:
                    max_d = d
                    cx = col
                    cy = row
            result[cy][cx] = 255
            ypts.append([cx, cy])

        xpts = []
        for col in range(w):
            cx = 0
            cy = 0
            max_d = 0
            for row in range(h):
                d = dist[row][col]
                if d > max_d:
                    max_d = d
                    cx = col
                    cy = row
            result[cy][cx] = 255
            xpts.append([cx, cy])

        cv.imshow("lines", result)
        cv.imwrite("D:/skeleton.png", result)

        frame = self.line_fitness(ypts, image=frame)
        frame = self.line_fitness(xpts, image=frame, color=(255, 0, 0))

        cv.imshow("fit-lines", frame)
        cv.imwrite("D:/fitlines.png", frame)
        return self.lines

    def line_fitness(self, pts, image, color=(0, 0, 255)):
        h, w, ch = image.shape
        [vx, vy, x, y] = cv.fitLine(np.array(pts), cv.DIST_L1, 0, 0.01, 0.01)
        y1 = int((-x * vy / vx) + y)
        y2 = int(((w - x) * vy / vx) + y)
        cv.line(image, (w - 1, y2), (0, y1), color, 2)
        return image


if __name__ == "__main__":
    src = cv.imread("D:/javaopencv/two_lines.jpg")
    ld = LineDetector()
    lines = ld.find_lines(src)
    cv.waitKey(0)
    cv.destroyAllWindows()
