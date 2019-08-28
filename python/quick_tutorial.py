import cv2 as cv
import numpy as np

# 加载与现实图像
src = cv.imread("D:/images/lena.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# 转换为灰度
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)
print(src.shape)
print(gray.shape)
cv.imwrite("D:/gray.png", gray)

# 创建空白图像
black = np.zeros_like(src)
cv.imshow("black", black)
cv.imwrite("D:/black.png", black)

# 调节亮度
black[:,:,:] = 50
lighter = cv.add(src, black)
darker = cv.subtract(src, black)
cv.imshow("lightness", lighter)
cv.imshow("darkness", darker)
cv.imwrite("D:/lightness.png", lighter)
cv.imwrite("D:/darkness.png", darker)

# 调节对比度
dst = cv.addWeighted(src, 1.2, black, 0.0, 0)
cv.imshow("contrast", dst)
cv.imwrite("D:/contrast.png", dst)

# scale
h, w, c = src.shape
dst = cv.resize(src, (h//2, w//2))
cv.imshow("resize-image", dst)

# 左右翻转
dst = cv.flip(src, 1)
cv.imshow("flip", dst)

# 上下翻转
dst = cv.flip(src, 0)
cv.imshow("flip0", dst)
cv.imwrite("D:/flip0.png", dst)

# rotate
M = cv.getRotationMatrix2D((w//2, h//2),45, 1)
dst = cv.warpAffine(src, M, (w, h))
cv.imshow("rotate", dst)
cv.imwrite("D:/rotate.png", dst)

# 色彩
# HSV
hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
cv.imshow("hsv", hsv)

# 色彩表 - 支持14种色彩变换
dst = cv.applyColorMap(src, cv.COLORMAP_AUTUMN)
cv.imshow("color table", dst)
cv.imwrite("D:/color_table.png", dst)

# blur
blur = cv.blur(src, (15, 15))
cv.imshow("blur", blur)
cv.imwrite("D:/blur.png", blur)

# gaussian blur
gblur = cv.GaussianBlur(src, (0, 0), 15)
cv.imshow("gaussian blur", gblur)
cv.imwrite("D:/gaussian.png", gblur)

# custom filter - blur
k = np.ones(shape=[5, 5], dtype=np.float32) / 25
dst = cv.filter2D(src, -1, k)
cv.imshow("custom blur", dst)
cv.imwrite("D:/custom_blur.png", dst)

# EPF
dst = cv.bilateralFilter(src, 0, 100, 10)
cv.imshow("bi-filter", dst)
cv.imwrite("D:/bi_blur.png", dst)

# gradient
dx = cv.Sobel(src, cv.CV_32F, 1, 0)
dy = cv.Sobel(src, cv.CV_32F, 0, 1)
dx = cv.convertScaleAbs(dx)
dy = cv.convertScaleAbs(dy)
cv.imshow("grad-x", dx)
cv.imshow("grad-y", dy)
cv.imwrite("D:/grad.png", dx)

# edge detect
edge = cv.Canny(src, 100, 300)
cv.imshow("edge", edge)
cv.imwrite("D:/edge.png", edge)

# 直方图均衡化
eh = cv.equalizeHist(gray)
cv.imshow("eh", eh)
cv.imwrite("D:/eh.png", eh)

# 角点检测
corners = cv.goodFeaturesToTrack(gray, 100, 0.05, 10)
# print(len(corners))
for pt in corners:
    # print(pt)
    b = np.random.random_integers(0, 256)
    g = np.random.random_integers(0, 256)
    r = np.random.random_integers(0, 256)
    x = np.int32(pt[0][0])
    y = np.int32(pt[0][1])
    cv.circle(src, (x, y), 5, (int(b), int(g), int(r)), 2)
cv.imshow("corners detection", src)
cv.imwrite("D:/corners.png", src)

# 二值化
src = cv.imread("D:/images/zsxq/zsxq_12.jpg")
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.imshow("binary input", gray)

# 固定阈值
ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
cv.imshow("binary", binary)
cv.imwrite("D:/binary.png", binary)

# 全局阈值
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow("otsu", binary)

# 自适应阈值
binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 10)
cv.imshow("ada", binary)
cv.imwrite("D:/ada.png", binary)

# 轮廓分析
contours, hireachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
result = np.zeros_like(src)
for cnt in range(len(contours)):
    cv.drawContours(result, contours, cnt, (0, 0, 255), 2, 8)
cv.imshow("contour", result)
cv.imwrite("D:/contour.png", result)

# 膨胀与腐蚀操作
se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))
d = cv.dilate(binary, se)
e = cv.erode(binary, se)
cv.imshow("dilate", d)
cv.imshow("erode", e)

# 开闭操作
op = cv.morphologyEx(binary, cv.MORPH_OPEN, se)
cl = cv.morphologyEx(binary, cv.MORPH_CLOSE, se)
cv.imshow("open", op)
cv.imshow("close", cl)


cv.waitKey(0)
cv.destroyAllWindows()