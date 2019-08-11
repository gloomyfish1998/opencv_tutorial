from skimage import data, io, filters, feature, segmentation
from skimage import color, exposure, measure, morphology, draw
from matplotlib import pyplot as plt
from skimage import transform as tf


def show_image():
    image = data.chelsea()
    io.imshow(image)
    io.show()

    gray = color.rgb2gray(image)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()

    ax[0].imshow(image)
    ax[0].set_title("Input RGB")
    ax[1].imshow(gray, cmap=plt.cm.gray)
    ax[1].set_title("gray")

    fig.tight_layout()
    plt.show()

    hsv_img = color.rgb2hsv(image)
    hue_img = hsv_img[:, :, 0]
    value_img = hsv_img[:, :, 2]

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 2))

    ax0.imshow(image)
    ax0.set_title("RGB image")
    ax0.axis('off')
    ax1.imshow(hue_img, cmap='hsv')
    ax1.set_title("Hue channel")
    ax1.axis('off')
    ax2.imshow(value_img)
    ax2.set_title("Value channel")
    ax2.axis('off')

    fig.tight_layout()
    plt.show()


def filters_demo():
    image = data.chelsea()
    gray = color.rgb2gray(image)
    blur = filters.gaussian(image, 15)
    usm = filters.unsharp_mask(image, 3, 1.0)
    sobel = filters.sobel(gray)
    prewitt = filters.prewitt(gray)
    eh = exposure.equalize_adapthist(gray)
    lapl = filters.laplace(image, 3)
    median = filters.median(gray)
    fig, axes = plt.subplots(4, 2, figsize=(4, 2))
    ax = axes.ravel()

    ax[0].imshow(image)
    ax[0].set_title("Input RGB")
    ax[1].imshow(blur)
    ax[1].set_title("Gaussian Blur")

    ax[2].imshow(usm)
    ax[2].set_title("sharpen")
    ax[3].imshow(sobel, cmap=plt.cm.gray)
    ax[3].set_title("Sobel")

    ax[4].imshow(prewitt, cmap=plt.cm.gray)
    ax[4].set_title("prewitt")
    ax[5].imshow(eh, cmap=plt.cm.gray)
    ax[5].set_title("equalize_adapthist")

    ax[6].imshow(lapl)
    ax[6].set_title("laplace")
    ax[7].imshow(median, cmap=plt.cm.gray)
    ax[7].set_title("median")

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    ax[4].axis('off')
    ax[5].axis('off')
    ax[6].axis('off')
    ax[7].axis('off')

    fig.tight_layout()
    plt.show()


def binary_demo():
    image = io.imread("D:/images/dice.jpg")
    gray = color.rgb2gray(image)
    ret = filters.threshold_otsu(gray)
    print(ret)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()

    ax[0].imshow(image)
    ax[0].set_title("Input RGB")
    ax[1].imshow(gray > ret, cmap='gray')
    ax[1].set_title("binary")
    ax[0].axis('off')
    ax[1].axis('off')

    fig.tight_layout()
    plt.show()


def contour_demo():
    image = io.imread("D:/images/contours.png")
    gray = color.rgb2gray(image)
    ret = filters.threshold_otsu(gray)
    print(ret)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()
    # 轮廓发现
    binary = gray > ret
    ax[0].imshow(gray > ret, cmap='gray')
    ax[0].set_title("binary")
    contours = measure.find_contours(binary, 0.8)
    for n, contour in enumerate(contours):
        ax[1].plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax[1].set_title("contours")

    ax[0].axis('off')
    ax[1].axis('off')
    fig.tight_layout()
    plt.show()


def canny_demo():
    image = io.imread("D:/images/master.jpg")
    gray = color.rgb2gray(image)
    edge = feature.canny(gray, 3)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()

    ax[0].imshow(image)
    ax[0].set_title("Input RGB")
    ax[1].imshow(edge, cmap='gray')
    ax[1].set_title("Canny")
    ax[0].axis('off')
    ax[1].axis('off')

    fig.tight_layout()
    plt.show()


def morph_demo():
    image = data.horse()
    gray = color.rgb2gray(image)
    ret = filters.threshold_otsu(gray)
    binary = gray < ret # 二值化反
    skele = morphology.skeletonize(binary)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()

    ax[0].imshow(gray,cmap='gray')
    ax[0].set_title("Input ")
    ax[1].imshow(skele, cmap='gray')
    ax[1].set_title("skeletonize")
    ax[0].axis('off')
    ax[1].axis('off')

    fig.tight_layout()
    plt.show()


def corner_demo():
    image = io.imread("D:/images/home.jpg")
    gray = color.rgb2gray(image)
    coords = feature.corner_peaks(feature.corner_harris(gray), min_distance=5)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()

    ax[0].imshow(image)
    ax[0].set_title("Input ")
    ax[1].imshow(image)
    ax[1].set_title("harris corner detection")
    ax[0].axis('off')
    ax[1].axis('off')
    ax[1].plot(coords[:, 1], coords[:, 0], color='red', marker='o',
            linestyle='None', markersize=4)

    fig.tight_layout()
    plt.show()


def feature_match():
    img1 = color.rgb2gray(data.astronaut())
    tform = tf.AffineTransform(scale=(1.2, 1.2), translation=(0, -100))
    img2 = tf.warp(img1, tform)
    img3 = tf.rotate(img1, 25)

    keypoints1 = feature.corner_peaks(feature.corner_harris(img1), min_distance=5)
    keypoints2 = feature.corner_peaks(feature.corner_harris(img2), min_distance=5)
    keypoints3 = feature.corner_peaks(feature.corner_harris(img3), min_distance=5)

    extractor = feature.BRIEF()

    extractor.extract(img1, keypoints1)
    keypoints1 = keypoints1[extractor.mask]
    descriptors1 = extractor.descriptors

    extractor.extract(img2, keypoints2)
    keypoints2 = keypoints2[extractor.mask]
    descriptors2 = extractor.descriptors

    extractor.extract(img3, keypoints3)
    keypoints3 = keypoints3[extractor.mask]
    descriptors3 = extractor.descriptors

    matches12 = feature.match_descriptors(descriptors1, descriptors2, cross_check=True)
    matches13 = feature.match_descriptors(descriptors1, descriptors3, cross_check=True)

    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray()

    feature.plot_matches(ax[0], img1, img2, keypoints1, keypoints2, matches12)
    ax[0].axis('off')
    ax[0].set_title("Original Image vs. Transformed Image")

    feature.plot_matches(ax[1], img1, img3, keypoints1, keypoints3, matches13)
    ax[1].axis('off')
    ax[1].set_title("Original Image vs. Transformed Image")

    plt.show()


if __name__ == "__main__":
    feature_match()