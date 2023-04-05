import cv2
import numpy as np
from matplotlib import pyplot as plt

# images are tiny otherwise
plt.rcParams['figure.figsize'] = [15, 15]


def homogenize(X):
    """
    Given real NxK points, return Nx(K+1) homogeneous points
    """
    return np.hstack([X, np.ones((X.shape[0], 1))])


def dehomogenize(X):
    """
    Given homogeneous NxK points, return Nx(K-1) real points
    """
    K = X.shape[1]
    return X[:, :(K - 1)] / X[:, (K - 1)][:, None]


def ncolors(n):
    """
    Generate a consistent set of n colors that's somewhat evenly spaced.
    Uses the classic sample uniformly on H trick plus different saturations
    to get some more colors.

    Return a list of lists of ints [(r,g,b), ...]
    """
    def ngaps(n):
        # Generate: 1/2, 1/4, 3/4, 1/8, 3/8, ...
        vals = []
        num, den = 1, 2
        while len(vals) < n:
            vals.append((1.0 * num) / den)
            num += 2
            if num > den:
                num, den = 1, den * 2
        return vals

    colors = []

    # number of saturations to use
    numSats = 3 if n > 10 else 1

    # number of hues (we'll cycle through this numSats times)
    numHues = n // numSats
    numHues += 1 if ((numHues * numSats) < n) else 0

    gaps = ngaps(numHues)
    for i in range(n):
        h = gaps[i % numHues] * 180  # where in the hue wheel we are
        satInd = i // numHues  # which saturation index we're onto
        sat = (satInd + 1.0) / (numSats) * 255  # saturation
        # convert hsv -> rgb
        rgb = cv2.cvtColor(np.uint8([[[h, sat, 255]]]), cv2.COLOR_HSV2RGB)
        # force it to be a list of ints; opencv's picky
        colors.append(list([int(v) for v in rgb.flatten()]))
    return colors


def drawlines(img1, img2, lines, pts1, pts2):
    """
    Draw points and epipolar lines

    Substantially Improved version of opencv documentation
    https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html

    This:
    - uses a consistent set of colors, which should make debugging easier
    - doesn't blow up with vertical lines
    """

    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    colorsUse = ncolors(len(lines))

    H, W = img1.shape[0], img1.shape[1]

    X, Y = np.meshgrid(np.arange(W).astype(np.float),
                       np.arange(H).astype(np.float))

    for ii, (r, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
        color = colorsUse[ii]

        # r is [a,b,c]; draw the line ax+by+c = 0 by finding its intersection
        # with the bounding rectangle.  If it's in the image, r will cross it
        toCheck = [np.cross([0, 0, 1], [W, 0, 1]),
                   np.cross([0, 0, 1], [0, H, 1]),
                   np.cross([W, H, 1], [W, 0, 1]),
                   np.cross([W, H, 1], [0, H, 1])]

        # find the intersection with all the boundaries
        isects = []
        for i in range(len(toCheck)):
            isect = np.cross(toCheck[i], r)
            isects.append(isect[:2] / isect[2])
        # sort by distance to center of image
        isects.sort(key=lambda v: np.linalg.norm(v - np.array([W / 2, H / 2])))
        x0, y0 = map(int, isects[0])
        x1, y1 = map(int, isects[1])

        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 5)
        img1 = cv2.circle(img1, tuple(map(int, pt1)), 10, color, -1)
        img2 = cv2.circle(img2, tuple(map(int, pt2)), 10, color, -1)

    return img1, img2


def draw_epipole(img, epiH):
    # do not do anything if it is undefined or is exactly 0. Everything else
    # goes through ok -- even if the dehomogenizing makes it infinite, that's
    # fine and we'll catch it later
    if epiH is None or epiH[2] == 0:
        return img

    epi = epiH[:2] / epiH[2]
    H, W = img.shape[0], img.shape[1]
    # check to see if we can even draw it

    if (0 <= epi[0] < W) and (0 <= epi[1] < H):
        epi = tuple(map(int, epi))
        img = cv2.circle(img, epi, 20, (0, 0, 0), -1)
        img = cv2.circle(img, epi, 10, (255, 255, 255), -1)
    return img


def draw_epipolar(img1, img2, F, pts1, pts2,
                  epi1=None, epi2=None, filename=None):
    """
    Improved version of opencv documentation
    https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html

    Inputs:
    - img1: image1
    - img2: image2
    - F: the fundamental matrix
    - epi1: the epipole for image 1 in homogeneous coordinates
    - epi2: the epipole for image 2 in homogeneous coordinates
    - filename: if is None, then plt.show() else save it

    """

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_with_lines, img2_points = drawlines(img1, img2, lines1, pts1, pts2)
    img1_with_lines = draw_epipole(img1_with_lines, epi1)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_with_lines, img1_points = drawlines(img2, img1, lines2, pts2, pts1)
    img2_with_lines = draw_epipole(img2_with_lines, epi2)

    plt.figure()
    plt.subplot(221)
    plt.imshow(img1_points)
    plt.title("Image 1 points")
    plt.subplot(222)
    plt.imshow(img2_with_lines)
    plt.title("Image 2 points and epipolar lines from image 1")

    plt.subplot(223)
    plt.imshow(img1_with_lines)
    plt.title("Image 1 points and epipolar lines from image 2")

    plt.subplot(224)
    plt.imshow(img2_points)
    plt.title("Image 2 points")

    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()


def visualize_pcd(points, filename=None):
    """
    Visualize the point cloud.

    Inputs:
    - points: the matrix of points
    - filename: if None, plt.show() else save it here

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[0].T, points[2].T, points[1].T)
    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()
