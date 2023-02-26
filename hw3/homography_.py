"""
Homography fitting functions
You should write these
"""
import numpy as np
import random
from common import homography_transform
from matplotlib import pyplot as plt
import cv2

def fit_homography(XY):
    '''
    Given a set of N correspondences XY of the form [x,y,x',y'],
    fit a homography from [x,y,1] to [x',y',1].
    
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
    Output -H: a (3,3) homography matrix that (if the correspondences can be
            described by a homography) satisfies [x',y',1]^T === H [x,y,1]^T

    '''

    input = XY[: ,: 2]
    output = XY[: ,2 : 4]
    n = XY.shape[0]
    A = np.zeros((2 * n, 9))

    for i in range(n):
        A[2 * i, :] = [0, 0, 0, -input[i, 0], -input[i, 1], -1, output[i, 1] * input[i, 0], output[i, 1] * input[i, 1], output[i, 1]]
        A[2 * i + 1, :] = [input[i, 0], input[i, 1], 1, 0, 0, 0, -output[i, 0] * input[i, 0], -output[i, 0] * input[i, 1], -output[i, 0]]

    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape((3, 3))
    H = H / H[2, 2]
    return H


def RANSAC_fit_homography(XY, eps=1, nIters=1000):
    '''
    Perform RANSAC to find the homography transformation 
    matrix which has the most inliers
        
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
            eps: threshold distance for inlier calculation
            nIters: number of iteration for running RANSAC
    Output - bestH: a (3,3) homography matrix fit to the 
                    inliers from the best model.

    Hints:
    a) Sample without replacement. Otherwise you risk picking a set of points
       that have a duplicate.
    b) *Re-fit* the homography after you have found the best inliers
    '''
    bestH, bestCount, bestInliers = np.eye(3), -1, np.zeros((XY.shape[0],))
    bestRefit = np.eye(3)

    size = XY.shape[0]
    for trials in range(nIters):
        count = 0
        rand1 = random.randrange(0, size)
        rand2 = random.randrange(0, size)
        rand3 = random.randrange(0, size)
        rand4 = random.randrange(0, size)
        point1 = XY[rand1]
        point2 = XY[rand2]
        point3 = XY[rand3]
        point4 = XY[rand4]
        set = np.vstack((point1, point2, point3, point4))
        H = fit_homography(set)
        #############
        # a = XY[0 : 4, 0 : 2].reshape((-1, 2)).astype(np.float32)
        # b = XY[0 : 4, 2 : 4].reshape((-1, 2)).astype(np.float32)
        # H = cv2.getPerspectiveTransform(a, b)
        #############
        original = np.vstack((XY[:, 0 : 2].T, np.ones((1, size))))
        target = np.vstack((XY[:, 2 : 4].T, np.ones((1, size))))
        transform = H.dot(original)
        diff = target - transform
        error = np.linalg.norm(diff, axis = 0)
        inliers = error < eps
        count = np.sum(inliers)
        if(count > bestCount):
            bestCount = count
            bestH = H
            
    bestRefit = bestH

    return bestRefit

if __name__ == "__main__":
    #If you want to test your homography, you may want write any code here, safely
    #enclosed by a if __name__ == "__main__": . This will ensure that if you import
    #the code, you don't run your test code too
    case = np.load("./task4/points_case_5.npy")
    H = fit_homography(case)
    H_R = RANSAC_fit_homography(case)
    print(H)
    print(H_R)
    n = case.shape[0]
    input = np.vstack((case[: , : 2].T, np.ones((1, n))))
    transform = H.dot(input)
    
    plt.scatter(case[:, 0], case[:, 1], label = 'orginal')
    plt.scatter(case[:, 2], case[:, 3], label = 'target')
    print(transform)
    plt.scatter(transform[0, :] / transform[2, :], transform[1, :] / transform[2, :], label = 'transform')
#     plt.scatter(transform[0, :], transform[1, :], label = 'transform')
    plt.legend()
#     plt.savefig('./task4/case_5')
    plt.show()
    pass
