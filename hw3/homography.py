"""
Homography fitting functions
You should write these
"""
import numpy as np
import matplotlib.pyplot as plt
from common import homography_transform

def fit_homography(XY):
    '''
    Given a set of N correspondences XY of the form [x,y,x',y'],
    fit a homography from [x,y,1] to [x',y',1].
    
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
    Output -H: a (3,3) homography matrix that (if the correspondences can be
            described by a homography) satisfies [x',y',1]^T === H [x,y,1]^T

    '''
    # 1. extract the data 
    x = XY[:, 0]
    y = XY[:, 1]
    x_prime = XY[:, 2]
    y_prime = XY[:, 3]
    
    # 2. construct the p matrix
    p_trans = np.vstack((x,y,np.ones(len(x)))).T
    zero_row = np.zeros((1,3))
    A = np.zeros((2*len(x), 9))
    
    # 3. build the A matrix
    for i in range(len(x)):
        p_trans_neg = (-p_trans[i]).reshape(1,3)
        y_prime_p = (y_prime[i]*p_trans[i]).reshape(1,3)
        x_prime_p = (x_prime[i]*p_trans[i]).reshape(1,3)
        A[2*i]   = np.hstack((   zero_row, p_trans_neg, y_prime_p))
        A[2*i+1] = np.hstack((-p_trans_neg,    zero_row, -x_prime_p))
        
    # 4. calculate the homography matrix, and the last entry is 1
    [U, S, Vt] = np.linalg.svd(A)
    print('U dimention: ', U.shape)
    print('S dimention: ', S.shape)
    print('Vt dimention: ', Vt.shape)
    H = Vt[-1].reshape(3, 3)
    H = H * (1/H[2,2])
        
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
    return bestRefit

if __name__ == "__main__":
    #If you want to test your homography, you may want write any code here, safely
    #enclosed by a if __name__ == "__main__": . This will ensure that if you import
    #the code, you don't run your test code too
    
    pass
