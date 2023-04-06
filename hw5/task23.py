from utils import dehomogenize, homogenize, draw_epipolar, visualize_pcd
import numpy as np
import cv2
import pdb
import os


def find_fundamental_matrix(shape, pts1, pts2):
    """
    Computes Fundamental Matrix F that relates points in two images by the:

        [u' v' 1] F [u v 1]^T = 0
        or
        l = F [u v 1]^T  -- the epipolar line for point [u v] in image 2
        [u' v' 1] F = l'   -- the epipolar line for point [u' v'] in image 1

    Where (u,v) and (u',v') are the 2D image coordinates of the left and
    the right images respectively.

    Inputs:
    - shape: Tuple containing shape of img1
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    """
    
    height, width = shape
    s = np.sqrt(height **2 + width **2)
    
    # 1. Compute scaling matrix T
    T = np.array([[1/s,    0, -1/2],
                  [  0,  1/s, -1/2],
                  [  0,    0,    1]])
    
    # 2. Compute the scaled points
    pts_norm = T @ (np.hstack((pts1, np.ones((pts1.shape[0], 1)))).T)
    pts_prime_norm = T @ (np.hstack((pts2, np.ones((pts2.shape[0], 1)))).T)
    
    pts_norm = pts_norm.T
    pts_prime_norm = pts_prime_norm.T
    
    # 3. Compute U matrix
    U = np.zeros((pts_norm.shape[0], 9))
    for i in range(pts_norm.shape[0]):
        u, v, _ = pts_norm[i]
        u_prime, v_prime, _ = pts_prime_norm[i]
        U[i] = np.array([u_prime*u, u_prime*v, u_prime, v_prime*u, v_prime*v, v_prime, u, v, 1])
        
        
    # 4. Rank reduce F_init to F_rank2
    _, _, V_svd = np.linalg.svd(U)
    F_normalized = V_svd[-1].reshape(3, 3)

    U_svd, S_svd, V_svd = np.linalg.svd(F_normalized)
    S_svd[-1] = 0
    F_rank2 = U_svd @ np.diag(S_svd) @ V_svd
    
    # 5. Compute the F matrix and normalize it to the last entry
    F = T.T @ F_rank2 @ T
    F = F / F[-1, -1]
    
    return F


def compute_epipoles(F):
    """
    Given a Fundamental Matrix F, return the epipoles represented in
    homogeneous coordinates.

    Check: e2@F and F@e1 should be close to [0,0,0]

    Inputs:
    - F: the fundamental matrix

    Return:
    - e1: the epipole for image 1 in homogeneous coordinates
    - e2: the epipole for image 2 in homogeneous coordinates
    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    U, S, V = np.linalg.svd(F)
    e1 = V[-1]
    e1 = e1 / e1[-1]  # Normalize

    U, S, V = np.linalg.svd(F.T)
    e2 = V[-1]
    e2 = e2 / e2[-1]  # Normalize
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return e1, e2


def find_triangulation(K1, K2, F, pts1, pts2):
    """
    Extracts 3D points from 2D points and camera matrices. Let X be a
    point in 3D in homogeneous coordinates. For two cameras, we have

        p1 === M1 X
        p2 === M2 X

    Triangulation is to solve for X given p1, p2, M1, M2.

    Inputs:
    - K1: Numpy array of shape (3,3) giving camera instrinsic matrix for img1
    - K2: Numpy array of shape (3,3) giving camera instrinsic matrix for img2
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - pcd: Numpy array of shape (N,4) giving the homogeneous 3D point cloud
      data
    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    
    # 1. Compute the essential matrix E for the fundamental matrix
    E = K2.T @ F @ K1

    # 2. Decompose the essential matrix E into R and t
    R1, R2, t = cv2.decomposeEssentialMat(E)
    
    # 3.1 Compute M1 and M2
    M1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    # 3.2 Four possible camera matrices for M2
    M2_possibility = [
        K2 @ np.hstack((R1,  t)),
        K2 @ np.hstack((R1, -t)),
        K2 @ np.hstack((R2,  t)),
        K2 @ np.hstack((R2, -t))
    ]

    # 4. Find the M2 that makes the most points positive
    max_positive_points = 0
    best_M2 = None
    for M2 in M2_possibility:
        pcd = cv2.triangulatePoints(M1, M2, pts1.T, pts2.T)
        pcd /= pcd[3]
        
        # 4.1 Count the number of points in front of both cameras
        positive_points = np.sum((pcd[2] > 0) & ((M2 @ pcd)[-1] > 0))
        if positive_points > max_positive_points:
            max_positive_points = positive_points
            best_M2 = M2
            
    pcd = cv2.triangulatePoints(M1, best_M2, pts1.T, pts2.T)
    pcd /= pcd[3]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pcd


if __name__ == '__main__':

    # You can run it on one or all the examples
    names = os.listdir("task23")
    output = "results/"

    if not os.path.exists(output):
        os.mkdir(output)

    for name in names:
        print(name)

        # load the information
        img1 = cv2.imread(os.path.join("task23", name, "im1.png"))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(os.path.join("task23", name, "im2.png"))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        data = np.load(os.path.join("task23", name, "data.npz"))
        pts1 = data['pts1'].astype(float)
        pts2 = data['pts2'].astype(float)
        K1 = data['K1']
        K2 = data['K2']
        shape = img1.shape

        # compute F
        F = find_fundamental_matrix(shape, pts1, pts2)
        
        #This will give you an answer you can compare with
        #Your answer should match closely once you've divided by the last entry
        FOpenCV, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

        # compute the epipoles
        e1, e2 = compute_epipoles(F)
        print(e1, e2)
        #to get the real coordinates, divide by the last entry
        print(e1[:2]/e1[-1], e2[:2]/e2[-1])

        outname = os.path.join(output, name + "_us.png")
        # If filename isn't provided or is None, this plt.shows().
        # If it's provided, it saves it
        draw_epipolar(img1, img2, F, pts1[::10, :], pts2[::10, :],
                      epi1=e1, epi2=e2, filename=outname)

        if 1:
            #you can turn this on or off
            pcd = find_triangulation(K1, K2, F, pts1, pts2)
            visualize_pcd(pcd,
                          filename=os.path.join(output, name + "_rec.png"))


