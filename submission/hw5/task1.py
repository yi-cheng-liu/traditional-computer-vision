import numpy as np
import utils


def find_projection(pts2d, pts3d):
    """
    Computes camera projection matrix M that goes from world 3D coordinates
    to 2D image coordinates.

    [u v 1]^T === M [x y z 1]^T

    Where (u,v) are the 2D image coordinates and (x,y,z) are the world 3D
    coordinates

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - M: Numpy array of shape (3,4)

    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    N = pts2d.shape[0]

    A = np.zeros((2 * N, 12))

    for i in range(N):
        x, y, z = pts3d[i, :]
        u, v = pts2d[i, :]

        A[2 * i, :]     = np.array([0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v])
        A[2 * i + 1, :] = np.array([x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u])

    _, _, V = np.linalg.svd(A)
    M = V[-1].reshape(3, 4)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return M

def compute_distance(pts2d, pts3d):
    """
    use find_projection to find matrix M, then use M to compute the average 
    distance in the image plane (i.e., pixel locations) 
    between the homogeneous points M X_i and 2D image coordinates p_i

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - float: a average distance you calculated (threshold is 0.01)

    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    
    M = find_projection(pts2d, pts3d) 
    print(M)
    N = pts2d.shape[0]
    total_distance = 0
    
    for i in range(N):
        x, y, w = pts3d[i, :]
        u, v = pts2d[i, :]
        
        homogeneous_points = M @ np.array([x, y, w, 1])
        # proj([x, ,y, w]) = [x/w, y/w]
        projected = homogeneous_points / homogeneous_points[2]
        
        distance = np.linalg.norm(projected[:2] - np.array([u, v]))
        total_distance += distance
    
    distance = total_distance / N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distance

if __name__ == '__main__':
    pts2d = np.loadtxt("task1/pts2d.txt")
    pts3d = np.loadtxt("task1/pts3d.txt")

    # Alternately, for some of the data, we provide pts1/pts1_3D, which you
    # can check your system on via
    
    # data = np.load("task23/ztrans/data.npz")
    # pts2d = data['pts1']
    # pts3d = data['pts1_3D']
    
   
    foundDistance = compute_distance(pts2d, pts3d)
    print("Distance: %f" % foundDistance)
