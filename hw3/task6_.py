"""
Task6 Code
"""
import numpy as np
import common 
from common import save_img, read_img, get_match_points
from homography_ import fit_homography, homography_transform, RANSAC_fit_homography
import os
import cv2

def compute_distance(desc1, desc2):
    '''
    Calculates L2 distance between 2 binary descriptor vectors.
        
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
    
    Output - dist: a (N,M) L2 distance matrix where dist(i,j)
             is the squared Euclidean distance between row i of 
             desc1 and desc2. You may want to use the distance
             calculation trick
             ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y
    '''
    dist = np.linalg.norm(desc1, axis = 1).reshape((-1, 1)) ** 2 + np.linalg.norm(desc2, axis = 1).reshape((1, -1)) ** 2 - 2 * np.dot(desc1, desc2.T)
    return dist

def find_matches(desc1, desc2, ratioThreshold):
    '''
    Calculates the matches between the two sets of keypoint
    descriptors based on distance and ratio test.
    
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
            ratioThreshhold : maximum acceptable distance ratio between 2
                              nearest matches 
    
    Output - matches: a list of indices (i,j) 1 <= i <= N, 1 <= j <= M giving
             the matches between desc1 and desc2.
             
             This should be of size (K,2) where K is the number of 
             matches and the row [ii,jj] should appear if desc1[ii,:] and 
             desc2[jj,:] match.
    '''
    matches = []
    matched = []
    dist = compute_distance(desc1, desc2)
    h, w = dist.shape
    argument_sort = np.argsort(dist, axis = 1)
    for i in range(h):
        # print(dist[i][argument_sort[i][0]], ratioThreshold * dist[i][argument_sort[i][1]])
        if((dist[i][argument_sort[i][0]] < ratioThreshold * dist[i][argument_sort[i][1]]) and matched.count(argument_sort[i][1]) == 0):
            # print(dist[i][argument_sort[i][0]], ratioThreshold * dist[i][argument_sort[i][1]])
            matches.append((i, argument_sort[i][0]))
            matched.append(argument_sort[i][1])

    return matches

def draw_matches(img1, img2, kp1, kp2, matches):
    '''
    Creates an output image where the two source images stacked vertically
    connecting matching keypoints with a line. 
        
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            kp1: Keypoint matrix for image 1 of shape (N,4)
            kp2: Keypoint matrix for image 2 of shape (M,4)
            matches: List of matching pairs indices between the 2 sets of 
                     keypoints (K,2)
    
    Output - Image where 2 input images stacked vertically with lines joining 
             the matched keypoints
    Hint: see cv2.line
    '''
    #Hint:
    #Use common.get_match_points() to extract keypoint locations
    stacked = cv2.vconcat([img1, img2])
    # save_img(stacked,"123.jpg")
    # print(img1.shape)
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    matches_len = len(matches) # len = K
    for i in range(matches_len):
        idx_img1 = matches[i][0]
        idx_img2 = matches[i][1]
        x1 = int(kp1[idx_img1][0])
        y1 = int(kp1[idx_img1][1])
        x2 = int(kp2[idx_img2][0])
        y2 = int(kp2[idx_img2][1])
        start = (x1, y1)
        end = (x2, y2 + h1)
        color = (0, 0, 255)
        thickness = 2
        stacked = cv2.line(stacked, start, end, color, thickness)
    # save_img(stacked,"stacked.jpg")
    output = stacked
    return output


def warp_and_combine(img1, img2, H):
    '''
    You may want to write a function that merges the two images together given
    the two images and a homography: once you have the homography you do not
    need the correspondences; you just need the homography.
    Writing a function like this is entirely optional, but may reduce the chance
    of having a bug where your homography estimation and warping code have odd
    interactions.
    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            H: homography mapping betwen them
    Output - V: stitched image of size (?,?,3); unknown since it depends on H
    '''

    corners1 = np.float32([[0, 0], [0, img1.shape[1]], [img1.shape[0], img1.shape[1]], [img1.shape[0], 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, img2.shape[1]], [img2.shape[0], img2.shape[1]], [img2.shape[0], 0]]).reshape(-1, 1, 2)
    transformed_corners2 = cv2.perspectiveTransform(corners2, H)
    # corner2_min = np.min(transformed_corners2, axis = 0)

    min_x = min(np.min(corners1[:, :, 0]), np.min(transformed_corners2[:, :, 0]))
    min_y = min(np.min(corners1[:, :, 1]), np.min(transformed_corners2[:, :, 1]))
    max_x = max(np.max(corners1[:, :, 0]), np.max(transformed_corners2[:, :, 0]))
    max_y = max(np.max(corners1[:, :, 1]), np.max(transformed_corners2[:, :, 1]))
    # t = [int(-min_x), int(-min_y)]
    # t = [0, 0]
    # Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    height = int(max_x - min_x)
    width = int(max_y - min_y)
    warped_img2 = cv2.warpPerspective(img2, H, (width, height))

    # save_img(warped_img2,"warpedim2.jpg")
    # merged = np.zeros((width, height, 3), dtype = np.uint8)
    merged = np.zeros((height, width, 3), dtype = np.uint8)
    print(merged.shape)
    # merged[t[1] : t[1] + img1.shape[0], t[0] : t[0] + img1.shape[1], :] = img1[:, :, :3]
    merged[ : img1.shape[0], : img1.shape[1], :] = img1[:, :, :3]

    mask1 = np.logical_and(merged > 0, warped_img2 > 0)
    merged[mask1] = warped_img2[mask1] / 2 + merged[mask1] / 2
    mask2 = np.logical_and(merged == 0, warped_img2 > 0)
    merged[mask2] = warped_img2[mask2]
        
    return merged


def make_warped(img1, img2):
    '''
    Take two images and return an image, putting together the full pipeline.
    You should return an image of the panorama put together.
    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 1 of shape (H2,W2,3)
    
    Output - Final stitched image
    Be careful about:
    a) The final image size 
    b) Writing code so that you first estimate H and then merge images with H.
    The system can fail to work due to either failing to find the homography or
    failing to merge things correctly.
    '''

    keypoints1, descriptors1 = common.get_AKAZE(img1)
    keypoints2, descriptors2 = common.get_AKAZE(img2)

    matches = find_matches(descriptors1, descriptors2, 0.3)
    matches_len = len(matches) # len = K
    XY = np.zeros((matches_len, 4))
    for i in range(matches_len):
        idx_img1 = matches[i][0]
        idx_img2 = matches[i][1]
        point1 = np.array([int(keypoints1[idx_img1][0]), int(keypoints1[idx_img1][1])])
        point2 = np.array([int(keypoints2[idx_img2][0]), int(keypoints2[idx_img2][1])])
        stk_point = np.hstack((point2, point1))
        # print(stk_point)
        XY[i, :] = stk_point
    # print(XY)
    
    # H_ = cv2.getPerspectiveTransform(b, a)
    H = RANSAC_fit_homography(XY, 1, 100000)
    
    # H = fit_homography(XY)
    stitched = warp_and_combine(img1, img2, H)
    # save_img(stitched,"merged.jpg")
    return stitched 


if __name__ == "__main__":
    case = 1
    #Possible starter code; you might want to loop over the task 6 images
    to_stitch = ['eynsham', 'florence2', 'florence3', 'florence3_alt', 'lowetag', 'mertonchapel', 'mertoncourtyard', 'vgg']
    to_switch = ['eynsham']
    if(case == 1):
        for i in range(len(to_stitch)):
            folder = to_stitch[i]
            I1 = read_img(os.path.join('task6',folder,'p1.jpg'))
            I2 = read_img(os.path.join('task6',folder,'p2.jpg'))
            keypoints1, descriptors1 = common.get_AKAZE(I1)
            keypoints2, descriptors2 = common.get_AKAZE(I2)
            matches = find_matches(descriptors1, descriptors2, 0.3)
            res = draw_matches(I1, I2, keypoints1, keypoints2, matches)
            
    elif(case == 2):
        folder = to_stitch[5]
        I1 = read_img(os.path.join('task6',folder,'p1.jpg'))
        I2 = read_img(os.path.join('task6',folder,'p2.jpg'))
        make_warped(I1, I2)

