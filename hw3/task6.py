"""
Task6 Code
"""
import numpy as np
import common 
from common import save_img, read_img, get_AKAZE, get_match_points
from homography import fit_homography, homography_transform, RANSAC_fit_homography
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
    x_norm = np.linalg.norm(desc1, axis=1, keepdims=True)
    y_norm = np.linalg.norm(desc2, axis=1, keepdims=True)
    dist = x_norm ** 2  + (y_norm ** 2).T - 2 * np.dot(desc1, desc2.T) # return (N, M)
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
    distance = compute_distance(desc1, desc2)
    dist_idx = np.argsort(distance, axis=1)
    for i in range(len(distance)):
        first = distance[i, dist_idx[i, 0]]
        second = distance[i, dist_idx[i, 1]] * ratioThreshold
        if first < second:
            matches.append([i, dist_idx[i, 0]])

    return np.array(matches)

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
    # Hint:
    # Use common.get_match_points() to extract keypoint locations
    match_points = get_match_points(kp1, kp2, matches)
    stacked_img = cv2.vconcat([img1, img2])
    H1, _, _ = img1.shape
    for i in range(len(match_points)):
        start_point = tuple(match_points[:, :2][i].astype(int))
        end_point = (match_points[:, 2][i].astype(int)), (match_points[:, 3][i].astype(int) + H1)
        cv2.line(stacked_img, start_point, end_point, color=(0, 0, 255), thickness=1)
    
    return stacked_img


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
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # compute the size of the output image based on the homography
    corners_img1 = np.array([[0     ,      0, 1], 
                             [0     , h1 - 1, 1], 
                             [w1 - 1, h1 - 1, 1], 
                             [w1 - 1, 0     , 1]])
    corners_warped_img1 = (np.dot(H, corners_img1.T).T)
    
    # normalize
    corners_warped_img1 /= corners_warped_img1[:, 2:]
    
    # find the smallest and largest of x and y, and find the size after the homography
    xmin = int(min(corners_warped_img1[:, 0]))
    xmax = int(max(corners_warped_img1[:, 0]))
    ymin = int(min(corners_warped_img1[:, 1]))
    ymax = int(max(corners_warped_img1[:, 1]))
    size = (xmax - xmin + 1, ymax - ymin + 1)
    
    # compute the translation which the second image would have to overlap with
    tx = -xmin
    ty = -ymin
    
    # apply the homography and translation to the second image
    translation_matrix = np.array([[1, 0, tx], 
                                   [0, 1, ty], 
                                   [0, 0,  1]])
    H_translated = np.dot(translation_matrix, H)

    # use cv2.warpPerspective
    img1_warped_translated = cv2.warpPerspective(img1, H_translated, size)
    h1_trans, w1_trans = img1_warped_translated.shape[:2]
    
    # create an output image for placing the right and the left
    combined = np.zeros((max(h1_trans, ty + h2), max(w1_trans, tx+w2), 3), dtype=np.uint8)
    
    # place img1_tranlated in the left and img2 in the right
    
    if tx > 0 and ty > 0:
        im1_x = (0, h1_trans)
        im1_y = (0, w1_trans)
        im2_x = (ty, ty + h2)
        im2_y = (tx, tx + w2)
        combined[im1_x[0]: im1_x[1], im1_y[0]:im1_y[1], :] = img1_warped_translated
        combined[im2_x[0]: im2_x[1], im2_y[0]:im2_y[1], :] = img2
        # combined[   : h1_trans,   :w1_trans, :] = img1_warped_translated
        # combined[ ty:  ty + h2, tx:   tx+w2, :] = img2
        
    if tx < 0:
        combined = np.zeros((max(h1_trans, ty + h2), max(w1_trans-tx, tx+w2-tx), 3), dtype=np.uint8)
        combined[   : h1_trans,   -tx:w1_trans-tx, :] = img1_warped_translated
        combined[ ty:  ty + h2, tx-tx:   tx+w2-tx, :] = img2
    
    if ty < 0:
        combined = np.zeros((max(h1_trans-ty, ty + h2-ty), max(w1_trans, tx+w2), 3), dtype=np.uint8)
        combined[   -ty: h1_trans-ty,   :w1_trans, :] = img1_warped_translated
        combined[ ty-ty:  ty + h2-ty, tx:   tx+w2, :] = img2
        
        
    
    return combined


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

    stitched = None
    return stitched 


if __name__ == "__main__":
    
    # 1. read the image
    cases = ["eynsham", "florence2", "florence3", "florence3_alt", "lowetag", "mertonchapel", "mertoncourtyard", "vgg"]
    cases1 = ["eynsham"]
    if not os.path.exists("./result/task6"):
        os.makedirs("./result/task6")
            
    for case_name in cases:
        p1 = read_img(os.path.join("task6", case_name, "p1.jpg"))
        p2 = read_img(os.path.join("task6", case_name, "p2.jpg"))
        
        # 2. get the keypoints and descriptor
        kps1, desc1 = get_AKAZE(p1)
        kps2, desc2 = get_AKAZE(p2)
        
        # 3. calculate the distance, find matches, draw the lines, and save img
        ratio = 0.6
        matches = find_matches(desc1, desc2, ratio)
        res = draw_matches(p1, p2, kps1, kps2, matches)
        save_img(res, "result/task6/task6_result_" + case_name + ".jpg")
        
        point_case_6 = np.load("./task4/points_case_6.npy")
        XY = get_match_points(kps1, kps2, matches)
        H = fit_homography(XY)
        bestH = RANSAC_fit_homography(XY)
        # print("H", H)
        # print("bestH: ", bestH)
        print(case_name)
        combined_img = warp_and_combine(p1, p2, bestH)
        save_img(combined_img, "task6_"+case_name+"_combined.jpg")
        
        
        # ans = warp_and_combine(p1, p2, H)
    # #Possible starter code; you might want to loop over the task 6 images
    # to_stitch = 'lowetag'
    # I1 = read_img(os.path.join('task6',to_stitch,'p1.jpg'))
    # I2 = read_img(os.path.join('task6',to_stitch,'p2.jpg'))
    # res = make_warped(I1,I2)
    # save_img(res,"result_"+to_stitch+".jpg")
