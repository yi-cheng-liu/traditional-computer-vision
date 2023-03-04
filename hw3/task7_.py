"""
Task 7 Code
"""
import numpy as np
import common 
from common import save_img, read_img
from homography import homography_transform, RANSAC_fit_homography
from task6 import make_warped, find_matches
import cv2
import os

def task7_warp_and_combine(img1, img2, H, transfer):
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
                but make sure in V, for pixels covered by both img1 and warped img2,
                you see only img2
    '''
    corners1 = np.float32([[0, 0], [0, img1.shape[1]], [img1.shape[0], img1.shape[1]], [img1.shape[0], 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, img2.shape[1]], [img2.shape[0], img2.shape[1]], [img2.shape[0], 0]]).reshape(-1, 1, 2)
    transformed_corners2 = cv2.perspectiveTransform(corners2, H)
    # corner2_min = np.min(transformed_corners2, axis = 0)

    min_x = min(np.min(corners1[:, :, 0]), np.min(transformed_corners2[:, :, 0]))
    min_y = min(np.min(corners1[:, :, 1]), np.min(transformed_corners2[:, :, 1]))
    max_x = max(np.max(corners1[:, :, 0]), np.max(transformed_corners2[:, :, 0]))
    max_y = max(np.max(corners1[:, :, 1]), np.max(transformed_corners2[:, :, 1]))
    height = int(max_x - min_x)
    width = int(max_y - min_y)
    warped_transfer = cv2.warpPerspective(transfer, H, (width, height))

    merged = np.zeros((height, width, 3), dtype = np.uint8)
    merged[ : img1.shape[0], : img1.shape[1], :] = img1[:, :, :3]

    mask1 = np.logical_and(merged >= 0, warped_transfer > 0)
    merged[mask1] = warped_transfer[mask1]
    # mask2 = np.logical_and(merged == 0, warped_img2 > 0)
    # merged[mask2] = warped_img2[mask2]
        
    return merged

def improve_image(scene, template, transfer):
    '''
    Detect template image in the scene image and replace it with transfer image.

    Input - scene: image (H,W,3)
            template: image (K,K,3)
            transfer: image (L,L,3)
    Output - augment: the image with 
    
    Hints:
    a) You may assume that the template and transfer are both squares.
    b) This will work better if you find a nearest neighbor for every template
       keypoint as opposed to the opposite, but be careful about directions of the
       estimated homography and warping!
    '''
    keypoints1, descriptors1 = common.get_AKAZE(scene)
    keypoints2, descriptors2 = common.get_AKAZE(template)
    rsized_transfer = cv2.resize(transfer, (template.shape[1], template.shape[0]))

    matches = find_matches(descriptors1, descriptors2, 0.5)
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
    stitched = task7_warp_and_combine(scene, template, H, rsized_transfer)
    save_img(stitched,"task7.jpg")
    return stitched 


if __name__ == "__main__":
    # Task 7
    folder = ["bbb", 'florence', 'lacroix']
    seals = ['michigan.png', 'monk.png', 'um.png']
    scene = read_img(os.path.join('./task7/scenes', folder[1],'scene.jpg'))
    template = read_img(os.path.join('./task7/scenes', folder[1],'template.png'))
    seals = read_img(os.path.join('./task7/seals', seals[1]))
    improve_image(scene, template, seals)
    pass
