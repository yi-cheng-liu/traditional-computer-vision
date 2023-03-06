"""
Task 7 Code
"""
import numpy as np
import common 
from common import save_img, read_img, get_AKAZE, get_match_points
from homography import homography_transform, RANSAC_fit_homography
from task6 import find_matches
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
# 1. get the dimentions of the images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 2. compute the size of the output image based on the homography
    corners_img2 = np.array([[0     ,      0, 1], 
                             [0     , h2 - 1, 1], 
                             [w2 - 1, h2 - 1, 1], 
                             [w2 - 1, 0     , 1]])
    corners_warped_img2= (np.dot(H, corners_img2.T).T)
    
    # 3. normalize the corners
    corners_warped_img2 /= corners_warped_img2[:, 2:]
    
    # 4. find the smallest and largest of x and y, and find the size after the homography
    xmin = int(min(corners_warped_img2[:, 0]))
    xmax = int(max(corners_warped_img2[:, 0]))
    ymin = int(min(corners_warped_img2[:, 1]))
    ymax = int(max(corners_warped_img2[:, 1]))
    size = (xmax - xmin + 1, ymax - ymin + 1)
    
    # 5. compute the translation which the second image would have to overlap with
    tx = -xmin
    ty = -ymin
    
    # 6. apply the homography and translation to the second image
    translation_matrix = np.array([[1, 0, tx], 
                                   [0, 1, ty], 
                                   [0, 0,  1]])
    H_translated = np.dot(translation_matrix, H)

    # 7. use cv2.warpPerspective
    img2_warped_translated = cv2.warpPerspective(transfer, H_translated, size)
    h2_trans, w2_trans = img2_warped_translated.shape[:2]
    
    # 8. create a combined image and place img1_tranlated and img2
    combined = np.zeros((max(ty+h1, h2_trans), max(tx+w1, w2_trans), 3), dtype=np.uint8)
    im1_x = ( ty,    ty+h1)
    im1_y = ( tx,    tx+w1)
    im2_x = (  0, h2_trans)
    im2_y = (  0, w2_trans)
        
    if tx < 0 and ty > 0:
        combined = np.zeros((max(ty+h1, h2_trans), max(w1, w2_trans-tx), 3), dtype=np.uint8)
        im1_y = (  0,          w1)
        im2_y = (-tx, w2_trans-tx)

    elif tx > 0 and ty < 0:
        combined = np.zeros((max(h1, h2_trans-ty), max(tx+w1, w2_trans), 3), dtype=np.uint8)
        im1_x = (  0,          h1)
        im2_x = (-ty, h2_trans-ty)
    
    elif tx < 0 and ty < 0:
        combined = np.zeros((max(h1, h2_trans-ty), max(w1, w2_trans-tx), 3), dtype=np.uint8)
        im1_y = (  0,          w1)
        im2_y = (-tx, w2_trans-tx)
        im1_x = (  0,          h1)
        im2_x = (-ty, h2_trans-ty)
        
    combined[im1_x[0]: im1_x[1], im1_y[0]:im1_y[1], :] = img1
    combined_transfer = np.zeros_like(combined)
    combined_transfer[im2_x[0]: im2_x[1], im2_y[0]:im2_y[1], :] = img2_warped_translated
    
    mask1 = np.logical_and(combined >= 0, combined_transfer > 0)
    combined[mask1] = combined_transfer[mask1]
    
    return combined

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
    # 1. get the keypoints and descriptor
    kps1, desc1 = get_AKAZE(scene)
    kps2, desc2 = get_AKAZE(template)
    
    # 2. change the transfer size to the size of the template
    tem_h, tem_w = template.shape[1], template.shape[0]
    transfer_resized = cv2.resize(transfer, (tem_h, tem_w))
    
    # 2. calculate the distance, get the Nx4 array of matches, and find the best Homography transform H
    ratio = 0.6
    matches = find_matches(desc2, desc1, ratio)
    XY = get_match_points(kps2, kps1, matches)
    bestH = RANSAC_fit_homography(XY)
    
    augment = task7_warp_and_combine(scene, template, bestH, transfer_resized)
    
    return augment

if __name__ == "__main__":
    # Task 7
    scenes = ["bbb", 'florence', 'lacroix', 'lego']
    seals = ['british', 'michigan', 'monk', 'um', 'wolverine', 'toysrus']
        
    for scene_name in scenes:
        # create folder for the results
        if not os.path.exists("./result/task7/"+scene_name):
            os.makedirs("./result/task7/"+scene_name)

        for transfer in seals:
            # read the images from task7 folder
            scene = read_img(os.path.join('./task7/scenes', scene_name,'scene.jpg'))
            template = read_img(os.path.join('./task7/scenes', scene_name,'template.png'))
            seal = read_img(os.path.join('./task7/seals/'+transfer+'.png'))
            
            # AUGMENT REALITY
            print(scene_name+' - '+transfer)
            augmented_img = improve_image(scene, template, seal)
            save_img(augmented_img, "result/task7/"+scene_name+"/"+transfer+"_augmented.jpg")
    pass
