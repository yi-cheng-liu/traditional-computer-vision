"""
Task 5 Code
"""
import numpy as np
from matplotlib import pyplot as plt
from common import save_img, read_img
from homography import fit_homography, homography_transform
import os
import cv2


def make_synthetic_view(img, corners, size):
    '''
    Creates an image with a synthetic view of selected region in the image
    from the front. The region is bounded by a quadrilateral denoted by the
    corners array. The size array defines the size of the final image.

    Input - img: image file of shape (H,W,3)
            corner: array containing corners of the book cover in 
            the order [top-left, top-right, bottom-right, bottom-left]  (4,2)
            size: array containing size of book cover in inches [height, width] (1,2)

    Output - A fronto-parallel view of selected pixels (the book as if the cover is
            parallel to the image plane), using 100 pixels per inch.
    '''
    height = size[0,0]
    width = size[0,1]
    
    # method 1: use fit_homography written in homography.py
    final_point = np.float32([(0, 0), (100*width-1, 0), (100*width-1, 100*height-1), (0, 100*height-1)])
    data = np.hstack((corners, final_point))
    print(data)
    H = fit_homography(data)
    print(H)
    output = cv2.warpPerspective(img, H, (int(100*width), int(100*height)))
    
    # # method 2: use cv2.findHomography
    # # 1. locate the final destination
    # final_point = np.float32([(0, 0), (100*width, 0), (100*width, 100*height), (0, 100*height)])
    # # 2. calculate the homography matrix
    # H = cv2.findHomography(corners, final_point)[0]
    # output = cv2.warpPerspective(img, H, (int(100*width), int(100*height)))

    return output
    
if __name__ == "__main__":
    # Task 5

    case_name = "threebody"

    I = read_img(os.path.join("task5",case_name,"book.jpg"))
    corners = np.load(os.path.join("task5",case_name,"corners.npy"))
    size = np.load(os.path.join("task5",case_name,"size.npy"))

    result = make_synthetic_view(I, corners, size)
    save_img(result, case_name+"_frontoparallel.jpg")

