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
    return None
    
if __name__ == "__main__":
    # Task 5

    case_name = "threebody"

    I = read_img(os.path.join("task5",case_name,"book.jpg"))
    corners = np.load(os.path.join("task5",case_name,"corners.npy"))
    size = np.load(os.path.join("task5",case_name,"size.npy"))

    result = make_synthetic_view(I, corners, size)
    save_img(result, case_name+"_frontoparallel.jpg")

