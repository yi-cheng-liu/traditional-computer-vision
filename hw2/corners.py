import os
import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use zero padding (Set mode = 'constant'). Refer docs for further info.

from common import read_img, save_img


def corner_score(image, u=5, v=5, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: a image of size H x W
    """
    H, W = image.shape
    H_window, W_window = window_size
    H_pad, W_pad = H_window//2, W_window//2
    results = np.zeros_like(image)
    image_pad = np.pad(image, ((H_pad, H_pad), (W_pad, W_pad)))
    image_pad_offset = np.roll(image_pad, (u, v), axis=(0,1))
    for i in range(H):
        for j in range(W):
            unshift = image_pad[i : i+H_window, j : j+W_window]
            shifted = image_pad_offset[i : i+H_window, j : j+W_window]
            results[i, j] = np.sum(np.square(shifted - unshift))
    return results


def harris_detector(image, window_size=(5, 5)):
    """
    Given an input image, calculate the Harris Detector score for all pixels
    You can use same-padding for intensity (or 0-padding for derivatives)
    to handle window values outside of the image.

    Input- image: H x W
    Output- results: a image of size H x W
    """
    # compute the derivatives

    # For each image location, construct the structure tensor and calculate
    # the Harris response
    # sobel operator
    sobel_x = np.array([[1, 0, -1], 
                        [2, 0, -2], 
                        [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], 
                        [0, 0, 0], 
                        [-1, -2, -1]])
    
    # partial derivatives Ix and Iy
    Ix = scipy.ndimage.convolve(image, sobel_x, mode='constant')
    Iy = scipy.ndimage.convolve(image, sobel_y, mode='constant')
    # weight = np.sqrt(np.square(Ix) + np.square(Iy))
    
    # calculate the derivitate for the whole picture
    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2
    H, W = image.shape
    response = np.zeros(image.shape)
    half_w = window_size[0] // 2
    half_h = window_size[1] // 2
    for i in range(half_w, H - half_w):
        for j in range(half_h, W - half_h):
            # calculate only the M matrix for the window
            window_Ixx = Ixx[i-half_w : i+half_w+1, j-half_h : j+half_h+1]
            window_Ixy = Ixy[i-half_w : i+half_w+1, j-half_h : j+half_h+1]
            window_Iyy = Iyy[i-half_w : i+half_w+1, j-half_h : j+half_h+1]
            M = np.array([[np.sum(window_Ixx), np.sum(window_Ixy)],
                          [np.sum(window_Ixy), np.sum(window_Iyy)]])
            
            # calculate the response R
            alpha = 0.04       # 0.04 ~ 0.06
            response[i, j] = np.linalg.det(M) - alpha * (np.trace(M) ** 2)

    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # -- TODO Task 6: Corner Score --
    # (a): Complete corner_score()

    # (b)
    # Define offsets and window size and calulcate corner score
    u = 5
    v = 5
    W = (5,5)
    
    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score.png")

    # Computing the corner scores for various u, v values.
    score = corner_score(img, 0, 5, W)
    save_img(score, "./feature_detection/corner_score05.png")

    score = corner_score(img, 0, -5, W)
    save_img(score, "./feature_detection/corner_score0-5.png")

    score = corner_score(img, 5, 0, W)
    save_img(score, "./feature_detection/corner_score50.png")

    score = corner_score(img, -5, 0, W)
    save_img(score, "./feature_detection/corner_score-50.png")

    # (c): No Code

    # -- TODO Task 7: Harris Corner Detector --
    # (a): Complete harris_detector()

    # (b)
    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")


if __name__ == "__main__":
    main()
