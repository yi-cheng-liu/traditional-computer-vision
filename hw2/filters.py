import os

import numpy as np
import scipy.ndimage
from common import read_img, save_img


def image_patches(image, patch_size=(16, 16)):
    """
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """
    # TODO: Use slicing to complete the function
    H, W = image.shape
    M, N = patch_size
    output = []
    # ignore the end of the row and column which cannot be patched as 16x16
    for i in range(0, H-M, M):       # row
        for j in range(0, W-N, N):   # column
            patch = np.array(image[i:i+M , j:j+N])
            patch_std = np.std(patch)
            patch_mean = np.mean(patch)
            patch = (patch - patch_mean)/patch_std
            output.append(patch)
    return output

def convolve(image, kernel):
#     """
#     Return the convolution result: image * kernel.
#     Reminder to implement convolution and not cross-correlation!
#     Caution: Please use zero-padding.

#     Input- image: H x W
#            kernel: h x w
#     Output- convolve: H x W
#     """
    H, W = image.shape
    h, w = kernel.shape
    
    # set the paddings
    h_pad, w_pad = h//2, w//2
    image_pad = np.pad(image, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant')
    
    # flip the kernel
    kernel = np.flip(kernel)
        
    # Apply linear filter - convolution
    output = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
           output[i, j] = np.sum(image_pad[i:i+h, j:j+w] * kernel)
    return output

def edge_detection(image):
    """
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # TODO: Fix kx, ky
    kx = np.array([-1, 0, 1]).reshape(1,3)  # 1 x 3
    ky = kx.T                               # 3 x 1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(np.square(Ix) + np.square(Iy))

    return Ix, Iy, grad_magnitude

def sobel_operator(image):
    """
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    # TODO: Use convolve() to complete the function
    sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])   # 3 x 3
    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])   # 3 x 3
    Gs = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])   # 3 x 3
    
    Gx = convolve(image, sx)
    Gy = convolve(image, sy)
    
    kx = np.array([[1, 0, -1]])
    gaussian = scipy.ndimage.convolve(Gs, kx)
    print('Sobel filter')
    print(sx)
    print('Gaussian filter')
    print(gaussian)
    
    grad_magnitude = np.sqrt(np.square(Gx) + np.square(Gy))

    return Gx, Gy, grad_magnitude

def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # (Done) -- TODO Task 1: Image Patches --
    # (a)
    print("img.shape:", img.shape)
    save_img(img, "./image_patches/img.png")
    # First complete image_patches()
    patches = image_patches(img)
    print("patches.shape ", len(patches))
    # Now choose any `  three patches and save them
    # chosen_patches should have those patches stacked vertically/horizontally
    chosen_patches = np.hstack(patches[150: 153]) 
    save_img(chosen_patches, "./image_patches/q1_patch.png")
    
    # (b), (c): No code
    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # (Done) -- TODO Task 2: Convolution and Gaussian Filter --
    # (a): No code
    
    # (b): Complete convolve()
    kernel_test = np.ones((5, 5))
    convolve_res = convolve(img, kernel_test)
    convolve_scipy = scipy.ndimage.convolve(img, kernel_test, mode='constant')
    save_img(img, "./Task2/img.png")
    save_img(convolve_res, "./Task2/convolve_res.png")
    save_img(convolve_scipy, "./Task2/convolve_scipy.png")
    if(convolve_res.all() == convolve_scipy.all()):
        print("Same!!")
    else:
        print("Not same")
    
    # (c)
    # Calculate the Gaussian kernel described in the question.
    # There is tolerance for the kernel.
    sigma = 2
    kernel_size = 5
    kernel_gaussian = np.zeros((kernel_size, kernel_size))
    center = kernel_size//2
    for i in range(kernel_gaussian.shape[0]):
        for j in range(kernel_gaussian.shape[1]):
            x = i - center
            y = j - center
            kernel_gaussian[i, j] = (1 / (2 * np.pi * sigma**2)) \
                                    * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    # normalizing the kernel
    kernel_gaussian = kernel_gaussian/ np.sum(kernel_gaussian)    
    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")
    

    # (d), (e): No code

    # (f): Complete edge_detection()
    
    # (g)
    # Use edge_detection() to detect edges
    # for the orignal and gaussian filtered images.
    _, _, edge_detect = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")

    # (Done) -- TODO Task 3: Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # (a): No code
    
    # (b): Complete sobel_operator()

    # (c)
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    print("Sobel Operator is done. ")

    # -- TODO Task 4: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0,  1, 0], 
                            [1, -4, 1], 
                            [0,  1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3,   2,   2,   2, 3, 0, 0],
                            [0, 2, 3,   5,   5,   5, 3, 2, 0],
                            [3, 3, 5,   3,   0,   3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5,   3,   0,   3, 5, 3, 3],
                            [0, 2, 3,   5,   5,   5, 3, 2, 0],
                            [0, 0, 3,   2,   2,   2, 3, 0, 0]])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    filtered_LoG2 = convolve(img, kernel_LoG2)
    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # (b)
    # Follow instructions in pdf to approximate LoG with a DoG
    print("LoG Filter is done. ")


if __name__ == "__main__":
    main()
