import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb


def coverPalette(N):
    # Return the palette we're using
    return np.linspace(1, 0, 2 ** N)


def linearToSRGB(M):
    """Given a matrix of linear intensities, convert to sRGB
    Adapted from: https://www.nayuki.io/page/gamma-aware-image-dithering"""
    M = np.clip(M, 0, 1)
    mask = (M < 0.0031308).astype(float)
    # Sometimes it's faster in python to do the math twice and then mask
    # rather than check at each location.
    out1 = 12.92 * M
    out2 = (M ** (1 / 2.4)) * 1.055 - 0.055
    return mask * out1 + (1 - mask) * out2


def SRGBToLinear(M):
    """Given a matrix of sRGB intensities, convert to linear
    Adapted from: https://www.nayuki.io/page/gamma-aware-image-dithering"""
    M = np.clip(M, 0, 1)
    mask = (M < 0.04045).astype(float)
    return mask * (M / 12.92) + (1 - mask) * (((M + 0.055) / 1.055) ** 2.4)


def reconstructImage(IQ, palette):
    """
    Given a quantized image IQ and their value, return a floating point image
    """
    # opencv is full of these assertions.
    # If you're looking here you may have an error.
    # Check to see what the assertion says
    assert(np.issubdtype(IQ.dtype, np.integer))
    return palette[IQ]


def upscaleNN(I, target_size):
    """
    NN upsample I until it hits a target size but without going over 4096
    """
    h, w = I.shape[:2]
    scale = 1
    while True:
        if min(h * scale, w * scale) >= target_size:
            break
        if max(h * (scale + 1), w * (scale + 1)) > 4096:
            break
        scale += 1
    # usually you should call a library but providing the call here defeats
    # the purpose :)
    shape = (scale, scale) if I.ndim == 2 else (scale, scale, 1)
    return np.kron(I, np.ones(shape))


def resizeToSquare(I, maxDim):
    """Given an image, make sure it's no bigger than maxDim on either side"""
    if(len(I) > maxDim or len(I[0]) > maxDim):
        I = cv2.resize(I, (maxDim, maxDim))
    return I


def quantize(v, palette):
    """
    Given a scalar v and array of values palette,
    return the index of the closest value
    """
    if(np.isscalar(v)):
        distances = [(v - p) ** 2 for p in palette]
        # Find the index of the closest value in the palette
        closest_index = distances.index(min(distances))
        return closest_index
    else:
        v = v.reshape(-1,1)
        # Calculate the square of the difference between v and palette
        distances = (v - palette) ** 2
        # Find the index of the closest value(s) in the palette
        closest_indices = np.argmin(distances, axis=1)
        return closest_indices
        

def quantizeNaive(IF, palette):
    """Given a floating-point image return quantized version (Naive)"""
    output = np.zeros(IF.shape,dtype=np.uint8)
    for i in range(len(IF)):
        for j in range(len(IF[0])):
            output[i][j] = quantize(IF[i][j], palette)
    return output


def quantizeFloyd(IF, palette):
    """
    Given a floating-point image return quantized version (Floyd-Steinberg)
    """
    copy_img = np.copy(IF)
    output = np.zeros(copy_img.shape,dtype=np.uint8)
    for i in range(len(copy_img)):
        for j in range(len(copy_img[0])):            
            oldValue = copy_img[i][j]
            colorIndex = quantize(oldValue,palette)
            output[i][j] = colorIndex
            newValue = palette[colorIndex]
            quant_error = oldValue - newValue
            if(i < len(copy_img)-1):
                copy_img[i + 1][j    ] += quant_error * 7 / 16
            if(1 < i and j < len(copy_img[0])-1):
                copy_img[i - 1][j + 1] += quant_error * 3 / 16
            if(j < len(copy_img[0])-1):
                copy_img[i    ][j + 1] += quant_error * 5 / 16
            if(i < len(copy_img)-1 and j < len(copy_img[0])-1):
                copy_img[i + 1][j + 1] += quant_error * 1 / 16
    return output


def quantizeFloydGamma(IF, palette):
    """
    Given a floating-point image return quantized version
    (Floyd-Steinberg with Gamma Correction)
    """
    copy_img = np.copy(IF)
    output = np.zeros(copy_img.shape,dtype=np.uint8)
    for i in range(len(copy_img)):
        for j in range(len(copy_img[0])):            
            oldValue = copy_img[i][j]
            colorIndex = quantize(oldValue,palette)
            output[i][j] = colorIndex
            newValue = palette[colorIndex]
            oldValue_linear = SRGBToLinear(oldValue)
            newValue_linear = SRGBToLinear(newValue)
            quant_error = oldValue_linear - newValue_linear
            if(i < len(copy_img)-1):
                copy_img[i + 1][j    ] += quant_error * 7 / 16
            if(1 < i and j < len(copy_img[0])-1):
                copy_img[i - 1][j + 1] += quant_error * 3 / 16
            if(j < len(copy_img[0])-1):
                copy_img[i    ][j + 1] += quant_error * 5 / 16
            if(i < len(copy_img)-1 and j < len(copy_img[0])-1):
                copy_img[i + 1][j + 1] += quant_error * 1 / 16
    return output


def parse():
    parser = argparse.ArgumentParser(description='run dither')
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('algorithm', help="What function to call")
    parser.add_argument('--numbits', default=1, type=int,
                        help="Number of bits to use; play with this!")
    parser.add_argument('--resizeto', default=500, type=int,
                        help="What to resize to; right now doesn't work")
    parser.add_argument('--grayscale', default=1, type=int,
                        help="Whether to grayscale first")
    parser.add_argument('--scaleup', default=1000, type=int,
                        help="Downsampling behaves nicer than upsampling")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    
    # args.algorithm is the algorithm which we want to use
    if args.algorithm not in globals():
        print("I don't recognize that algorithm")
        sys.exit(1)
        
    # if the folder not exist, add the folder to path
    if not os.path.exists(args.target):
        os.mkdir(args.target)

    # Get all the images in the directory ending with .jpg
    images = [fn for fn in os.listdir(args.source) if fn.endswith(".jpg")]
    # os.listdir is NOT guaranteed to read things in consistent order!
    images.sort()
    
    # Get algorithm and the palette
    # This returns the function with the given argument; this is generally
    # Horrible horrible security, but makes it convenient for the homework
    # and for you to debug
    algo_fn = globals()[args.algorithm]
    # This is just an array of values
    palette = coverPalette(args.numbits)
    print('palette: ' + str(palette))

    # Keep track of heights; we'll need it since we show some small images and
    # Will otherwise be at the mercy of the webbrowser resize
    heights = {}

    # for i,value in enumerate(array):
    #    --is shorthand for--
    # for i in range(len(array)):
    #    value = array[i]
    for imageI, image in enumerate(images):
        print("%d/%d" % (imageI, len(images)))
        # Load the image; using opencv we'll usually have to provide uint8
        I = cv2.imread(os.path.join(args.source, image))
        I = resizeToSquare(I, args.resizeto)
        
        # Convert to [0, 1]
        I = I.astype(np.float64) / 255              # change float to float64

        if args.grayscale:
            I = np.mean(I, axis=2)
        
        
        # Call the algorithm and reconstruct the image using the palette
        IQ = algo_fn(I, palette)
        R = reconstructImage(IQ, palette)
        
        # # I - correct image
        # if not os.path.exists("results_false_img_I"):
        #     os.mkdir("results_false_img_I")    
        #     i = 0
        # plt.imsave("results_false_img_I/false_img_%d.png" % i, I)
        # # IQ - after algorithm
        # if not os.path.exists("results_false_img_IQ"):
        #     os.mkdir("results_false_img_IQ")    
        # plt.imsave("results_false_img_IQ/false_img_%d.png" % i, IQ)
        # # R - fixed image
        # if not os.path.exists("results_false_img_R"):
        #     os.mkdir("results_false_img_R")    
        # plt.imsave("results_false_img_R/false_img_%d.png" % i, R)
        # i += 1
        
        

        # pdb.set_trace()
        # Store the height before we upsample
        # Sometimes it's hard to see the pixels and high-dpi screens screw up
        # Back in my day we were happy if our monitors' width had three digits
        heights[image] = I.shape[0]
        if args.scaleup > 0:
            I, R = upscaleNN(I, args.scaleup), upscaleNN(R, args.scaleup)

        # As a sanity check, we'll write back whatever image we get (including
        # whether we grayscaled it)
        # Beware that you shouldn't tamper with images that you get passed!
        # Note the *255!
        I_path = os.path.join(args.target, image + "_orig.png")
        cv2.imwrite(I_path, (I * 255).astype(np.uint8))
        R_path = f"{image}_{args.algorithm}.png"
        R_path = os.path.join(args.target, R_path)
        cv2.imwrite(R_path, (R * 255).astype(np.uint8))

    # Generate a quick viewer in a few lines. Making a super fast and
    # simple webpage can btw, make looking stuff faster than opening the folder
    # in the file explorer since you can show stuff in a table
    view_algos = ["orig", args.algorithm]

    with open(os.path.join(args.target, "view.html"), "w") as fh:
        fh.write("<html><body><table>")
        fh.write("<tr>")
        for algo in view_algos:
            fh.write(f"<td>{algo}</td>")
        fh.write("</tr>")
        for image in images:
            height = heights[image]
            fh.write("<tr>")
            for algo in view_algos:
                img_path = f"{image}_{algo}.png"
                fh.write(f"<td><img height='{height}' src='{img_path}'></td>")
            fh.write("</tr>")
        fh.write("</table></body></html>")
        fh.close()
