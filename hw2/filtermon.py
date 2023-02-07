import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from common import read_img, save_img
import os


def conv(I, f):
    """Apply same-sized convolution with a filter with zero-padding"""
    # Note that this is convolution! This is filtering but with f[::-1,::-1]
    return scipy.signal.convolve2d(
        I, f, mode='same', boundary='fill', fillvalue=0.0)


def nnUpsample(I, factor):
    """Nearest neighbor upsample an image by the given factor"""
    return np.kron(I, np.ones((factor, factor)))

# -- TODO Task 5: Who's That Filter? --
# (a): Fill in the filters to get the data to match

# load the filter
data = np.load('log1d.npz')
data_log50 = data['log50']
data_gauss50 = data['gauss50']
data_gauss53 = data['gauss53']
data_gauss_diff = data_gauss53 - data_gauss50
plt.figure()
plt.subplot(411)

plt.subplot(121)
plt.title('data_gauss50 and data_gauss53')
plt.plot(data_gauss50, label='data_gauss50')
plt.plot(data_gauss53, label='data_gauss53')
plt.legend()
plt.subplot(122)
plt.title('data_log50 and data_gauss_diff')
plt.plot(data_log50, label='data_log50')
plt.plot(data_gauss_diff, label='data_gauss_diff')
plt.legend()
plt.show()

if not os.path.exists("./Task4_2"):
    os.makedirs("./Task4_2")
save_img(data_log50, "./Task4_2/log50.png")
save_img(data_gauss50, "./Task4_2/gauss50.png")
save_img(data_gauss53, "./Task4_2/gauss53.png")
save_img(data_gauss_diff, "./Task4_2/gauss_diff.png")

filter0 = np.diag([0, 1, 0])
filter1 = np.ones((3,3))
filter2 = np.ones((3,3)) * (1/9)
filter3 = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
filter4 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])


# (b): No code

filters = [filter0, filter1, filter2, filter3, filter4]

np.random.seed(442)
data = (plt.imread("filtermon/442.png").astype(float)
        [:, :, 0] < 0.5).astype(float)


plt.figure()
plt.imshow(nnUpsample(data, 10))
plt.colorbar()
plt.savefig("input.png")


for fi, f in enumerate(filters):
    c = conv(data, f)
    sol = np.load("filtermon/output_%d.npy" % fi)

    matches = False

    if np.allclose(c, sol, rtol=1e-2, atol=1e-5):
        print("Filter %d matches" % fi)
        matches = True
    else:
        print("Filter %d doesn't match" % fi)

    plt.figure()
    fig, axs = plt.subplots(1, 2)

    im = axs[0].imshow(nnUpsample(c, 10))
    axs[0].set_title("Yours (%s)" % ("Match!" if matches else "No Match"))
    plt.colorbar(im, ax=axs[0])

    im = axs[1].imshow(nnUpsample(sol, 10))
    axs[1].set_title("Target")
    plt.colorbar(im, ax=axs[1])

    plt.tight_layout()
    plt.savefig("comparison_%d.pdf" % (fi))
