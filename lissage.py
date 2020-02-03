import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve


# Load the data from data.astropy.org

def fantom(file1):
    file2 = []
    for i in range(len(file1)):
        file3 = []
        for c in range(len(file1[i])):
            if not np.isnan(file1[i][c]):
                file3 += [file1[i][c]]
        if file3 != []:
            file2 += [file3]
    hdu = fits.PrimaryHDU()
    hdu.data = file2
    # print(file2)
    return file2


def contrast(file1):
    file1 = np.float64(file1)
    file1 = ((file1 - 128) / 128) * (np.pi / 2)
    file1 = 1*np.tanh(file1) + 1
    file1 = (file1 * 128) / 1
    return file1


filename = get_pkg_data_filename('ds92.fits')
#filename = fantom(filename)
hdu = fits.open(filename)[0]

# Scale the file to have reasonable numbers
# (this is mostly so that colorbars do not have too many digits)
# Also, we crop it so you can see individual pixels
#img = hdu.data[50:90, 60:100] * 1e5

# This example is intended to demonstrate how astropy.convolve and
# scipy.convolve handle missing data, so we start by setting the
# brightest pixels to NaN to simulate a "saturated" data set
#img[img > 2e1] = np.nan
img = hdu.data
img = fits.getdata("ds92.fits")
img = fantom(img)
img = contrast(img)
plt.imshow(img)
plt.show()
# We also create a copy of the data and set those NaNs to zero.  We will
# use this for the scipy convolution
img_zerod = img.copy()
img_zerod[np.isnan(img)] = 0

# We smooth with a Gaussian kernel with x_stddev=1 (and y_stddev=1)
# It is a 9x9 array
kernel = Gaussian2DKernel(x_stddev=1)


# Convolution: scipy's direct convolution mode spreads out NaNs (see
# panel 2 below)
scipy_conv = scipy_convolve(img, kernel, mode='same', method='direct')

plt.imshow(scipy_conv)
plt.show()
