__author__ = 'aymgal'

import numpy as np
import warnings

from lenstronomy.LightModel.Profiles import starlets_util
from lenstronomy.LightModel.Profiles.interpolation import Interpol
from lenstronomy.Util import util

__all__ = ['SLIT_Starlets']


class SLIT_Starlets(object):
    """
    Decomposition of an image using the Isotropic Undecimated Walevet Transform,
    also known as "starlet" or "B-spline", using the 'a trous' algorithm.

    Astronomical data (galaxies, stars, ...) are often very sparsely represented in the starlet basis.

    Based on Starck et al. : https://ui.adsabs.harvard.edu/abs/2007ITIP...16..297S/abstract
    """
    param_names = ['amp', 'n_scales', 'n_pix_x', 'n_pix_y', 'scale', 'center_x', 'center_y']
    lower_limit_default = {'amp': [0], 'n_scales': 2, 'n_pix_x': 5, 'n_pix_y': 5, 'center_x': -1000, 'center_y': -1000, 'scale': 0.000000001}
    upper_limit_default = {'amp': [1e8], 'n_scales': 20, 'n_pix_x': 1e10, 'n_pix_y': 1e10, 'center_x': 1000, 'center_y': 1000, 'scale': 10000000000}

    def __init__(self, thread_count=1, backend='pysparse', fast_inverse=True, second_gen=False, 
                 show_pysap_plots=False, force_no_backend=False):
        """
        Load pySAP package if found, and initialize the Starlet transform.

        :param thread_count: number of threads used for pySAP computations
        :param fast_inverse: if True, reconstruction is simply the sum of each scale (only for 1st generation starlet transform)
        :param second_gen: if True, uses the second generation of starlet transform 
        :param show_pysap_plots: if True, displays pySAP plots when calling the decomposition method
        :param force_no_pysap: if True, does not load pySAP and computes starlet transforms in python.
        """
        if force_no_backend is True:
            warnings.warn("The pySAP package is not used for starlet operations (forced).")
            self._backend = None
        else:
            self._backend = backend.lower()

        if self._backend == 'pysparse':
            try:
                import pysparse
            except ImportError as e:
                warnings.warn("The pysparse module from pySAP is not used for starlet operations (error during import: {}). "
                              .format(e) + "They will be performed using (slower) python routines.")
                self._backend = None
            else:
                self._class = pysparse.MRStarlet
        elif self._backend == 'pysap':
            if second_gen is True:
                raise ValueError("Second generation starlet transform not supported by pySAP (set use_pysparse=True to use it)")
            try:
                import pysap
            except ImportError as e:
                warnings.warn("The pysap module from pySAP is not used for starlet operations (error during import: {}). "
                              .format(e) + "They will be performed using (slower) python routines.")
                self._backend = None
            else:
                self._class = pysap.load_transform('BsplineWaveletTransformATrousAlgorithm')
        else:
            self._class = None

        if second_gen is False:
            self._fast_inverse = fast_inverse
        else:
            self._fast_inverse = False

        self._second_gen = second_gen
        self._show_pysap_plots = show_pysap_plots
        self.interpol = Interpol()
        self.thread_count = thread_count

    def function(self, x, y, amp=None, n_scales=None, n_pix_x=None, n_pix_y=None, scale=1, center_x=0, center_y=0):
        """
        1D inverse starlet transform from starlet coefficients stored in coeffs
        Follows lenstronomy conventions for light profiles.

        :param amp: decomposition coefficients ('amp' to follow conventions in other light profile)
        This is an ndarray with shape (n_scales, sqrt(n_pixels), sqrt(n_pixels)) or (n_scales*n_pixels,)
        :param n_scales: number of decomposition scales
        :param n_pixels: number of pixels in a single scale
        :return: reconstructed signal as 1D array of shape (n_pixels,)
        """
        if len(amp.shape) == 1:
            coeffs = util.array2cube(amp, n_scales, n_pix_x, n_pix_y)
        elif len(amp.shape) == 3:
            coeffs = amp
        else:
            raise ValueError("Starlets 'amp' has not the right shape (1D or 3D arrays are supported)")
        image = self.function_2d(coeffs, n_scales, n_pix_x, n_pix_y)
        image = self.interpol.function(x, y, image=image, scale=scale,
                                       center_x=center_x, center_y=center_y,
                                       amp=1, phi_G=0)
        return image

    def function_2d(self, coeffs, n_scales, n_pix_x, n_pix_y):
        """
        2D inverse starlet transform from starlet coefficients stored in coeffs

        :param coeffs: decomposition coefficients, 
        ndarray with shape (n_scales, n_pix_x, n_pix_y)
        :param n_scales: number of decomposition scales
        :return: reconstructed signal as 2D array of shape (n_pix_x, n_pix_y)
        """
        if self._backend is not None:
            return self._inverse_transform(coeffs, n_scales, n_pix_x, n_pix_y)
        else:
            return starlets_util.inverse_transform(coeffs, fast=self._fast_inverse, 
                                                   second_gen=self._second_gen)

    def decomposition(self, image, n_scales, n_pix_x, n_pix_y):
        """
        1D starlet transform from starlet coefficients stored in coeffs

        :param image: 2D image to be decomposed, ndarray with shape (sqrt(n_pixels), sqrt(n_pixels))
        :param n_scales: number of decomposition scales
        :return: reconstructed signal as 1D array of shape (n_scales*n_pixels,)
        """
        if len(image.shape) == 1:
            image_2d = util.array2image(image, nx=n_pix_x, ny=n_pix_y)
        elif len(image.shape) == 2:
            image_2d = image
        else:
            raise ValueError("image has not the right shape (1D or 2D arrays are supported for starlets decomposition)")
        return util.cube2array(self.decomposition_2d(image_2d, n_scales))

    def decomposition_2d(self, image, n_scales):
        """
        2D starlet transform from starlet coefficients stored in coeffs

        :param image: 2D image to be decomposed, ndarray with shape (sqrt(n_pixels), sqrt(n_pixels))
        :param n_scales: number of decomposition scales
        :return: reconstructed signal as 2D array of shape (n_scales, sqrt(n_pixels), sqrt(n_pixels))
        """
        if self._backend is not None:
            coeffs = self._transform(image, n_scales)
        else:
            coeffs = starlets_util.transform(image, n_scales, second_gen=self._second_gen)
        return coeffs

    def _inverse_transform(self, coeffs, num_scales, num_pixels_x, num_pixels_y):
        if self._fast_inverse:
            # for 1st gen starlet the reconstruction can be performed by summing all scales 
            return np.sum(coeffs, axis=0)
        self._prepare_transform(num_scales, num_pixels_x, num_pixels_y)
        coeffs = self._array2list(coeffs)
        if self._backend == 'pysparse':
            return self._inverse_transform_pysparse(coeffs, num_scales)
        else:
            return self._inverse_transform_pysap(coeffs, num_scales)

    def _inverse_transform_pysparse(self, coeffs, num_scales):
        image = self._transf.recons(coeffs, adjoint=False)
        return image

    def _inverse_transform_pysap(self, coeffs, num_scales):
        """reconstructs image from starlet coefficients"""
        self._transf.analysis_data = coeffs
        result = self._transf.synthesis()
        if self._show_pysap_plots:
            result.show()
        image = result.data
        return image

    def _transform(self, image, num_scales):
        self._prepare_transform(num_scales, image.shape[0], image.shape[1])
        if self._backend == 'pysparse':
            coeffs = self._transform_pysparse(image, num_scales)
        else:
            coeffs = self._transform_pysap(image, num_scales)
        coeffs = self._list2array(coeffs)
        return coeffs

    def _transform_pysparse(self, image, num_scales):
        coeffs = self._transf.transform(image, num_scales)
        return coeffs

    def _transform_pysap(self, image, num_scales):
        """decomposes an image into starlets coefficients"""
        self._transf.data = image
        self._transf.analysis()
        if self._show_pysap_plots:
            self._transf.show()
        coeffs = self._transf.analysis_data
        return coeffs

    def _prepare_transform(self, num_scales, num_pixels_x, num_pixels_y):
        """if needed, update the loaded pySAP transform to correct number of scales"""
        if (not hasattr(self, '_transf')
            or num_scales != self._num_scales 
            or num_pixels_x != self._num_pixels_x 
            or num_pixels_y != self._num_pixels_y):
            if self._backend == 'pysparse':
                self._transf = self._class(bord=0, gen2=self._second_gen, verbose=False,
                                           nb_procs=self.thread_count)
            else:
                if num_pixels_x != num_pixels_y:
                    raise ValueError("The 'pysap' backend only supports starlet transform for of square images")
                self._transf = self._class(nb_scale=num_scales, verbose=False, 
                                           nb_procs=self.thread_count)
            self._num_scales = num_scales
            self._num_pixels_x = num_pixels_x
            self._num_pixels_y = num_pixels_y

    @staticmethod
    def _list2array(coeffs):
        """convert pySAP decomposition coefficients to numpy array"""
        return np.asarray(coeffs)

    @staticmethod
    def _array2list(coeffs):
        """convert coefficients stored in numpy array to list required by pySAP"""
        coeffs_list = []
        for i in range(coeffs.shape[0]):
            coeffs_list.append(coeffs[i, :, :])
        return coeffs_list

    def delete_cache(self):
        """delete the cached interpolated image"""
        self.interpol.delete_cache()
