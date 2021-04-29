import numpy as np
import numpy.testing as npt
import pytest
import unittest

from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LightModel.Profiles.gaussian import Gaussian
from lenstronomy.LightModel.Profiles.starlets import SLIT_Starlets
from lenstronomy.Util import util


_force_no_backend = False  # if issues on Travis-CI to install pysap, force use python-only functions


class TestSLITStarlets(object):
    """
    class to test SLIT_Starlets light profile
    """
    def setup(self):
        # different versions of Starlet transforms
        self.starlets = SLIT_Starlets(fast_inverse=False, second_gen=False, force_no_backend=_force_no_backend)
        self.starlets_fast = SLIT_Starlets(fast_inverse=True, second_gen=False, force_no_backend=_force_no_backend)
        self.starlets_2nd = SLIT_Starlets(second_gen=True, force_no_backend=_force_no_backend)

        # define a test image with gaussian components
        self.n_scales = 3
        self.n_pix_x = 40
        self.n_pix_y = 50  # testing non-square images at the same time
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = util.make_grid_with_coordtransform(self.n_pix_x, 1)
        self.x, self.y = util.grid_from_coordinate_transform(self.n_pix_x, self.n_pix_y, Mpix2coord, ra_at_xy_0, dec_at_xy_0)

        # build a non-trivial positive image from sum of gaussians
        gaussian = Gaussian()
        gaussian1 = gaussian.function(self.x, self.y, amp=100, sigma=1, center_x=-7, center_y=-7)
        gaussian2 = gaussian.function(self.x, self.y, amp=500, sigma=3, center_x=-3, center_y=-3)
        gaussian3 = gaussian.function(self.x, self.y, amp=2000, sigma=5, center_x=+5, center_y=+5)
        self.test_image = util.array2image(gaussian1 + gaussian2 + gaussian3, nx=self.n_pix_x, ny=self.n_pix_y)

        self.test_coeffs = np.zeros((self.n_scales, self.n_pix_x, self.n_pix_y))
        self.test_coeffs[0, :, :] = util.array2image(gaussian1, nx=self.n_pix_x, ny=self.n_pix_y)
        self.test_coeffs[1, :, :] = util.array2image(gaussian2, nx=self.n_pix_x, ny=self.n_pix_y)
        self.test_coeffs[2, :, :] = util.array2image(gaussian3, nx=self.n_pix_x, ny=self.n_pix_y)

    def test_reconstructions_2d(self):
        """

        :return:
        """
        # PySAP requires call to decomposition once before anything else
        self.starlets.decomposition_2d(self.test_image, self.n_scales)
        self.starlets_fast.decomposition_2d(self.test_image, self.n_scales)
        self.starlets_2nd.decomposition_2d(self.test_image, self.n_scales)

        image = self.starlets.function_2d(coeffs=self.test_coeffs, n_scales=self.n_scales, n_pix_x=self.n_pix_x, n_pix_y=self.n_pix_y)
        image_fast = self.starlets_fast.function_2d(coeffs=self.test_coeffs, n_scales=self.n_scales, n_pix_x=self.n_pix_x, n_pix_y=self.n_pix_y)
        assert image.shape == (self.n_pix_x, self.n_pix_y)
        assert image_fast.shape == (self.n_pix_x, self.n_pix_y)

        image_2nd = self.starlets_2nd.function_2d(coeffs=self.test_coeffs, n_scales=self.n_scales, n_pix_x=self.n_pix_x, n_pix_y=self.n_pix_y)
        assert image_2nd.shape == (self.n_pix_x, self.n_pix_y)
        assert np.all(image_2nd >= 0)

    def test_decompositions_2d(self):
        """

        :return:
        """
        # test equality between fast and std transform (which are identical)
        coeffs = self.starlets.decomposition_2d(self.test_image, self.n_scales)
        coeffs_fast = self.starlets_fast.decomposition_2d(self.test_image, self.n_scales)
        assert coeffs.shape == (self.n_scales, self.n_pix_x, self.n_pix_y)
        assert coeffs_fast.shape == (self.n_scales, self.n_pix_x, self.n_pix_y)
        npt.assert_almost_equal(coeffs, coeffs_fast, decimal=3)

        # test non-negativity of second generation starlet transform
        coeffs_2nd = self.starlets_2nd.decomposition_2d(self.test_image, self.n_scales)
        assert coeffs_2nd.shape == (self.n_scales, self.n_pix_x, self.n_pix_y)

    def test_function(self):
        """

        :return:
        """
        # PySAP requires a call to decomposition once before anything else
        self.starlets.decomposition(self.test_image, self.n_scales, self.n_pix_x, self.n_pix_y)
        self.starlets_fast.decomposition(self.test_image, self.n_scales, self.n_pix_x, self.n_pix_y)
        self.starlets_2nd.decomposition(self.test_image, self.n_scales, self.n_pix_x, self.n_pix_y)

        # test with a 1D input
        self.starlets.decomposition(util.image2array(self.test_image), self.n_scales, self.n_pix_x, self.n_pix_y)

        coeffs_1d = self.test_coeffs.reshape(self.n_scales*self.n_pix_x*self.n_pix_y)
        
        image_1d = self.starlets.function(self.x, self.y, amp=coeffs_1d, 
                                          n_scales=self.n_scales, n_pix_x=self.n_pix_x, n_pix_y=self.n_pix_y)
        assert image_1d.shape == (self.n_pix_x*self.n_pix_y,)
        image_1d_fast = self.starlets_fast.function(self.x, self.y, amp=coeffs_1d, 
                                                    n_scales=self.n_scales, n_pix_x=self.n_pix_x, n_pix_y=self.n_pix_y)
        assert image_1d_fast.shape == (self.n_pix_x*self.n_pix_y,)
        image_1d_2nd = self.starlets_2nd.function(self.x, self.y, amp=coeffs_1d, 
                                                  n_scales=self.n_scales, n_pix_x=self.n_pix_x, n_pix_y=self.n_pix_y)
        assert image_1d_2nd.shape == (self.n_pix_x*self.n_pix_y,)

    def test_identity_operations_fast(self):
        """
        test the decomposition/reconstruction 

        :return:
        """
        coeffs = self.starlets_fast.decomposition_2d(self.test_image, self.n_scales)
        test_image_recon = self.starlets_fast.function_2d(coeffs=coeffs, n_scales=self.n_scales, n_pix_x=self.n_pix_x, n_pix_y=self.n_pix_y)
        npt.assert_almost_equal(self.test_image, test_image_recon, decimal=5)

    def test_identity_operations_2nd(self):
        """
        test the decomposition/reconstruction 

        :return:
        """
        coeffs = self.starlets_2nd.decomposition_2d(self.test_image, self.n_scales)
        test_image_recon = self.starlets_2nd.function_2d(coeffs=coeffs, n_scales=self.n_scales, n_pix_x=self.n_pix_x, n_pix_y=self.n_pix_y)
        npt.assert_almost_equal(self.test_image, test_image_recon, decimal=5)

    def test_delete_cache(self):
        amp = self.test_coeffs.reshape(self.n_scales*self.n_pix_x*self.n_pix_y)
        kwargs_starlets = dict(amp=amp, n_scales=self.n_scales, n_pix_x=self.n_pix_x, n_pix_y=self.n_pix_y, center_x=0, center_y=0, scale=1)
        output = self.starlets_fast.function(self.x, self.y, **kwargs_starlets)
        assert hasattr(self.starlets_fast.interpol, '_image_interp')
        self.starlets_fast.delete_cache()
        assert not hasattr(self.starlets_fast.interpol, '_image_interp')

    def test_array2list(self):
        n_scales = 3
        num_pix = 20
        coeffs = np.ones((n_scales, num_pix, num_pix))
        pysap_list = self.starlets._array2list(coeffs)
        assert len(pysap_list) == n_scales
        for i in range(n_scales):
            assert pysap_list[i].shape == coeffs[i].shape

    def test_list2array(self):
        n_scales = 3
        num_pix = 20
        pysap_list = n_scales * [np.ones((num_pix, num_pix))]
        coeffs = self.starlets._list2array(pysap_list)
        assert coeffs.shape == (n_scales, num_pix, num_pix)
        for i in range(n_scales):
            assert pysap_list[i].shape == coeffs[i].shape

class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            # try to set decomposition scale to higher than maximal value
            starlets = SLIT_Starlets(force_no_backend=True)
            # define a test image with gaussian components
            num_pix = 50
            x, y = util.make_grid(num_pix, 1)
            # build a non-trivial positive image from sum of gaussians
            gaussian = Gaussian()
            gaussian1 = gaussian.function(x, y, amp=100, sigma=1, center_x=-7, center_y=-7)
            gaussian2 = gaussian.function(x, y, amp=500, sigma=3, center_x=-3, center_y=-3)
            gaussian3 = gaussian.function(x, y, amp=2000, sigma=5, center_x=+5, center_y=+5)
            test_image = util.array2image(gaussian1 + gaussian2 + gaussian3)
            n_scales = 100
            _ = starlets.decomposition_2d(test_image, n_scales)
        with self.assertRaises(ValueError):
            # try to set decomposition scale to negative value
            starlets = SLIT_Starlets(force_no_backend=True)
            # define a test image with gaussian components
            num_pix = 50
            x, y = util.make_grid(num_pix, 1)
            # build a non-trivial positive image from sum of gaussians
            gaussian = Gaussian()
            gaussian1 = gaussian.function(x, y, amp=100, sigma=1, center_x=-7, center_y=-7)
            gaussian2 = gaussian.function(x, y, amp=500, sigma=3, center_x=-3, center_y=-3)
            gaussian3 = gaussian.function(x, y, amp=2000, sigma=5, center_x=+5, center_y=+5)
            test_image = util.array2image(gaussian1 + gaussian2 + gaussian3)
            n_scales = -1
            _ = starlets.decomposition_2d(test_image, n_scales)
        with self.assertRaises(ValueError):
            # function_split is not supported/defined for pixel-based profiles
            light_model = LightModel(['SLIT_STARLETS'])
            num_pix = 20
            x, y = util.make_grid(num_pix, 1)
            kwargs_list = [{'amp': np.ones((3, num_pix, num_pix)), 'n_scales': 3, 'n_pix_x': 20**2, 'n_pix_y': 20**2, 'center_x': 0, 'center_y': 0, 'scale': 1}]
            _ = light_model.functions_split(x, y, kwargs_list)
        with self.assertRaises(ValueError):
            # provided a wrong shape for starlet coefficients
            starlet_class = SLIT_Starlets()
            num_pix = 20
            x, y = util.make_grid(num_pix, 1)
            coeffs_wrong = np.ones((3, num_pix**2))
            kwargs_list = {'amp': coeffs_wrong, 'n_scales': 3, 'n_pix_x': 20**2, 'n_pix_y': 20**2, 'center_x': 0, 'center_y': 0, 'scale': 1}
            _ = starlet_class.function(x, y, **kwargs_list)
            image_wrong = np.ones((1, num_pix, num_pix))
            _ = starlet_class.decomposition(image_wrong, 3, 20, 20)
        with self.assertRaises(ValueError):
            # provided a wrong shape for image to be decomposed
            starlet_class = SLIT_Starlets()
            num_pix = 20
            image_wrong = np.ones((2, num_pix, num_pix))
            _ = starlet_class.decomposition(image_wrong, 3, 20, 20)

if __name__ == '__main__':
    pytest.main()
