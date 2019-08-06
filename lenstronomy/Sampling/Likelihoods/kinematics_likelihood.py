import numpy as np
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Analysis.lens_analysis import LensAnalysis



class KinematicsLikelihood(object):
    """
    class to compute the likelihood of a model given a measurement of velocity dispersion
    """
    def __init__(self, multi_band_list, kwargs_model, vel_disp_measured, vel_disp_uncertainties,
                 lookup_table_class):
        """

        :param vel_disp_measured: relative time delays (in days) in respect to the first image of the point source
        :param vel_disp_uncertainties: time-delay uncertainties in same order as time_delay_measured
        TODO
        """
        self._vel_disp_measured = np.array(vel_disp_measured)
        self._vel_disp_errors = np.array(vel_disp_uncertainties)
        kwargs_data0 = multi_band_list[0][0]  #TODO support multiband
        self.imageData0 = ImageData(**kwargs_data0)
        self._num_pix = self.imageData0.num_pixel_axes[0]  # assume square image
        self._delta_pix = self.imageData0.pixel_width
        self._analysis = LensAnalysis(kwargs_model)
        self._lookup = lookup_table_class


    def logL(self, kwargs_lens, kwargs_lens_light, kwargs_cosmo):
        """
        routine to compute the log likelihood of the time delay distance
        :param kwargs_lens: lens model kwargs list
        :param kwargs_ps: point source kwargs list
        :param kwargs_cosmo: cosmology and other kwargs
        :return: log likelihood of the model given the time delay data
        """
        D_d_model  = kwargs_cosmo['D_d']
        D_dt_model = kwargs_cosmo['D_dt']
        vel_disp_model = self._vel_disp_model(kwargs_lens, kwargs_lens_light, D_d_model, D_dt_model)
        logL = self._logL_vel_disp(vel_disp_model, self._vel_disp_measured, self._vel_disp_errors)
        return logL


    def half_light_radius(self, kwargs_lens_light, center_x, center_y, model_bool_list=None):
        return self._analysis.half_light_radius_lens(kwargs_lens_light, self._delta_pix, self._num_pix, 
                center_x=center_x, center_y=center_y, model_bool_list=model_bool_list)


    def _logL_vel_disp(self, vel_disp_model, vel_disp_measured, vel_disp_errors):
        """
        log likelihood of modeled vel disp vs measured vel disp under considerations of errors

        :return: log likelihood of data given model
        """
        logL = - (vel_disp_model - vel_disp_measured) ** 2 / (2 * vel_disp_errors ** 2)
        return logL


    def _vel_disp_model(self, kwargs_lens, kwargs_lens_light, D_d, D_dt):
        gamma = kwargs_lens[0]['gamma']
        theta_E = kwargs_lens[0]['theta_E']
        r_eff = self._r_eff(kwargs_lens_light)
        vel_disp0 = self._lookup.velocity_dispersion_interp(gamma, theta_E, r_eff)
        D_d0  = self._lookup.cosmo.D_d
        D_s0  = self._lookup.cosmo.D_s
        D_ds0 = self._lookup.cosmo.D_ds
        z_d = self._lookup.cosmo.z_lens
        return D_dt / (1 + z_d) * vel_disp0 * D_ds0 / (D_d0 * D_s0)


    def _r_eff(self, kwargs_lens_light):
        center_x = kwargs_lens_light[0]['center_x']
        center_y = kwargs_lens_light[0]['center_y']
        r_eff = self.half_light_radius(kwargs_lens_light, center_x, center_y)
        return r_eff


    @property
    def num_data(self):
        return 1
