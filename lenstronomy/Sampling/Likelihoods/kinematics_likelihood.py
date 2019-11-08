import numpy as np
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Analysis.lens_analysis import LensAnalysis



class KinematicsLikelihood(object):
    """
    class to compute the likelihood of a model given a measurement of velocity dispersion
    """
    def __init__(self, multi_band_list, kwargs_model, vel_disp_measured, vel_disp_uncertainty,
                 lookup_table_class):
        """

        :param vel_disp_measured: relative vel disp (in km/s)
        :param vel_disp_uncertainty: vel disp uncertainty (in km/s)
        TODO
        """
        print("kwargs_model", kwargs_model)
        self.lensAnalysis = LensAnalysis(kwargs_model)
        self._vel_disp_measured = np.array(vel_disp_measured)
        self._vel_disp_error = np.array(vel_disp_uncertainty)
        kwargs_data0 = multi_band_list[0][0]  #TODO support multiband
        self.imageData0 = ImageData(**kwargs_data0)
        self._num_pix = self.imageData0.num_pixel_axes[0]  # assume square image
        self._delta_pix = self.imageData0.pixel_width
        self._lookup = lookup_table_class
        if not self._lookup.is_built:
            self._lookup.build_lookup_table()
        self._use_r_eff = (self._lookup.fix_r_eff is None)


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
        logL = self._logL_vel_disp(vel_disp_model, self._vel_disp_measured, self._vel_disp_error)
        return logL


    def _logL_vel_disp(self, vel_disp_model, vel_disp_measured, vel_disp_error):
        """
        log likelihood of modeled vel disp vs measured vel disp under considerations of errors

        :return: log likelihood of data given model
        """
        if not np.isfinite(vel_disp_model):
            return -1e15
        return - (vel_disp_model - vel_disp_measured) ** 2 / (2 * vel_disp_error ** 2)


    def _vel_disp_model(self, kwargs_lens, kwargs_lens_light, D_d, D_dt):
        gamma = kwargs_lens[0]['gamma']
        theta_E = kwargs_lens[0]['theta_E']
        if self._use_r_eff:
            r_eff = self._r_eff(kwargs_lens_light)
        else:
            r_eff = None
        # print("kwargs_lens_light", kwargs_lens_light)
        # if r_eff > 0.2:
        #     print("r_eff", r_eff)

        vel_disp0 = self._lookup.velocity_dispersion_interp(gamma, theta_E, r_eff=r_eff)
        D_s0  = self._lookup.lensCosmo.D_s   # important : same cosmo than the lookup table 
        D_ds0 = self._lookup.lensCosmo.D_ds  # important : same cosmo than the lookup table 
        z_d = self._lookup.z_lens
        vel_disp_model_2 = D_dt / (1 + z_d) / D_d * vel_disp0 / (D_s0/D_ds0)
        return np.sqrt(vel_disp_model_2)


    def _r_eff(self, kwargs_lens_light):
        center_x = kwargs_lens_light[0]['center_x']
        center_y = kwargs_lens_light[0]['center_y']
        r_eff = self.lensAnalysis.half_light_radius_lens(kwargs_lens_light, 
            self._delta_pix, self._num_pix, center_x=center_x, center_y=center_y, 
            model_bool_list=None)
        return r_eff


    @property
    def num_data(self):
        return 1
