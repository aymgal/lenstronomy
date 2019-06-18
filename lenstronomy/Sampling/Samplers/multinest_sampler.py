__author__ = 'aymgal'

import os
import json
import shutil
import numpy as np

import lenstronomy.Util.sampling_util as utils

import pymultinest
from pymultinest.analyse import Analyzer


class MultiNestSampler(object):
    """
    Wrapper for nested sampling algorithm MultInest by F. Feroz & M. Hobson
    papers : arXiv:0704.3704, arXiv:0809.3437, arXiv:1306.2144
    pymultinest doc : https://johannesbuchner.github.io/PyMultiNest/pymultinest.html
    """

    def __init__(self, likelihood_module, prior_type='uniform', 
                 prior_means=None, prior_sigmas=None,
                 output_dir=None, output_basename='-',
                 remove_output_dir=False, use_mpi=False):
        """
        :param likelihood_module: likelihood_module like in likelihood.py (should be callable)
        :param prior_type: 'uniform' of 'gaussian', for converting the unit hypercube to param cube
        :param prior_means: if prior_type is 'gaussian', mean for each param
        :param prior_sigmas: if prior_type is 'gaussian', std dev for each param
        :param output_dir: name of the folder that will contain output files
        :param output_basename: prefix for output files
        :param remove_output_dir: remove the output_dir folder after completion
        :param use_mpi: flag directly passed to MultInest sampler (NOT TESTED)
        """
        self._ll = likelihood_module
        self.lowers, self.uppers = self._ll.param_limits

        if prior_type == 'gaussian':
            if prior_means is None or prior_sigmas is None:
                raise ValueError("For gaussian prior type, means and sigmas are required")
            self.means, self.sigmas = prior_means, prior_sigmas
        elif prior_type != 'uniform':
            raise ValueError("Sampling type {} not supported".format(prior_type))
        self.prior_type = prior_type

        num_params, self.param_names = self._ll.param.num_param()

        # here we assume number of dimensons = number of parameters
        self.n_dims = self.n_params = num_params

        if output_dir is None:
            self._output_dir = 'multinest_out_default'
        else:
            self._output_dir = output_dir

        if os.path.exists(self._output_dir):
            shutil.rmtree(self._output_dir, ignore_errors=True)
        os.mkdir(self._output_dir)

        self.files_basename = os.path.join(output_dir, output_basename)

        # required for analysis : save parameter names in json file
        with open(self.files_basename + 'params.json', 'w') as file:
            json.dump(self.param_names, file, indent=2)

        self._rm_output = remove_output_dir
        self._use_mpi = use_mpi


    def prior(self, cube, ndim, nparams):
        """
        compute the mapping between the unit cube and parameter cube (in-place)

        :param cube: unit hypercube, sampled by the algorithm
        :param ndim: number of sampled parameters
        :param nparams: total number of parameters
        """
        cube_py = self._multinest2python(cube, ndim)
        if self.prior_type == 'gaussian':
            _ = utils.cube2args_gaussian(cube_py, self.lowers, self.uppers, 
                                         self.means, self.sigmas, self.n_dims)
        elif self.prior_type == 'uniform':
            _ = utils.cube2args_uniform(cube_py, self.lowers, self.uppers, 
                                        self.n_dims)
        for i in range(self.n_dims):
            cube[i] = cube_py[i]


    def log_likelihood(self, args, ndim, nparams):
        """
        compute the log-likelihood given list of parameters

        :param args: parameter values
        :param ndim: number of sampled parameters
        :param nparams: total number of parameters
        :return: log-likelihood (from the likelihood module)
        """
        args_py = self._multinest2python(args, ndim)
        logL, _ = self._ll(args_py)
        if not np.isfinite(logL):
            print("WARNING : logL is not finite : return very low value instead")
            logL = -1e15
        return float(logL)


    def run(self, kwargs_run):
        """
        run the MultiNest nested sampler

        see https://johannesbuchner.github.io/PyMultiNest/pymultinest.html for content of kwargs_run

        :param kwargs_run: kwargs directly passed to pymultinest.run
        :return: samples, means, logZ, logZ_err, logL
        """
        print("prior type :", self.prior_type)
        print("parameter names :", self.param_names)
    
        pymultinest.run(self.log_likelihood, self.prior, self.n_dims,
                        outputfiles_basename=self.files_basename,
                        resume=False, verbose=True,
                        init_MPI=self._use_mpi, **kwargs_run)

        analyzer = Analyzer(self.n_dims, outputfiles_basename=self.files_basename)
        samples  = analyzer.get_equal_weighted_posterior()[:, :-1]
        
        data = analyzer.get_data()  # gets data from the *.txt output file
        logL = -0.5 * data[:, 1]  # since the second data column is -2*logL

        stats    = analyzer.get_stats()
        logZ     = stats['global evidence']
        logZ_err = stats['global evidence error']
        means    = stats['modes'][0]['mean']  # or better to use stats['marginals'][:]['median'] ???

        print("MultiNest output files have been saved to {}*"
              .format(self.files_basename))

        if self._rm_output:
            shutil.rmtree(self._output_dir, ignore_errors=True)
            print("MultiNest output directory removed")

        return samples, means, logZ, logZ_err, logL


    def _multinest2python(self, multinest_list, num_dims):
        """convert ctypes list to standard python list"""
        python_list = []
        for i in range(num_dims):
            python_list.append(multinest_list[i])
        return python_list