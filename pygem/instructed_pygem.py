"""
This class contains methods to run 2D simulations of glacier evolution,
using the ice-flow module of IGM as a solver for ice dynamics and ice thickness evolution.
The mass balance model is taken from PyGEM.

Code written by: Henning Åkesson, Johannes Brunner
Based on instructed_oggm.py by Julien Jehl, Fabien Maussion, and Guillaume Jouvet
"""

import igm.outputs.write_ncdf as igm_write

import numpy as np
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)  # Prevent TensorFlow from allocating all GPU memory
import os

from oggm import cfg, utils
from oggm.cfg import G, SEC_IN_YEAR, SEC_IN_DAY

import igm
from pygem.interface2d import Model2D

from igm.common.core.src import State
from igm.common.runner.configuration.utils import EmptyClass
from igm.common.runner.configuration.loader import load_yaml_as_cfg
from igm.utils.gradient.compute_divflux import compute_divflux


class IGM_Model2D(Model2D):
    def filter_ice_border(ice_thick):
        """Sets the ice thickness at the border of the domain to zero."""
        ice_thick[0, :] = 0
        ice_thick[-1, :] = 0
        ice_thick[:, 0] = 0
        ice_thick[:, -1] = 0
        return ice_thick

    def __init__(self, bed_topo, config, init_ice_thick=None, dx=None, dy=None, mb_model=None, y0=0.0, mb_elev_feedback="annual", ice_thick_filter=filter_ice_border, mb_filter=None, x=None, y=None, out_dir=None, sliding_option="constant"):
        """
         Initialize the IGM_Model2D class, which runs glacier evolution simulations
         using the ice-flow solver from IGM and the mass balance model from PyGEM.

         This class inherits from the Model2D class in PyGEM's interface2d module.

         The constructor initializes the glacier model with the provided bed topography,
         initial ice thickness, grid resolution, mass balance model, and other parameters.
         It also sets up the necessary configurations for IGM and initializes the glacier state.

         The ice-flow dynamics are handled by IGM, while the mass balance is computed using
         the specified PyGEM mass balance model.

         """

        super(IGM_Model2D, self).__init__(
            bed_topo,
            init_ice_thick=init_ice_thick,
            dx=dx,
            dy=dy,
            mb_model=mb_model,
            y0=y0,
            mb_elev_feedback=mb_elev_feedback,
            ice_thick_filter=ice_thick_filter,
            mb_filter=mb_filter,
            #sliding_option=sliding_option,
            #out_dir=out_dir,
        )

        """
        Parameters
        ----------    
        bed_topo : bedrock topography (2d array)
        
        init_ice_thick : initial ice thickness (zero everywhere by default) (2d array)
        
        dx, dy : map resolution (float)
        
        mb_model : mass balance model to use for the simulation (function)
        
        y0 : the starting year (int)
        
        mb_elev_feedback : when to update the mass balance model : ’annual’, ’monthly’, ’always’
                            (’annual’ by default) (str)

        ice_thick_filter : function to apply to the ice thickness *after* each time step. (function)
        
        mb_filter : the mask of the glacier (2d array)

        sliding_option : option for the sliding coefficient : 'constant' (default), 'elevation_dependent', 'igm_inversion'

        out_dir : output directory for IGM outputs (str)
        
        """

        self.state = State()

        self.cfg = load_yaml_as_cfg(config)

        # Parameter
        self.cfl = 0.25  # hard-coded, could be added to input parameters in config.yaml later
        self.max_dt = SEC_IN_YEAR

        # Grid parameters
        self.dx = dx
        self.x = x
        self.y = y

        # Disable the training of the iceflow emulator
        # would it be possible to retrain the emulator during the simulation?
        self.cfg.processes.iceflow.retrain_iceflow_emulator_freq = 0

        # Initialize the glacier variables in the IGM model state
        self.state.thk = tf.Variable(self.ice_thick)
        self.state.usurf = tf.Variable(self.surface_h)
        self.state.smb = tf.Variable(tf.zeros_like(self.ice_thick))

        ### Define ice-flow parameters used in IGM
        # Ice rheology
        self.state.arrhenius = tf.ones_like(self.state.thk) * cfg.PARAMS["glen_a"] * SEC_IN_YEAR * 1e18  # Rate factor in Glen's flow law, Pa^-3 yr^-1

        if sliding_option == "elevation_dependent":
            #sliding coefficient scaled with bed elevation (Åkesson et al. 2018 QSR, Eq. 2)
            beta_max = 1
            z_bed = self.bed_topo
            z_low = np.min(z_bed)
            sliding_coefficient = beta_max * min(max(0, z_bed), z_bed + z_low)/max(z_bed)
            self.state.slidingco = sliding_coefficient
            ## WORK IN PROGRESS....

        elif sliding_option == "igm_inversion":
            # run an IGM inversion to obtain a spatially variable sliding coefficient
            print("Sliding option 'igm_inversion' is not yet implemented.")
            return

        elif sliding_option == "constant":
            # spatially uniform sliding coefficient
            sliding_coefficient = 0.045  # could be added to input parameters in config.yaml later
            self.state.slidingco = tf.ones_like(self.state.thk) * sliding_coefficient

        else:
            print("Please choose a valid sliding option")
            return  

        # Set grid spacing and coordinates
        self.state.dX = tf.ones_like(self.state.thk) * self.dx
        self.state.x = tf.constant(self.x)
        self.state.y = tf.constant(self.y)

        # Misc
        self.state.it = -1  # iteration counter
        self.icemask = mb_filter  # glacier mask

        # Initialize the ice flow module in IGM
        igm.processes.iceflow.iceflow.initialize(self.cfg, self.state)

        if out_dir != None:
            self.cfg.outputs.write_ncdf.output_file = out_dir + "/igm_out.nc"
        igm_write.initialize(self.cfg, self.state)

    # Time loop
    def step(self, dt):
        # recast glacier variables into igm-like variables
        self.state.thk.assign(self.ice_thick)
        self.state.usurf.assign(self.surface_h)

        # IGM: compute ubar and vbar (the x- and y-components of the depth-averaged velocity)
        igm.processes.iceflow.iceflow.update(self.cfg, self.state)

        # Compute flux divergence from IGM using upwind fluxes
        divflux = (
            compute_divflux(
                self.state.ubar,
                self.state.vbar,
                self.state.thk,
                self.state.dX,
                self.state.dX,
            )
            / SEC_IN_YEAR
        )

        # compute max speed for the CFL stability condition
        velomax = (
            max(
                tf.math.reduce_max(tf.math.abs(self.state.ubar)),
                tf.math.reduce_max(tf.math.abs(self.state.vbar)),
            ).numpy()
            / SEC_IN_YEAR
        )

        # compute the maximum possible time step that complies with CFL condition
        if velomax > 0:  # for positive velocities
            dt_cfl = min(self.cfl * self.dx / velomax, self.max_dt)
        else:  # for non-moving ice
            dt_cfl = self.max_dt

        self.state.it += 1  # increment iteration counter

        # compute effective time step
        dt_use = utils.clip_scalar(np.min([dt_cfl, dt]), 0, self.max_dt)

        # compute the surface mass balance and assign it to the 2D IGM State
        self.state.smb.assign(self.get_mb())

        # compute new ice thickness using continuity equation, using flux divergence from IGM and SMB from PyGEM
        self.state.thk.assign(tf.maximum(self.state.thk + dt_use * (self.state.smb - divflux), 0))

        self.ice_thick = self.state.thk.numpy()

        # Compute next time stamp in time loop
        self.t += dt_use

        self.state.saveresult = True

        self.state.t = tf.convert_to_tensor(self.t)

        self.state.dx = self.state.dX

        date = utils.floatyear_to_date(self.yr)
        date = (date[0], date[0])

        # Write IGM outputs
        if self._mb_current_date != date or (self._mb_current_out is None):
            igm_write.run(self.cfg, self.state)

        return dt_use

    def get_state(self):
        return self.state
