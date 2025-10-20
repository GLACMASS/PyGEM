"""
    This class contains methods to run 2D simulations of glacier evolution,
    using the ice-flow module of IGM as a solver for ice dynamics and ice thickness evolution. 
    The mass balance model is taken from PyGEM.    
    
    Code written by: Henning Åkesson, Johannes Brunner
    Based on instructed_oggm.py by Julien Jehl, Fabien Maussion, and Guillaume Jouvet 
"""

import numpy as np
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) #Prevent TensorFlow from allocating all GPU memory
import os

from oggm import cfg, utils # Do we need something from PyGEM's config manager too?
from oggm.cfg import G, SEC_IN_YEAR, SEC_IN_DAY 

import igm
from pygem.interface2d import Interface2d #equivalent of oggm.core.sia2d import Model2D

class IGM_Model2D(Interface2d):
    """Class to run a 2D simulation of glacier evolution using IGM as ice flow solver."""
    def filter_ice_border(ice_thick):
        """Sets the ice thickness at the border of the domain to zero."""
        ice_thick[0, :] = 0
        ice_thick[-1, :] = 0
        ice_thick[:, 0] = 0
        ice_thick[:, -1] = 0
        return ice_thick

    def __init__(
        self,
        bed_topo,
        init_ice_thick=None,
        dx=None,
        dy=None,
        mb_model=None, # check what PyGEM uses  
        y0=0.0,
        mb_elev_feedback="annual", #FIXME - seems like PyGEM doesn't have this funtionality, needs added?   
        ice_thick_filter=filter_ice_border,
        mb_filter=None, #FIXME - check what PyGEM uses
        x=None,
        y=None,
    ):
        """
        Initialize the IGM_Model2D class.
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
        
        """
        # Get the IGM model state
        self.state = igm.State()

        self.cfg = igm.EmptyClass()  
        self.cfg.processes = igm.load_yaml_as_cfg(os.path.join("conf","processes","iceflow.yaml"))

        # Time parameters
        self.cfl = 0.25 # hard-coded, could be added to input parameters in config.yaml later
        self.max_dt = SEC_IN_YEAR

        # Grid parameters
        self.dx = dx
        self.dy = dy
        self.x = x
        self.y = y

        # Disable the training of the iceflow emulator
        self.cfg.processes.iceflow.retrain_iceflow_emulator_freq = 0

        # Initialize the glacier variables in the IGM model state
        self.state.thk = tf.Variable(self.ice_thick)
        self.state.usurf = tf.Variable(self.surface_h)
        self.state.smb = tf.Variable(tf.zeros_like(self.ice_thick))

        # Define ice-flow parameters used in IGM
        self.state.arrhenius = (tf.ones_like(self.state.thk) * cfg.PARAMS["glen_a"] * SEC_IN_YEAR * 1e18) # Rate factor in Glen's flow law, Pa^-3 yr^-1
        sliding_coefficient = 0.045 # Sliding coefficient, default 0.045. Hard-coded for now - add to input parameters in config.yaml later
        self.state.slidingco = tf.ones_like(self.state.thk) * sliding_coefficient

        # Set grid spacing and coordinates
        self.state.dY = tf.ones_like(self.state.thk) * self.dy
        self.state.dX = tf.ones_like(self.state.thk) * self.dx
        self.state.x = tf.constant(self.x)
        self.state.y = tf.constant(self.y)

        # Misc
        self.state.it = -1 # iteration counter
        self.icemask = mb_filter # glacier mask

        # Initialize the ice flow module in IGM
        igm.processes.iceflow.iceflow.initialize(self.cfg, self.state)

    # Time loop
    def step(self, dt):
        # recast glacier variables into igm-like variables

        self.state.thk.assign(self.ice_thick)
        self.state.usurf.assign(self.surface_h)

        # IGM: compute ubar and vbar (the x- and y-components of the depth-averaged velocity)
        igm.processes.iceflow.iceflow.update(self.cfg, self.state)

        # Compute flux divergence from IGM using upwind fluxes
        divflux = (
            igm.processes.utils.compute_divflux(
                self.state.ubar,
                self.state.vbar,
                self.state.thk,
                self.state.dX,
                self.state.dY,
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
        if velomax > 0: # for positive velocities
            dt_cfl = min(self.cfl * self.dx / velomax, self.max_dt)
        else: # for non-moving ice
            dt_cfl = self.max_dt

        self.state.it += 1 # increment iteration counter

        # compute effective time step
        dt_use = utils.clip_scalar(np.min([dt_cfl, dt]), 0, self.max_dt)

        # compute SMB, as defined by get_mb method in Interface2D
        self.state.smb.assign(self.get_mb()) #FIXME check if we need to adapt this to PyGEM

        # compute new ice thickness using continuity equation, using flux divergence from IGM and SMB from PyGEM
        self.state.thk.assign(
            tf.maximum(self.state.thk + dt_use * (self.state.smb - divflux), 0)
        )

        self.ice_thick = self.state.thk.numpy()

        # Compute next time stamp in time loop
        self.t += dt_use

        return dt_use