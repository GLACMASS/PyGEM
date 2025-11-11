"""
This class provides an interface to a 2d model in PyGEM,
and handles initialization of 2d variables

Code written by: Johannes Brunner, Henning Åkesson
Based on sia2d.py in OGGM, written by Fabien Maussion
"""

import numpy as np
from numpy import ix_
import xarray as xr
import os

from oggm import cfg, utils
from oggm.cfg import G, SEC_IN_YEAR, SEC_IN_DAY

from types import SimpleNamespace


def filter_ice_border(ice_thick):
    """Sets the ice thickness at the border of the domain to zero."""
    ice_thick[0, :] = 0
    ice_thick[-1, :] = 0
    ice_thick[:, 0] = 0
    ice_thick[:, -1] = 0
    return ice_thick


class Model2D(object):
    """Interface to a distributed model"""

    def __init__(self, bed_topo, init_ice_thick=None, dx=None, dy=None, mb_model=None, y0=0.0, glen_a=None, mb_elev_feedback="annual", ice_thick_filter=filter_ice_border, mb_filter=None):
        """Create a new 2D model from gridded data.

        Parameters
        ----------
        bed_topo : 2d array
            bed topography
        init_ice_thick : 2d array (optional)
            initial ice thickness (default is zero everywhere)
        dx : float
            map resolution (m)
        dy : float
            map resolution (m)
        mb_model : oggm.core.massbalance model
            the mass balance model to use for the simulation
        y0 : int
            the starting year
        glen_a : float
            Glen's flow law parameter A
        mb_elev_feedback : str (default: 'annual')
            when to update the mass balance model ('annual', 'monthly', or
            'always')
        ice_thick_filter : func
            function to apply to the ice thickness *after* each time step.
            See filter_ice_border for an example. Set to None for doing nothing.
        mb_filter : ndarray
            2d array indicating the mask where positive mb is allowed
            (useful to allow only specific glaciers to grow)
        """

        # Mass balance
        self.mb_elev_feedback = mb_elev_feedback
        self.mb_model = mb_model
        self.mb_filter = mb_filter

        # Set rate factor in Glen's flow law to a default value, if not specified
        if glen_a is None:
            glen_a = cfg.PARAMS["glen_a"]
        self.glen_a = glen_a

        # Set the grid resolution in y to dx if not specified
        if dy is None:
            dy = dx

        # Initialize grid
        self.dx = dx
        self.dy = dy
        self.dxdy = dx * dy  # calculate area of one grid cell

        # Initialize time
        self.y0 = None
        self.t = None
        self.reset_y0(y0)

        # Ice thickness filter
        self.ice_thick_filter = ice_thick_filter

        # Initialize bed topography, ice thickness and grid dimensions
        self.bed_topo = bed_topo
        self.ice_thick = None
        self.reset_ice_thick(init_ice_thick)
        self.ny, self.nx = bed_topo.shape

    @property
    def mb_model(self):
        return self._mb_model

    @mb_model.setter
    def mb_model(self, value):
        # We need a setter because the MB func is stored as an attr too
        _mb_call = None
        if value:
            if self.mb_elev_feedback in ["always", "monthly"]:
                _mb_call = value.get_monthly_mb
            elif self.mb_elev_feedback in ["annual", "never"]:
                _mb_call = value.get_annual_mb
            else:
                raise ValueError("mb_elev_feedback not understood")
        self._mb_model = value
        self._mb_call = _mb_call
        self._mb_current_date = None
        self._mb_current_out = dict()
        self._mb_current_heights = dict()

    def reset_y0(self, y0):
        """Reset the initial model time"""
        self.y0 = y0
        self.t = 0

    def reset_ice_thick(self, ice_thick=None):
        """Reset the ice thickness"""
        if ice_thick is None:
            ice_thick = self.bed_topo * 0.0
        self.ice_thick = ice_thick.copy()

    @property
    def yr(self):
        return self.y0 + self.t / SEC_IN_YEAR

    @property
    def area_m2(self):
        return np.sum(self.ice_thick > 0) * self.dxdy

    @property
    def volume_m3(self):
        return np.sum(self.ice_thick * self.dxdy)

    @property
    def volume_km3(self):
        return self.volume_m3 * 1e-9

    @property
    def area_km2(self):
        return self.area_m2 * 1e-6

    @property
    def surface_h(self):
        return self.bed_topo + self.ice_thick

    def get_mb(self, year=None):
        """Get the mass balance at the requested height and time.

        Optimized so that no mb model call is necessary at each step.
        """

        if year is None:
            year = self.yr

        # Do we have to optimise?
        if self.mb_elev_feedback == "always":
            _mb = self._mb_call(self.surface_h.flatten(), year=year)
            _mb = _mb.reshape((self.ny, self.nx))
            if self.mb_filter is not None:
                _mb[~self.mb_filter & (_mb > 0)] = 0
            return _mb

        date = utils.floatyear_to_date(year)
        
        if self.mb_elev_feedback == "annual":
            # ignore month changes
            date = (date[0], date[0])

        if self._mb_current_date != date or (self._mb_current_out is None):
            # We need to reset all
            self._mb_current_date = date

            fls = create_pseudo_flowline(self.ice_thick, self.surface_h)

            _mb = self._mb_call(self.surface_h.flatten(), year=self.yr, fl_id=0, fls=fls)
            _mb = _mb.reshape((self.ny, self.nx))
            if self.mb_filter is not None:
                _mb[~self.mb_filter & (_mb > 0)] = 0
            self._mb_current_out = _mb

        # Store current SMB
        return self._mb_current_out

    def step(self, dt):
        """Advance one step."""
        raise NotImplementedError

    def run_until(self, y1, stop_if_border=False):
        """Run until a selected year.
        Parameters
        ----------
        y1 : int
            the end year (determined within each iteration of the time loop
            in run_until_and_store)
        stop_if_border : bool
            if True, stop the run if the ice thickness at the border exceeds 10m
            (default: False)
        """

        # calculate the total time to run (in seconds)
        t = (y1 - self.y0) * SEC_IN_YEAR

        # time loop
        while self.t < t:
            self.step(t - self.t)  # step until the end time
            # check if the ice thickness at the border exceeds 10m
            if stop_if_border:
                if np.any(self.ice_thick[0, :] > 10) or np.any(self.ice_thick[-1, :] > 10) or np.any(self.ice_thick[:, 0] > 10) or np.any(self.ice_thick[:, -1] > 10):
                    raise RuntimeError("Glacier exceeds boundaries")
            # apply ice thickness filter if defined (
            if self.ice_thick_filter is not None:
                self.ice_thick = self.ice_thick_filter(self.ice_thick)

        if np.any(~np.isfinite(self.ice_thick)):
            raise FloatingPointError("nan in numerical solution.")

    def run_until_equilibrium(self, rate=0.001, ystep=5, max_ite=200):
        """Run until an equilibrium is reached (can take a while.
        From OGGM sia2d.py - not used in PyGEM now, but could be useful later."""

        ite = 0
        was_close_zero = 0
        t_rate = 1
        while (t_rate > rate) and (ite <= max_ite) and (was_close_zero < 5):
            ite += 1
            v_bef = self.volume_m3
            self.run_until(self.yr + ystep)
            v_af = self.volume_m3
            if np.isclose(v_bef, 0.0, atol=1):
                t_rate = 1
                was_close_zero += 1
            else:
                t_rate = np.abs(v_af - v_bef) / v_bef
        if ite > max_ite:
            raise RuntimeError("Did not find equilibrium.")

    def run_2D_until_and_store(self, ye, step=2, run_path=None, grid=None, print_stdout=False, stop_if_border=False):
        """Run until a selected year and store the output in a NetCDF file.
        Parameters
        ----------
        ye : int
            the end year
        step : int
            how often to store the output (default: every 2 years)
        run_path : str
            the path to the NetCDF output file (optional)
        grid : oggm.core.grid.Grid
            the OGGM grid to use for the output (optional)
        print_stdout : str or False
            if a string is given, print the progress to stdout with this
            string as a prefix (e.g. 'My run')
        stop_if_border : bool
            if True, stop the run if the ice thickness at the border exceeds 10m
            (default: False)
        Returns
        -------
        run_ds : xarray.Dataset
            the dataset containing the output
        """

        # array of years to store the output
        yrs = np.arange(np.floor(self.yr), np.floor(ye) + 1, step)

        # arrays to store the model output (nyrs,ny,nx)
        out_thick = np.zeros((len(yrs), self.ny, self.nx))
        out_usurf = np.zeros((len(yrs), self.ny, self.nx))
        #out_smb = np.zeros((len(yrs), self.ny, self.nx))
        out_velsurf_mag = np.zeros((len(yrs), self.ny, self.nx))
        out_vol = np.zeros(len(yrs))
        out_area = np.zeros(len(yrs))

        # time loop
        for i, yr in enumerate(yrs):
            # Progress message every 10 years: max ice thickness
            if print_stdout and (yr / 10) == int(yr / 10):
                print("{}: year {} of {}, " "max thick {:.1f}m".format(print_stdout, int(yr), int(ye), self.ice_thick.max()), end="\r", flush=True)
            self.run_until(yr, stop_if_border=stop_if_border)
            # store the model variables in the output arrays
            out_thick[i, :, :] = self.ice_thick
            # out_usurf[i, :, :] = self.surface_h
            #out_velsurf_mag[i, :, :] = self.velsurf_mag
            #out_smb[i, :, :] = self.smb
            out_vol[i] = self.volume_km3
            out_area[i] = self.area_km2

        run_ds = grid.to_dataset() if grid else xr.Dataset()
        run_ds["ice_thickness"] = xr.DataArray(out_thick, dims=["time", "y", "x"], coords={"time": yrs})
        # run_ds["usurf"] = xr.DataArray(out_usurf, dims=["time", "y", "x"], coords={"time": yrs})
        #run_ds["velsurf_mag"] = xr.DataArray(out_velsurf_mag, dims=["time", "y", "x"], coords={"time": yrs})
        # run_ds["smb"] = xr.DataArray(out_smb, dims=["time", "y", "x"], coords={"time": yrs})
        run_ds["bed_topo"] = xr.DataArray(self.bed_topo, dims=["y", "x"])
        run_ds["vol"] = xr.DataArray(out_vol, dims=["time"], coords={"time": yrs})
        # run_ds["area"] = xr.DataArray(out_area, dims=["time"], coords={"time": yrs})

        # write output dataset to netcdf
        if run_path is not None:
            # remove existing file if it exists
            if os.path.exists(run_path):
                os.remove(run_path)
            # save dataset to netcdf
            run_ds.to_netcdf(run_path)

        return run_ds


def create_pseudo_flowline(thk_2d, surface_h_2d):
    # This function creates an object that mimics a OGGM flowline from 2D input data
    # No information is lost as all grid points are included in the pseudo flowline object
    igm_fls = []
    igm_fl = {}
    igm_fls.append(igm_fl)
    igm_fl["thick"] = thk_2d.flatten()
    igm_fl["surface_h"] = surface_h_2d.flatten()
    igm_fl["widths_m"] = np.full_like(igm_fl["thick"], 100)
    igm_fl["dx_meter"] = np.full_like(igm_fl["thick"], 100)
    igm_fl["section"] = np.full_like(igm_fl["thick"], 100)
    # igm_fls.width_m = 100
    igm_fls = dict_to_namespace(igm_fls)
    return igm_fls


# Helpers
def dict_to_namespace(obj):
    # We need this function for the pseudo-flowline to work inside PyGEM-MB
    """Recursively convert dicts (or list of dicts) to SimpleNamespace objects."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    else:
        return obj
