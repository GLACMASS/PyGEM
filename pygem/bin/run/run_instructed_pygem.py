"""
Script to test running PyGEM with IGM as the ice-flow model.

Code written by: Henning Åkesson, Johannes Brunner
Inspired by run_instructed_oggm.py by Julien Jehl, Fabien Maussion, and Guillaume Jouvet
"""

# general imports
import os, sys, glob, json
import numpy as np
import xarray as xr

# pygem imports
# import argparse
from pygem.setup.config import ConfigManager
from pygem.massbalance import PyGEMMassBalance
from pygem import class_climate, output


### imports for oggm and igm
from oggm import cfg, utils, workflow, tasks, shop
from oggm.cfg import G, SEC_IN_YEAR, SEC_IN_DAY
from oggm.shop import bedtopo

from oggm.core.flowline import SemiImplicitModel

import pygem.pygem_modelsetup as modelsetup

from pygem.instructed_pygem import IGM_Model2D
from pygem.interface2d import create_pseudo_flowline

# !!! Small change in flowline.py (OGGM) required for storing PyGEM MB output information with OGGM (mb_model_debug_refreeze bool error) !!!
# Use: if type(v) is bool:
#         v = str(v)
#         ds.attrs['mb_model_{}'.format(k)] = v

# Next steps:
# - Add timeseries output for IGM
# - Think about Glen_A and Sliding_F in IGM and OGGM
# - Think about different PyGEM calibrations
# - Think about also using monthly OGGM TI model (MB), it should be easy to integrate


flow_model = "IGM"  # choose either "OGGM" or "IGM"

# Inputs and config
climate_data_path = "/uio/hypatia/geofag-felles/projects/glacmass/data/PyGEM_input/climate_data/ERA5/"
calibration_data_file = "/uio/hypatia/geofag-felles/projects/glacmass/data/PyGEM_input/calibration/11.01450-modelprms_dict.json"
igm_config_file = "/uio/hypatia/geofag-felles/projects/glacmass/henning/igm-examples/instructed_oggm/params.yaml"
pygem_config_dir = "/uio/hypatia/geofag-personlig/geohyd-staff/johanmbr/PyGEM"

# Outputs
oggm_out_dir = "/uio/hypatia/geofag-personlig/geohyd-staff/johanmbr/PyGEM/PyGEM-IGM/outputs/OGGM"
igm_out_dir = "/uio/hypatia/geofag-personlig/geohyd-staff/johanmbr/PyGEM/PyGEM-IGM/outputs/IGM"


### Pick glacier of choice ###
# glac_no = 08.01126 # Nigardsbreen, Norway
glac_no = ["11.01450"]  # Aletsch glacier
# glac_no = 11.00897 # Hintereisferner, Austria

# Simulation period - be careful about initial thickness date!
startyear = 1979
endyear = 1985


def main():
    # PyGEM config
    config_manager = ConfigManager(base_dir=pygem_config_dir)
    pygem_prms = config_manager.read_config()  # NOTE: ensure that your root path in ~/PyGEM/config.yaml points to right dir
    rootpath = pygem_prms["root"]

    ### Handle OGGM data paths ###
    cfg.initialize(logging_level="WARNING")
    cfg.PATHS["working_dir"] = rootpath + "/PyGEM-IGM"
    base_url = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2025.1/elev_bands/W5E5_utm/"
    gdirs = workflow.init_glacier_directories(["RGI60-" + glac_no[0]], prepro_base_url=base_url, from_prepro_level=4, prepro_border=80)
    gdir = gdirs[0]
    bedtopo.add_consensus_thickness(gdir)

    # Load rgi_table for PyGEM
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)
    glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[0], :]
    print(glacier_rgi_table)

    # Load a stored calibration for PyGEM
    # ------ !!! Aletsch Calib !!! ---------------------
    with open(calibration_data_file, "r") as f:
        calib_params = json.load(f)
    calib_params = calib_params["emulator"]
    # Small fix for tsnow_threshold
    calib_params["tsnow_threshold"] = calib_params["tsnow_threshold"][0]

    # Create a PyGEM dates table
    dates_table_ref = modelsetup.datesmodelrun(
        startyear=startyear,
        endyear=endyear,
        option_wateryear=pygem_prms["climate"]["ref_wateryear"],
    )
    gdir.dates_table = dates_table_ref

    # Load PyGEM ERA5 climate data
    climate_data = load_climate_data("ERA5", gdir.dates_table, main_glac_rgi, pygem_prms)
    gdir.historical_climate = climate_data

    if flow_model == "OGGM":
        # Get the OGGM flowline - needed by PyGEMMassBalance
        # Do it on the same thickness field as used in IGM
        workflow.execute_entity_task(tasks.elevation_band_flowline, gdirs, bin_variables=["consensus_ice_thickness"])
        workflow.execute_entity_task(tasks.fixed_dx_elevation_band_flowline, gdirs, bin_variables=["consensus_ice_thickness"])
        tasks.init_present_time_glacier(gdir, use_binned_thickness_data="consensus_ice_thickness")
        fls = gdir.read_pickle("model_flowlines")  # or inversion flowlines?

    elif flow_model == "IGM":
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            ds = ds.load()
        thick = ds.consensus_ice_thickness.where(~ds.consensus_ice_thickness.isnull(), 0)
        surface_h = ds.topo
        fls = create_pseudo_flowline(thick.values, surface_h.values)

    else:
        print("Please choose a valid flow model")
        return

    # Create SMB model
    mbmod = PyGEMMassBalance(
        gdir,
        calib_params,
        glacier_rgi_table,
        fls=fls,
        fl_id=0,
    )

    # Create Ice flow model and run
    if flow_model == "IGM":
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            ds = ds.load()

        # Create model like in the run_instructed_oggm.py example from IGM
        # But use PyGEMMassBalance instead of LinearMassBalance
        thick = ds.consensus_ice_thickness.where(~ds.consensus_ice_thickness.isnull(), 0)
        bed = ds.topo - thick
        mask = ds.glacier_mask.data == 1
        distributed_ev_model = IGM_Model2D(bed.data, init_ice_thick=thick.data, config=igm_config_file, dx=gdir.grid.dx, mb_model=mbmod, y0=startyear, mb_filter=mask, x=ds.x, y=ds.y, out_dir=igm_out_dir)

        # Run the model
        igm_simulation_output = distributed_ev_model.run_2D_until_and_store(endyear, run_path=gdir.dir + "/igm_out.nc", step=1, grid=gdir.grid, print_stdout="My run")
        print(igm_simulation_output.vol)
        np.savetxt(gdir.dir + "/../IGM_vol_evolution.txt", igm_simulation_output.vol, fmt="%.4f")

        igm_state_obj = distributed_ev_model.get_state()
        print(igm_state_obj)

    if flow_model == "OGGM":

        ev_model = SemiImplicitModel(
            fls,
            y0=startyear,
            mb_model=mbmod,
        )

        # Run the model
        ev_model.run_until_and_store(endyear, fl_diag_path=oggm_out_dir + "/fl_diagnostic.nc", geom_path=oggm_out_dir + "/geom_diagnostic.nc", diag_path=oggm_out_dir + "/diagnostic.nc")


def load_climate_data(ref_climate_name, dates_table, main_glac_rgi, pygem_prms, debug=False):
    # ===== LOAD CLIMATE DATA =====
    # Climate class
    assert ref_climate_name == "ERA5", "Error: Calibration not set up for " + ref_climate_name
    gcm = class_climate.GCM(name=ref_climate_name)
    gcm.var_fp = climate_data_path
    gcm.fx_fp = climate_data_path

    # Air temperature [degC]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table, verbose=debug)
    if pygem_prms["mb"]["option_ablation"] == 2 and ref_climate_name in ["ERA5"]:
        gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn, main_glac_rgi, dates_table, verbose=debug)
    else:
        gcm_tempstd = np.zeros(gcm_temp.shape)
    # Precipitation [m]
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table, verbose=debug)
    # Elevation [m asl]
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Lapse rate [degC m-1] (always monthly)
    gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(
        gcm.lr_fn,
        gcm.lr_vn,
        main_glac_rgi,
        dates_table,
        upscale_var_timestep=True,
        verbose=debug,
    )

    glac = 0

    # Add climate data to glacier directory
    historical_climate = {
        "elev": gcm_elev[glac],
        "temp": gcm_temp[glac, :],
        "tempstd": gcm_tempstd[glac, :],
        "prec": gcm_prec[glac, :],
        "lr": gcm_lr[glac, :],
    }

    return historical_climate


main()
