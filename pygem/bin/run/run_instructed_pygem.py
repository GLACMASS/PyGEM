"""
Script to test run PyGEM with IGM 2d as the ice-flow model.
The script also provides the option to use OGGM SIA 1d flowline as ice-flow model for comparison.

Code written by: Henning Åkesson, Johannes Brunner
Inspired by run_instructed_oggm.py by Julien Jehl, Fabien Maussion, and Guillaume Jouvet
"""

# general imports
import subprocess
import os, sys, glob, json
import numpy as np
import xarray as xr
from datetime import datetime

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
#   - Glen_A:
#     -> add option to set Glen_A based on PyGEM calibration?
#   - Sliding parameter
#   -> IGM: add possbility to run with spatially variable sliding parameter, either set manually as a function of elevation (Åkesson et al. 2018),
#     or add capability to run an IGM inversion to get sliding parameter field
# Think about different PyGEM calibrations

# Think about using monthly OGGM TI model (MB), it should easy to integrate


### Pick glacier of choice ###
#glac_no = ["08.01126"]  # Nigardsbreen, Norway
glac_no = ["08.00312"]  # Storbreen, Norway
#glac_no = ["11.01450"]  # Aletsch glacier
#glac_no = ["11.00897"] # Hintereisferner, Austria

#glac_no = ["RGI2000-v7.0-C-11-01450"]  # Aletsch glacier
# glac_no = ["rgi2000-v7.0-c-08-01742"] # hardangerjøkulen, Norway
# glac_no = ["rgi2000-v7.0-g-08-03147"] # Storbreen, Norway

# Simulation period - be careful about initial thickness date!
ref_startyear = 2000
ref_endyear = 2020

# Ice-flow model options
flow_model = "IGM"  # choose either "OGGM" or "IGM"
slidingoption = "constant"  # sliding coefficient for IGM: 'constant', 'elevation_dependent', 'igm_inversion'

# SMB calibration options
isruncalibration = True # True: run a calibration. False: use SMB parameter values a stored calibration
option_calibration = "HH2015"

# Data options
rgi_version = "RGI6"  # RGI version to use: RGI6 (RGI6.0) or RGI7 (RGI7.0)
rgi_product = "70C" # 'Only for RGI7: '70C' for glacier complexes or '70G' for individual glacier (see https://www.glims.org/rgi_user_guide/products/glacier_complex_product.html)
thickness_product = "consensus_ice_thickness"  # thickness product to use: 'consensus_ice_thickness' (Farinotti et al. 2019) or 'millan_ice_thickness' (Millan et al 2022)

# Inputs and config
climate_data_path = "/uio/hypatia/geofag-felles/projects/glacmass/data/PyGEM_input/climate_data/ERA5/"
igm_config_file = "/uio/hypatia/geofag-felles/projects/glacmass/henning/igm-examples/instructed_oggm/params.yaml"
#pygem_config_dir = "/uio/hypatia/geofag-personlig/geohyd-staff/johanmbr/PyGEM"
pygem_config_dir = "/uio/hypatia/geofag-felles/projects/glacmass/henning/pygem/PyGEM"
# working_dir = "/uio/hypatia/geofag-felles/projects/glacmass/henning/pygem/PyGEM/PyGEM_input"
working_dir = "/uio/hypatia/geofag-felles/projects/glacmass/henning/pygem/PyGEM/PyGEM-IGM"

# Output directories
# oggm_out_dir = "/uio/hypatia/geofag-personlig/geohyd-staff/johanmbr/PyGEM/PyGEM-IGM/outputs/OGGM"
# igm_out_dir = "/uio/hypatia/geofag-personlig/geohyd-staff/johanmbr/PyGEM/PyGEM-IGM/outputs/IGM"
oggm_out_dir = "/uio/hypatia/geofag-felles/projects/glacmass/henning/pygem/PyGEM/PyGEM-IGM/outputs/OGGM"
igm_out_dir = "/uio/hypatia/geofag-felles/projects/glacmass/henning/pygem/PyGEM/PyGEM-IGM/outputs/IGM"


def main():
    # PyGEM config
    config_manager = ConfigManager(base_dir=pygem_config_dir)
    pygem_prms = config_manager.read_config()  # NOTE: ensure that your root path in ~/PyGEM/config.yaml points to right dir
    rootpath = pygem_prms["root"]

    print(rootpath)
   # working_dir = rootpath + pygem_prms['oggm']['oggm_gdir_relpath']
    print(working_dir)


    ### Handle OGGM data paths ###
    cfg.initialize(logging_level="WARNING")
    cfg.PATHS["working_dir"] = working_dir
    # Load data from RGI version
    if rgi_version == "RGI6":
        base_url = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2025.1/elev_bands/W5E5_utm/"
        gdirs = workflow.init_glacier_directories(["RGI60-" + glac_no[0]], prepro_base_url=base_url, from_prepro_level=4, prepro_border=80)
    elif rgi_version == "RGI7":
        base_url = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/2025.6/elev_bands_w_data/"
        gdirs = workflow.init_glacier_directories(glac_no, prepro_base_url=base_url, from_prepro_level=2, prepro_border=10, prepro_rgi_version=rgi_product)
    else:
        print("Please choose a valid RGI version")
        return
    
    
    gdir = gdirs[0]

    # Add bed topography
    if thickness_product == "millan_ice_thickness":
        #FIXME to be added
        print("Millan thickness product not yet fully implemented")
    if thickness_product == "consensus_ice_thickness":
        bedtopo.add_consensus_thickness(gdir)

    # Load rgi_table for PyGEM
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)
    glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[0], :]
    print(glacier_rgi_table)

    # Perform calibration, or specify file with stored calibration
    if isruncalibration == True:
        print("Running calibration for glacier RGI " + glac_no[0])
        glac_no_value = float(glac_no[0])  # Extracts the first element and converts it to float

        try:
            # Call calibration script and pass arguments
            result = subprocess.run(
                [sys.executable, pygem_config_dir + '/pygem/bin/run/run_calibration.py', 
                '-rgi_glac_number', str(glac_no_value), 
                '-ref_startyear', str(ref_startyear), 
                '-ref_endyear', str(ref_endyear), 
                '-option_calibration', option_calibration],
                check=True, 
                text=True, 
                capture_output=True
            )
            
            # Print the output
            print("Output from run_calibration.py:")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print("An error occurred while running run_calibration.py:")
            print(e.stderr)

    else: 
        # Nothing to be done, load stored calibration below
        print ("Using stored calibration for glacier RGI " + glac_no[0])
        # calibration_data_file = "/uio/hypatia/geofag-felles/projects/glacmass/data/PyGEM_input/calibration/11/11.01450-modelprms_dict.json"

    # Set calibration data file path to the output of the calibration, or the stored calibration data
    rgi_region = glac_no[0].split(".")[0] # Get RGI region in use
    calibration_data_file = rootpath + "/Output/calibration/" + rgi_region + "/" + glac_no[0] + "-modelprms_dict.json"

    # Load calibrated SMB parameters for PyGEM
    with open(calibration_data_file, "r") as f:
        calib_params = json.load(f)
    # calib_params = calib_params["emulator"]
    calib_params = calib_params[option_calibration]
    # Small fix for tsnow_threshold
    calib_params["tsnow_threshold"] = calib_params["tsnow_threshold"][0]

    # Create a PyGEM dates table
    dates_table_ref = modelsetup.datesmodelrun(
        startyear=ref_startyear,
        endyear=ref_endyear,
        option_wateryear=pygem_prms["climate"]["ref_wateryear"],
    )
    gdir.dates_table = dates_table_ref

    # Load PyGEM ERA5 climate data
    climate_data = load_climate_data("ERA5", gdir.dates_table, main_glac_rgi, pygem_prms)
    gdir.historical_climate = climate_data

    # Load glacier data and create geometry
    if flow_model == "OGGM":
        # Get the OGGM flowline - needed by PyGEMMassBalance
        # Do it using the same thickness product as used in IGM
        workflow.execute_entity_task(tasks.elevation_band_flowline, gdirs, bin_variables=[thickness_product])
        workflow.execute_entity_task(tasks.fixed_dx_elevation_band_flowline, gdirs, bin_variables=[thickness_product])
        tasks.compute_downstream_line(gdir)
        tasks.compute_downstream_bedshape(gdir)
        tasks.init_present_time_glacier(gdir, use_binned_thickness_data=thickness_product)
        fls = gdir.read_pickle("model_flowlines")  # or inversion flowlines?

    elif flow_model == "IGM":
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            ds = ds.load()
        thick = ds[thickness_product].where(~ds[thickness_product].isnull(), 0)
        surface_h = ds.topo
        bed = ds.topo - thick
        mask = ds.glacier_mask.data == 1
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
        # Create evolution model (see also run_instructed_oggm.py example from IGM)
        distributed_ev_model = IGM_Model2D(bed.data, init_ice_thick=thick.data, config=igm_config_file, dx=gdir.grid.dx, mb_model=mbmod, y0=ref_startyear, mb_filter=mask, x=ds.x, y=ds.y, out_dir=igm_out_dir, sliding_option=slidingoption)

        # Run the model
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S") # get current time, for naming output files
        igm_simulation_output = distributed_ev_model.run_2D_until_and_store(ref_endyear, run_path=None, step=1, grid=gdir.grid, print_stdout="My run")
        # igm_simulation_output = distributed_ev_model.run_2D_until_and_store(ref_endyear, run_path=igm_out_dir + f"/igm_out_{current_time}.nc", step=1, grid=gdir.grid, print_stdout="My run")
        print(igm_simulation_output.vol)
        # np.savetxt(igm_out_dir + f"/IGM_vol_evolution_{current_time}.txt", igm_simulation_output.vol, fmt="%.4f")

        igm_state_obj = distributed_ev_model.get_state()
        # print(igm_state_obj)

    if flow_model == "OGGM":
        # Create evoution model
        ev_model = SemiImplicitModel(
            fls,
            y0=ref_startyear,
            mb_model=mbmod,
        )

        # Run the model
        ev_model.run_until_and_store(ref_endyear, fl_diag_path=oggm_out_dir + "/fl_diagnostic.nc", geom_path=oggm_out_dir + "/geom_diagnostic.nc", diag_path=oggm_out_dir + "/diagnostic.nc")


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
