"""
Script to test running PyGEM with IGM as the ice-flow model.

Code written by: Henning Åkesson, Johannes Brunner
Inspired by run_instructed_oggm.py by Julien Jehl, Fabien Maussion, and Guillaume Jouvet
"""

# general imports
import os, sys, glob, json
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# pygem imports
# import argparse
# from pygem.run_simulation import get_parser # needed?
from pygem.setup.config import ConfigManager
from pygem.massbalance import PyGEMMassBalance
from pygem import class_climate, output


### imports for oggm and igm
from oggm import cfg, utils, workflow, tasks, shop
from oggm.cfg import G, SEC_IN_YEAR, SEC_IN_DAY
from oggm.shop import bedtopo

from oggm.core.flowline import FluxBasedModel, SemiImplicitModel

import pygem.gcmbiasadj as gcmbiasadj

import pygem.pygem_modelsetup as modelsetup
from types import SimpleNamespace

from pygem.instructed_pygem import IGM_Model2D


# from pygem.instructed_pygem import IGM_Model2D

flow_model = "IGM"  # choose either "OGGM" or "IGM"

# toDo: Compare output of "IGM" and "OGGM" options


def main():
    # PyGEM congig
    config_manager = ConfigManager(base_dir="/uio/hypatia/geofag-personlig/geohyd-staff/johanmbr/PyGEM")
    pygem_prms = config_manager.read_config()  # NOTE: ensure that your root path in ~/PyGEM/config.yaml points to
    rootpath = pygem_prms["root"]

    ### Pick glacier of choice ### (glac_no is the PyGEM variable)
    # glac_no = 08.01126 # Nigardsbreen, Norway
    glac_no = ["15.03733"]  # Great Aletsch, Switzerland
    # glac_no = 11.00897 # Hintereisferner, Austria

    # Simulation period
    startyear = 1990
    endyear = 2010

    ### Handle OGGM data paths ###
    cfg.initialize(logging_level="WARNING")
    cfg.PATHS["working_dir"] = rootpath + "/PyGEM-IGM"
    base_url = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2025.1/elev_bands/W5E5_utm/"
    gdirs = workflow.init_glacier_directories(["RGI60-" + glac_no[0]], prepro_base_url=base_url, from_prepro_level=4, prepro_border=80)
    gdir = gdirs[0]

    # Load rgi_table for PyGEM
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)
    glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[0], :]
    print(glacier_rgi_table)

    # Load a stored calibration for PyGEM
    with open("/uio/hypatia/geofag-personlig/geohyd-staff/johanmbr/PyGEM/11.01450-modelprms_dict.json", "r") as f:
        modelprms_dict = json.load(f)
    modelprms_dict = modelprms_dict["emulator"]
    # Small fix for tsnow_threshold
    modelprms_dict["tsnow_threshold"] = modelprms_dict["tsnow_threshold"][0]

    # Create a PyGEM dates table
    dates_table_ref = modelsetup.datesmodelrun(
        startyear=startyear,
        endyear=endyear,
        option_wateryear=pygem_prms["climate"]["ref_wateryear"],
    )
    gdir.dates_table = dates_table_ref

    # Load PyGEM climate data
    my_climate_data = load_climate_data_calib("ERA5", gdir.dates_table, main_glac_rgi, pygem_prms)
    gdir.historical_climate = my_climate_data

    # Get the OGGM flowline - needed by PyGEMMassBalance
    if flow_model == "OGGM":
        fls = gdir.read_pickle("model_flowlines")  # or inversion flowlines?
    elif flow_model == "IGM":
        fls = load_igm_pseudo_flowline(gdir)
    else:
        print("Please choose a valid flow model")
        return

    # Create SMB and Ice Flow models
    mbmod = PyGEMMassBalance(
        gdir,
        modelprms_dict,
        glacier_rgi_table,
        fls=fls,
        fl_id=0,
    )

    if flow_model == "IGM":
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            ds = ds.load()

        # toDo: fix this
        thick = ds.consensus_ice_thickness.where(~ds.consensus_ice_thickness.isnull(), 0)
        bed = ds.topo - thick
        mask = ds.glacier_mask.data == 1
        topo = ds.topo  # ice surface elevation

        sdmodel = IGM_Model2D(bed.data, init_ice_thick=thick.data, dx=gdir.grid.dx, mb_model=mbmod, y0=startyear, mb_filter=mask, x=ds.x, y=ds.y)

        # Run the model
        ods = sdmodel.run_until_and_store(endyear, step=1, grid=gdir.grid, print_stdout="My run")

        # Store the results
        for i in range(0, ods.ice_thickness.shape[0], 10):
            plt.imshow(ods.ice_thickness[i, :])
            plt.colorbar()
            plt.savefig(gdir.dir + "/snapshot" + str(i) + ".png")
            plt.close()
            print(i)

    else:
        ev_model = SemiImplicitModel(
            fls,
            y0=startyear,
            mb_model=mbmod,
        )

        # Run the model
        pygem_simulation = ev_model.run_until_and_store(endyear)
        print("Num of simulation years: " + str(len(pygem_simulation.volume_m3.values)))
        print("Volume evolution: " + str(pygem_simulation.volume_m3.values))

    sys.exit()


def load_igm_pseudo_flowline(gdir):
    # Load data from OGGM gdir
    bedtopo.add_consensus_thickness(gdir)

    with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
        ds = ds.load()

    thick = ds.consensus_ice_thickness.where(~ds.consensus_ice_thickness.isnull(), 0)
    igm_fls = []
    igm_fl = {}
    igm_fls.append(igm_fl)
    igm_fl["thick"] = thick.values.flatten()
    igm_fl["surface_h"] = ds.topo.values.flatten()
    igm_fl["widths_m"] = np.full_like(igm_fl["thick"], 100)
    igm_fl["dx_meter"] = np.full_like(igm_fl["thick"], 100)
    igm_fl["section"] = np.full_like(igm_fl["thick"], 100)
    # igm_fls.width_m = 100
    igm_fls = dict_to_namespace(igm_fls)
    return igm_fls


def load_climate_data_calib(ref_climate_name, dates_table, main_glac_rgi, pygem_prms, debug=False):
    # ===== LOAD CLIMATE DATA =====
    # Climate class
    assert ref_climate_name == "ERA5", "Error: Calibration not set up for " + ref_climate_name
    gcm = class_climate.GCM(name=ref_climate_name)
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


# Helpers
def dict_to_namespace(obj):
    """Recursively convert dicts (or list of dicts) to SimpleNamespace objects."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    else:
        return obj


main()
