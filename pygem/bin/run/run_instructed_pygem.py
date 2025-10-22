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

### imports for oggm and igm
from oggm import cfg, utils, workflow, tasks
from oggm.cfg import G, SEC_IN_YEAR, SEC_IN_DAY
from pygem.instructed_pygem import IGM_Model2D


# instantiate ConfigManager
config_manager = ConfigManager()
#config_manager = ConfigManager(base_dir="/uio/hypatia/geofag-felles/projects/glacmass/henning/pygem/PyGEM")

# read the config
pygem_prms = config_manager.read_config()   # NOTE: ensure that your root path in ~/PyGEM/config.yaml points to
                                            # the appropriate location. If any errors occur, check this first.
rootpath=pygem_prms['root']


### Pick glacier of choice ### (glac_no is the PyGEM variable, rgi_ids is the OGGM variable)
# glac_no = 08.01126 # Nigardsbreen, Norway
# glac_no = 11.01450 # Great Aletsch, Switzerland
# glac_no = 11.00897 # Hintereisferner, Austria
rgi_ids = ["RGI08-01.126"]  # Nigardsbreen
# rgi_ids = ["RGI60-11.01450"]  # Aletsch
# rgi_ids = ['RGI60-11.00897']  # Hitereisferner

### Handle data paths ###
cfg.initialize(logging_level="WARNING")
cfg.PATHS["working_dir"] = utils.gettempdir(dirname="PyGEM-IGM", reset=True)
base_url = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/exps/igm_v1/"
gdirs = workflow.init_glacier_directories(
    rgi_ids, prepro_base_url=base_url, from_prepro_level=2, prepro_border=30
)
gdir = gdirs[0]

# Load data from OGGM gdir
with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
    ds = ds.load()
thick = ds.consensus_ice_thickness.where(~ds.consensus_ice_thickness.isnull(), 0)
bed  = ds.topo - thick
mask = ds.glacier_mask.data == 1
topo = ds.topo # ice surface elevation


# Define SMB model in PyGEM
# glac_no = rgi_ids # rename variable for PyGEM
mbmod = PyGEMMassBalance(glac_no=rgi_ids,
                           hemi='nh',
                           use_daily_climate=False,
                           smb_model_type='linear',
                           temp_correction=0.0,
                           precip_scaling=1.0)

# Time settings
startyear = 2000
endyear = 2020

# Define model
ev_model = IGM_Model2D(
    bed=bed.data,
    init_ice_thick=thick.data,
    mb_model=mbmod,
    y0=startyear,
    mb_filter=mask,
    dx=gdir.grid.dx,
    dy=gdir.grid.dy,
    x=gdir.grid.x,
    y=gdir.grid.y,
)
# Glen A is not needed in ev_model? since it is defined in instructed_pygem.py
# Remark: equivalent instructed_oggm code
# sdmodel = IGM_Model2D(bed.data, init_ice_thick=thick.data, dx=gdir.grid.dx,
#                      mb_model=mb, y0=0, mb_filter=mask, x=ds.x, y=ds.y)

# Run the model
diag = ev_model.run_until_and_store(
    until_year=endyear,
    grid=gdir.grid,
    print_stdout="PyGEM-IGM run"
)
# Remark: equivalent instructed_oggm code
# ods = sdmodel.run_until_and_store(100, grid=gdir.grid, print_stdout="My run")

#print('shape of volume:', ev_model.mb_model.glac_wide_volume_annual.shape, diag.volume_m3.shape)
                            
# Store glacier-wide annual outputs for analysis
ev_model.mb_model.glac_wide_volume_annual = (diag.volume_m3.values)
ev_model.mb_model.glac_wide_area_annual = (diag.area_m2.values)
ev_model.mb_model.glac_wide_length_annual = (diag.length_m.values)


########################

# Plot the results
savepath = "/uio/hypatia/geofag-felles/projects/glacmass/henning/pygem/Output/instructed_pygem/plots"
for i in range(0,diag.ice_thickness.shape[0],10):
    plt.imshow(diag.ice_thickness[i, :])
    plt.colorbar()
    plt.savefig(savepath + "/snapshot" + str(i) + ".png")
    #plt.savefig("snapshot" + str(i) + ".png")
    plt.close()
    print(i)
