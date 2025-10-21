"""
Script to test running PyGEM with IGM as the ice-flow model.

Code written by: Henning Åkesson, Johannes Brunner
Inspired by run_instructed_oggm.py by Julien Jehl, Fabien Maussion, and Guillaume Jouvet
"""


### imports ###
import os, sys, glob, json
# pygem imports
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
#config_manager = ConfigManager(base_dir="/uio/hypatia/geofag-felles/projects/glacmass/henning/pygem/PyGEM")

# read the config
pygem_prms = config_manager.read_config()   # NOTE: ensure that your root path in ~/PyGEM/config.yaml points to
                                            # the appropriate location. If any errors occur, check this first.
rootpath=pygem_prms['root']


### Pick glacier of choice ###
glac_no = 08.01126 # Nigardsbreen, Norway
# glac_no = 11.01450 # Great Aletsch, Switzerland
# glac_no = 11.00897 # Hintereisferner, Austria

### Handle data paths ###


# Define SMB model
##mb = LinearMassBalance(ela_h=2900.0)

# Define IGM model
##sdmodel = IGM_Model2D(bed.data, init_ice_thick=thick.data, dx=gdir.grid.dx,
##                    mb_model=mb, y0=0, mb_filter=mask, x=ds.x, y=ds.y)

# Run the model
##ods = sdmodel.run_until_and_store(100, grid=gdir.grid, print_stdout="My run")

#run_simulation -rgi_glac_number {glac_no} -ref_startyear 2000 -ref_endyear 2019 -option_dynamics IGM

########################

# Plot the results
#for i in range(0,ods.ice_thickness.shape[0],10):
#    plt.imshow(ods.ice_thickness[i, :])
#    plt.colorbar()
#    plt.savefig("/uio/hypatia/geofag-felles/projects/glacmass/henning/igm-examples/instructed_oggm/plots/snapshot" + str(i) + ".png")
#    #plt.savefig("snapshot" + str(i) + ".png")
#    plt.close()
#    print(i)
