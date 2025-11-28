### imports ###
import os, sys, glob, json
import subprocess

# pygem imports
from pygem.setup.config import ConfigManager

pygem_config_dir = "/uio/hypatia/geofag-felles/projects/glacmass/henning/pygem/PyGEM"


# instantiate ConfigManager
#config_manager = ConfigManager()
config_manager = ConfigManager(base_dir=pygem_config_dir)
# read the config
pygem_prms = config_manager.read_config()   # NOTE: ensure that your root path in ~/PyGEM/config.yaml points to
                                            # the appropriate location. If any errors occur, check this first.
rootpath=pygem_prms['root']


# glac_no = 08.01126 # Nigardsbreen, Norway
glac_no = 11.01450 # Great Aletsch, Switzerland
# glac_no = 11.00897 # Hintereisferner, Austria

ref_startyear=2000
ref_endyear=2019

isruncalibration = False
isprintcalibrationparameters = False
option_calibration = "HH2015"

option_dynamics = "IGM"
isrunsimulation = True

def main():

    # Perform calibration, or specify file with stored calibration
    if isruncalibration == True:
        print("--- Running calibration...")
        try:
            # Call calibration script and pass arguments
            result = subprocess.run(
                [sys.executable, pygem_config_dir + '/pygem/bin/run/run_calibration.py', 
                '-rgi_glac_number', str(glac_no), 
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
        #print ("Using stored calibration for glacier RGI " + glac_no)
        calibration_data_file = "/uio/hypatia/geofag-felles/projects/glacmass/data/PyGEM_input/calibration/11/11.01450-modelprms_dict.json"

    if isprintcalibrationparameters:
        # check the output - the parameter dictionary output should now have an `HH2015` key
        glacier_str = format(glac_no, ".5f")
        reg = glacier_str.split('.')[0].zfill(2)
        calib_path = f"{pygem_prms['root']}/Output/calibration/{reg}/{glacier_str}-modelprms_dict.json"
        #/uio/hypatia/geofag-felles/projects/glacmass/henning/pygem/Output/calibration/11/11.01450-modelprms_dict.json

        with open(calib_path, 'r') as f:
            modelprms_dict = json.load(f)

        print(modelprms_dict[option_calibration])

    if isrunsimulation == True:
        print("--- Running simulation...")
        try:
            # Call simulation script and pass arguments
            result = subprocess.run(
                [sys.executable, pygem_config_dir + '/pygem/bin/run/run_simulation.py', 
                '-rgi_glac_number', str(glac_no), 
                '-ref_startyear', str(ref_startyear), 
                '-ref_endyear', str(ref_endyear),
                '-option_calibration', option_calibration,
                '-option_dynamics', option_dynamics],
                check=True, 
                text=True, 
                capture_output=True
            )
            
            # Print the output
            print("Output from run_simulation.py:")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print("An error occurred while running run_simulation.py:")
            print(e.stderr)

    else: 
        # Nothing to be done, no simulation being run
        print("No simulation run.")

if __name__ == "__main__":
    main()