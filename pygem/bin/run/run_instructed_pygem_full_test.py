### imports ###
import os, sys, glob, json
import subprocess
from pygem.setup.config import ConfigManager


calib_run_file = "/uio/hypatia/geofag-personlig/geohyd-staff/johanmbr/repos/PyGEM/pygem/bin/run/run_calibration.py"  # Johannes
simulation_run_file = "/uio/hypatia/geofag-personlig/geohyd-staff/johanmbr/repos/PyGEM/pygem/bin/run/run_simulation.py"  # Johannes


# Individual glacier simulation(s)
glac_no = 08.00312  # Storbreen, Norway
# glac_no = 08.01126 # Nigardsbreen, Norway
# glac_no = 11.01450  # Great Aletsch, Switzerland
# glac_no = 11.00897 # Hintereisferner, Austria
# glac_no = 15.03733  # Khumbu Glacier, Nepal
# glac_no = [11.00897, 15.03733] # Hintereisferner, Austria and Khumbu Glacier, Nepal

# Regional simulation - FIXME: not implemented yet
# glac_no = ''
# rgi_region01 = '06' # Specify to run entire RGI region, otherwise set to None
# rgi_region02 = 'all'
# min_glac_area_km2 = 200 # Minimum glacier area in km2 for inclusion when specifying rgi_region01

# Calibration options
isruncalibration = False
isprintcalibrationparameters = False
option_calibration = "HH2015"
ref_startyear = 2000
ref_endyear = 2019
ref_climate_name = "ERA5"

# Dynamics options
option_dynamics = "IGM"

# Simulation options
isrunsimulationhistoric = True
isrunsimulationfuture = False

climate_names = ["ERA5", "CESM2"]  # historic, future
start_years = [2000, 2020]  # historic, future
end_years = [2009, 2100]  # historic, future

sim_climate_scenario = "ssp245"


def main():

    # Perform calibration, or specify file with stored calibration
    if isruncalibration == True:
        print("--- Running calibration...")
        try:
            # Call calibration script and pass arguments
            result = subprocess.run(
                [
                    sys.executable,
                    calib_run_file,
                    "-rgi_glac_number",
                    str(glac_no),
                    "-ref_startyear",
                    str(ref_startyear),
                    "-ref_endyear",
                    str(ref_endyear),
                    "-option_calibration",
                    option_calibration,
                ],
                # '-rgi_region01', str(rgi_region01),
                # '-rgi_region02', str(rgi_region02),
                check=True,
                text=True,
                capture_output=True,
            )

            # Print the output
            print("Output from run_calibration.py:")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print("An error occurred while running run_calibration.py:")
            print(e.stderr)

    else:
        # Nothing to be done, load stored calibration below
        glacier_str = format(glac_no, ".5f")
        print("Using stored calibration for glacier RGI " + glacier_str)

    if isprintcalibrationparameters:
        # check the output - the parameter dictionary output should now have an `HH2015` key
        glacier_str = format(glac_no, ".5f")
        reg = glacier_str.split(".")[0].zfill(2)
        config_manager = ConfigManager()
        pygem_prms = config_manager.read_config()
        calib_path = f"{pygem_prms['root']}/Output/calibration/{reg}/{glacier_str}-modelprms_dict.json"
        # /uio/hypatia/geofag-felles/projects/glacmass/henning/pygem/Output/calibration/11/11.01450-modelprms_dict.json

        with open(calib_path, "r") as f:
            modelprms_dict = json.load(f)

        print(modelprms_dict[option_calibration])

    if isrunsimulationhistoric == True:
        print("--- Running simulation...")
        try:
            # Call simulation script and pass arguments
            result = subprocess.run(
                [
                    sys.executable,
                    simulation_run_file,
                    "-rgi_glac_number",
                    str(glac_no),
                    "-sim_startyear",
                    str(start_years[0]),
                    "-sim_endyear",
                    str(end_years[0]),
                    "-sim_climate_name",
                    climate_names[0],
                    "-option_calibration",
                    option_calibration,
                    "-option_dynamics",
                    option_dynamics,
                ],
                # '-rgi_region01', str(rgi_region01),
                # '-rgi_region02', str(rgi_region02),
                check=True,
                text=True,
                capture_output=True,
            )

            # Print the output
            print("Output from run_simulation.py:")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print("An error occurred while running run_simulation.py:")
            print(e.stderr)

    else:
        # Nothing to be done, no simulation being run
        print("No historic simulation run.")

    if isrunsimulationfuture == True:
        print("--- Running future simulation...")
        try:
            # Call simulation script and pass arguments
            result = subprocess.run(
                [
                    sys.executable,
                    simulation_run_file,
                    "-rgi_glac_number",
                    str(glac_no),
                    "-sim_startyear",
                    str(start_years[1]),
                    "-sim_endyear",
                    str(end_years[1]),
                    "-sim_climate_name",
                    climate_names[1],
                    "-sim_climate_scenario",
                    sim_climate_scenario,
                    "-option_calibration",
                    option_calibration,
                    "-option_dynamics",
                    option_dynamics,
                ],
                # '-rgi_region01', str(rgi_region01),
                # '-rgi_region02', str(rgi_region02),
                check=True,
                text=True,
                capture_output=True,
            )

            # Print the output
            print("Output from run_simulation.py:")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print("An error occurred while running run_simulation.py:")
            print(e.stderr)

    else:
        # Nothing to be done, no simulation being run
        print("No future simulation run.")


if __name__ == "__main__":
    main()
