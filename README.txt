Riparian buffer optimization

This folder contains the cleaned version of the code we used for the thesis work.
The purpose of the script is to create variable-width riparian buffers around
watercourses by balancing ecological value and economic value.


Files in this folder

- run_buffer_optimization.py
  Main script.
- project_settings.example.json
  Example settings file with paths, constants, and lambda scenarios.


1. What the script does

The script:
- reads the input rasters and the watercourse shapefile
- aligns everything to a common grid
- creates a candidate corridor around the watercourse
- calculates ecological and economic values for each pixel
- solves the optimisation with graph cut
- saves output maps and summary files for several lambda scenarios


2. Input data needed

The script needs:
- a canopy height raster (CHM)
- a depth-to-water raster (DTW)
- a species raster
- a watercourse shapefile with line geometry

By default, the species raster is assumed to use:
- 1 = broadleaf
- 2 = pine
- 3 = spruce

If another dataset uses different species codes, that has to be changed in the
settings file.


3. Python packages needed

we used Python for this, and these packages are needed:
- numpy
- rasterio
- scipy
- matplotlib
- geopandas
- shapely
- maxflow

If something is missing, it can usually be installed with:

python -m pip install geopandas rasterio scipy matplotlib shapely PyMaxflow


4. How to run it

Step 1
Copy the example settings file and create your own version:

project_settings.example.json

For example:

project_settings.json

Step 2
Open the JSON file and update:
- the input file paths
- the output folder
- the constants if needed
- the lambda scenarios if needed

Step 3
Run the script:

python run_buffer_optimization.py --config project_settings.json

If running it from the main project folder instead of this subfolder, use:

python teacher_ready_code\run_buffer_optimization.py --config teacher_ready_code\project_settings.json


5. Important settings

The easiest place to change things is the JSON file.

Important model settings are:
- resolution_m
- max_buffer_width_m
- lambda_ecology_weight
- mu_boundary_weight
- sun_altitude_deg
- sun_azimuth_deg
- zoom_regions

The JSON file also has:

scenario_lambdas

That block controls which scenarios are run in one go.

Example:

"scenario_lambdas": {
  "scenario_1": 0.0,
  "scenario_2": 1.0,
  "scenario_3": 5.0
}

In this model:
- low lambda = more harvesting
- high lambda = more protection


6. Output structure

The output folder is organised by scenario.

Example:

outputs/
  balanced/
  balanced_low/
  balanced_high/
  max_ecology/
  max_economy/

Inside each scenario folder there are:
- buffer_width.tif
- harvest_mask.tif
- overview.png
- summary.txt
- zoom_1.png
- zoom_2.png
- zoom_3.png
- zoom_4.png

Each zoom also gets its own folder with separate layers:
- combined.png
- chm_background.png
- protected_buffer.png
- harvested_area.png
- watercourse.png

wearranged it like this so it is easier to compare scenarios and also easier to
use the different layers later in figures.


7. Using another dataset

The code is not only for Asa, but another dataset has to be similar enough.

It should still work if:
- the rasters are valid and can be aligned to one grid
- the watercourse data is line geometry
- the species classes are defined correctly in the settings

It will probably need adjustments if:
- the species codes are different
- there are more than three species groups
- the water data is polygons instead of lines
- CHM or DTW represent something different from the assumptions used here

So the general method is reusable, but the constants are still case-specific.


8. Common problems

Problem: import error at the start
- Usually means a Python package is missing.

Problem: no water pixels found
- Usually means the shapefile does not line up with the raster CRS.

Problem: strange species output
- Usually means the species raster codes do not match the settings file.

Problem: buffers look too wide or too narrow
- Check max_buffer_width_m, lambda values, and mu value.


9. Short workflow summary

The normal workflow is:
- update the JSON settings
- check that the file paths are correct
- check the species codes
- choose the lambda scenarios
- run the script
- inspect the scenario folders and summary files

That is basically the full process used here.
