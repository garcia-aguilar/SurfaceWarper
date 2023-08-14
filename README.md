# SurfaceWarper
Shape-evolving software for triangulated elastic surfaces subject to spontaneous curvature

There are three energy minimization algorithms available, all based on gradient descent. The mechanical energy implementes is a sum of a stretching energy and a bending energy. The stretching enery are the standard Hookean springs in the mesh edges and the bending energy implemented is the Fischer Energy (DOI: 10.1051/jp2:1993230). A discrete implementation for triangulated meshes is used, which allows for an analytical calculation of the total energy gradient. This analytical expression is implemented in the code and used for all three algorithms.  The calculation is available in the publications below. 

Authors: I. Garcia-Aguilar and S. Zwaan

More details on the methods and theoretical framework can be found at (also contains the analytical calculation of the energy gradient):
"On shape and elasticity: Bio-sheets, curved crystals, and odd droplets". I. Garcia-Aguilar, Doctoral thesis (2022).
https://scholarlypublications.universiteitleiden.nl/handle/1887/3458390.

An application of this software on biomolecular assemblies of tubulin can be found at:
I. Garcia-Aguilar, Zwaan, S., Giomi, L. "Polymorphism in tubulin assemblies: a mechanical model”.   Physical Review Research 5 (2), (2023)
https://doi.org/10.1103/PhysRevResearch.5.023093

![KeyImage_400x400](https://github.com/garcia-aguilar/SurfaceWarper/assets/126492604/8122c3fa-016b-404b-ba18-77ad97c09957)

## Repo Contents

### SurfaceWarper/
Main simulation scripts prefixed with: Coordinate_Minimizer_*.py

*Setup.py: sets up a new configuration from scratch, instead of reading one off an exisiting *.config files. 
*New.py: reads from an initial, equilibrated configuration. It is the start of the run simulation. Get system snapshots with a higher frequency 
*Cont.py: similarly to the above, it reads from an existing *config file, but has an offset number, from the last timestep stored. This is used to continue a simulation that has not reached the equilibrium configuration and has to be continued from a given checkpoint. 

### Analysis/
The code running the simulations creates 3D plots using matplotlib to track the simulation progress. To visualize the results, this directory contains scripts to visualize the output configuration files with Mayavi, including heat maps of the local fields (mean curvature, deviatoric curvature, area, Gaussian curvature, hooke stress)

### Examples/
Coming up: In/Output example files and visualization

### Old/ 
Coming up: Older versions of this code. 

## How to use it in loose terms

The code can setup a new configuration from a list of implemented geometries:
'S: Hexagonal Sheet'
'C: Cylinder'
'L: Long cylinder (with cap)'
'H: Helical Ribbon'
'O: Hemisphere'
'I: Icosahedron'

Through a run routine (see runThis()), the vertex coordinates are evolved according to an energy-minimization algorithm for a given number of "timesteps". There are three available algorithms: 1) Adaptative gradient descent [g_AG()], which is the one performing the best although can have long-tails towards convergence, 2) Simple gradient descent with a fixed step [g(lmin=step)] or 3) simple gradient descent with a step calculated through line minimization [g()]. For all cases, the full energy gradient is used, by implementing the analytically-calculated expressions in the code.

The evolution of the system is tracked through output plots (described below), and output files with values of the energy or convergence parameters, both updated after certain number of steps. Additionally, given a predetermined frequency, snapshots of the system are stored (*.config files) and an image of the configuration (\*.png files). The simulation ends when a convergence criterium is reached or the number of running steps is completed. 

The resulting surface in the *.config files can be plotted using the script in Analysis, which can be adapted to read specific file names. It does use an implemented Config() class to store quantitative surface data. 

Tip: summary() prints a summary of user functions

### Input and Output files

((Coming up...))
