# Design philosophy

The idea is to design loosely connected modules. The package consists
of different modules that can be used independently up to a point.

To install the package run:

`python setup.py build_ext --inplace`
TODO: update this
## Overview
The main division can be found as follows:
- Modeling
- Methods
- Plotting
Please note that a unittest is written for nearly every different main file
# Modeling
Models form a general model that can be found in models.py. Through inheritance
one can define their own modes. Note that only a minimal set of properties are required
for the different models

Current implemented models:
- Ising

![ising](Notebooks/ising_low.gif)
- Potts model

![potts](Notebooks/4statepotts.gif)
## Modules:
- Models: contains various models
- Utils: various different statistical, plotting and IO related functions
- Toolbox: the 'engine' responsible for Monte-Carlo methods

# Methods
This module contains the methods for the different information theoretical measures.
Currently, IDT for source and sinks are implemented.

## Modules:
- information.py: IDT and sampling methods (MCMC) and other miscellaneous functions

# plotting
This part of the model is responsible for plotting various parts of the results.


## Modules
- plotting.py contains functions to show graph (including adjacency) next to fits for
computing the IDT as well as computing the impact of the nudge on node distribution using
the Hellinger distance.


# TODO:
- General: clean up code; remove redundant information.
- move fastIsing in the models.py for cleaner outlook
- Expand the toolbox to be able to be used by experimentalists
