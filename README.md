# Behavioral Partial Least Square Correlation (PLSC) to analyze the role of different components in the CPM model
(still in progress)
## Background

This repository contains the code  for a project conducted under the supervision of members of the [MIP:lab](https://miplab.epfl.ch/).
We aim to probe the role of appraisal in emotion experience by conducting three PLSC analyses:
- **1** using discrete emotions and brain activity (fMRI).
- **2** using appraisal items and brain activity (fMRI).
- **3** using appraisals and discrete emotions. 

The overall aim is to compare each model in order to assess the relationship between appraisal and discrete emotions at the level of brain activation during the experience of an emotional event


Behavioral PLSC is a widely used technique for neuroimaging, as highlighted by [review article by Krishnan](https://pubmed.ncbi.nlm.nih.gov/20656037/). This package has been largely adapted from the [Matlab toolbox](https://github.com/valkebets/myPLS-1) and draws inspiration from a [Python interface for partial least squares (PLS) analysis](https://github.com/valkebets/myPLS-1).

However, while the Matlab toolbox has a significant number of tools dedicated to integrating neuroimaging-specific paradigms and aims to optimally relate neuroimaging to behavioral data for different types of neuroimaging data formats, the current Python code has been adapted to suit the specific needs of the current project, including data loading, pre-processing, PLSC analysis, and plots. 

## Implementation
`analysis_PLS.py` contains the main function initiates the BehavPLS classes and run the PLSC analysis for two first model. More specifically, this files takes as an input argument a condig file that contains all the parameters of the BehavPLS class to run and load the results into pkl format. See the directory `configs/`. 

`Emotion_PLS.py` contains the main function initiates the third PLS models (with Appraisals & Discrete items). 

`BehavPLS.py` contains class  to wrap the dataset (brain & behavior data). 


`compute.py` contains functions for the pre-processing of the data as well as for the PLS methods.

`plot.py` contains plotter function to visualize the results.

`visualization.ipynb` to visualize the results.

## Libraries
[numpy](https://numpy.org/)\
[sklearn](https://scikit-learn.org/stable/)\
[nilearn](https://nilearn.github.io/stable/index.html)\
[nibabel](https://nipy.org/nibabel/)
[pickle](https://docs.python.org/3/library/pickle.html)\
[scipy](https://scipy.org/)\
[pandas](https://pandas.pydata.org/)\
[matplotlib](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)\

## How to use
Example config files are available in the `configs/` directory.\
Simply run the following (as an example):

python3 analysis_PLS.py --config  config/test.yaml


## Acknowledgments
Many thanks to Elenor Morgenroth for providing the data, feedback, and in general a great supervision and Alessandra Griffa for guiding me through the static PLS background.
