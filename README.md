# Behavioral Partial Least Square Correlation (PLSC) to analyze the role of different components in the CPM model

## Background

This repository contains the code  for a project conducted under the supervision of members of the [MIP:lab](https://miplab.epfl.ch/).
The aim is to apply a multivariate statistical method (PLSC) to study the relationship between brain activity from fMRI films and emotional experience, focusing on both discrete emotions and appraisal.  

Behavioral PLSC is a widely used technique for neuroimaging, as highlighted by [review article by Krishnan](https://pubmed.ncbi.nlm.nih.gov/20656037/). This package has been largely adapted from the [Matlab toolbox](https://github.com/valkebets/myPLS-1) and draws inspiration from a [Python interface for partial least squares (PLS) analysis](https://github.com/valkebets/myPLS-1).

However, while the Matlab toolbox has a significant number of tools dedicated to integrating neuroimaging-specific paradigms and aims to optimally relate neuroimaging to behavioral data for different types of neuroimaging data formats, the current Python code has been adapted to suit the specific needs of the current project, including data loading, pre-processing, PLSC analysis, and plots. 

## Implementation
`analysis_PLS.py` contains the main function initiates the BehavPLS classes and run the PLSC analysis and result plotting. More specifically, this files takes as an input argument a condig file that contains all the parameters of the BehavPLS class to run and load the results into pkl format. See the directory `configs/`. 

`BehavPLS.py` contains class  to wrap the dataset (brain & behavior data). 

`compute.py` contains functions for the pre-processing of the data as well as for the PLS methods.

`plot.py` contains plotter function to visualize the results.

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
```
in coming
```
## Acknowledgments
Many thanks to Elenor Morgenroth for providing the data, feedback, and in general a great supervision and Alessandra Griffa for guiding me through the static PLS background.

