# Behavioral Partial Least Square Correlation (PLSC) to analyze the role of different components in the CPM model

## Background

This repository contains the code  for a project conducted under the supervision of members of the [MIP:lab](https://miplab.epfl.ch/).
The aim is to apply a multivariate statistical method (PLSC) to study the relationship between brain activity from fMRI films and emotional experience, focusing on both discrete emotions and appraisal.  

Behavioral PLSC is a widely used technique for neuroimaging, as highlighted by [review article by Krishnan](https://pubmed.ncbi.nlm.nih.gov/20656037/). This package has been largely adapted from the [Matlab toolbox](https://github.com/valkebets/myPLS-1) and draws inspiration from a [Python interface for partial least squares (PLS) analysis](https://github.com/valkebets/myPLS-1).

However, while the Matlab toolbox has a significant number of tools dedicated to integrating neuroimaging-specific paradigms and aims to optimally relate neuroimaging to behavioral data for different types of neuroimaging data formats, the current Python code has been adapted to suit the specific needs of the current project, including data loading, pre-processing, PLSC analysis, and plots. 
