# SOM_lc
# Variable Star Classification with Self-Organizing Maps

This repository contains the Python code developed for my Master's thesis in Astrophysics, focused on the **classification of variable stars using Self-Organizing Maps (SOMs)**.

## Overview

The project investigates the use of **unsupervised machine learning** techniques to classify astronomical variable stars from their light curves.

The dataset consists of light curves extracted from **7 DECam fields toward the Large Magellanic Cloud**, observed with the DECam camera mounted on the **Victor M. Blanco 4-m Telescope** at Cerro Tololo Inter-American Observatory.

The main goal is to train a Self-Organizing Map to identify patterns in the light curves and evaluate its ability to distinguish different classes of variable stars.

## Main Workflow

The pipeline includes:

* Loading and preprocessing astronomical light curves
* Feature extraction and normalization
* Training of a **Self-Organizing Map**
* Visualization of the trained SOM
* Analysis of the **U-Matrix**
* Analysis of the neuron hit distribution
* Classification of variable-star classes
* Evaluation using **confusion matrices, completeness and purity**
* Visualization and analysis of classification results

## Variable Star Classes

The analysis includes several classes of astronomical sources, including:

* RR Lyrae (RRab, RRc)
* δ Scuti
* Classical Cepheids
* Type II Cepheids
* Eclipsing binaries (EA, EB, EW)
* Other variable sources

## Technologies

The project was developed mainly in **Python**, using:

* NumPy
* Matplotlib
* PyTorch
* Scikit-learn
* Custom Self-Organizing Map implementation
* FITS/astronomical data analysis tools
## Results

The SOM successfully identified several meaningful structures in the dataset, showing particularly good performance for:

* RR Lyrae stars
* RRc & δ Scuti stars
* Type II Cepheids
* Microlensing events

The classification of **classical Cepheids** and especially **eclipsing binaries** was more challenging due to the similarity between their light-curve features and the complexity of their distributions in the feature space.

