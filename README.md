# Code and Data for: Machine learning of EEG spectra classifies unconsciousness during GABAergic anesthesia study

This is the code and data repository for Abel, Badgeley, ... Brown, "Machine learning of EEG spectra classifies unconsciousness during GABAergic anesthesia," in review. This contains all data used in training and testing the classification models, as well as all code for generating the relevant figures. The drug dose and timing data and the patient characteristics are excluded to comply with relevant data sharing and privacy restrictions. Furthermore, the features generated by the CNN are not included due to large filesize, and are instead available for direct transfer upon request. Thus, this repository differs only in excluding these data--all code will otherwise run and return the results of the manuscript.

For a full description, please see our manuscript: [link]

# Table of Contents
* [Dependencies](#Dependencies)
* [Files](#Files)
* [Data](#Data)
* [Analysis of data](#Analysis-of-data)
* [Authors](#Authors)
* [Funding](#Funding)

# Dependencies
This code is written for Python 3.7. Dependencies for instantiating a conda environment for this project are in the `Setup/classifier_env.yml` file. To do so, run:
```
conda env create -f Setup/classifier_env.yml
conda active classifier_env
```
In addition to the dependencies listed in the file, this also requires installation of our modified version of the `hmmlearn` Python package. This modified version is available at https://github.com/benyaminmk/hmmlearn. The original version is available at https://hmmlearn.readthedocs.io/.


# Files and Scripts
A short summary of each script is as follows:

* `fig3_crossvalidation.py`: Performs crossvalidation as explained in the manuscript and trains the three best models. Generates Figure 3.
* `fig4s1s5_volunteer_test.py`: Performs tests of the best models on the held-out volunteer test cohort. Generates Figures 4, S1, S4.
* `fig5fig7_OR_test.py`: Performs tests on the propofol and sevoflurane test cohorts. Generates Figures 5, 7.
* `fig6s2_orcases`: Generates visualizations of classifier performance for the three cases from the OR test cohort that we show in the manuscript. May, of course, be changed to examine other cases. Generates Figures 6, S2.

The `LocalResources` directory contains files for data processing and analysis used in this manuscript.
The `Results` directory contains trained models, crossvalidation results, and inference tables resulting from the model training and validation paradigm.

# Data
All data used in training and testing the classification models should be placed in the `Data` directory. Subdirectories split the subject data into `Volunteer` (from Purdon et al. PNAS 2013) and `OR` cases. The data are arranged by anonymized subject id, `_f` (multitaper frequencies), `_l` (labels 0=unconscious, 1=conscious), `_PCs` (first principal components of the multitaper spectrogram, calculated as explained in the manuscript), `_Sdb` (multitaper spectral power estimates in decibels), `_t` (time recordings corresponding to each spectrogram). OR data also has an `_EEGquality` file that indicates where signal drops out with a 0. The `rx_sorted_case_ids.yml` classifies each OR case as `propofol` or `sevoflurane`.

All data are available at physionet (link TBD).


# Authors
Code written by John Abel, Marcus Badgeley, Benyamin Meschede-Krasa, Gabriel Schamberg, Kimaya Lecamwasam. Under the same copyright as the publication. For questions, contact John Abel at `jhabel01 at gmail dot com`.
