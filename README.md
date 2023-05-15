# Repository containing source code for my MSc-thesis Developing a deep learning forecasting system for high-resolution and short-term sea ice concentration

This repository contain all code developed for my thesis-work. 

## Contents
### Data preparation scripts

The following directories contain scripts which regrids data onto the model domain, and structures the files such that they are ready to be processed into training samples

**AROME_ARCTIC_regrid**

**RawIceChart_dataset**

**OSI_SAF_regrid**

The prepared data is then structured into samples uing the script 

**PrepareDataset/createHDF.sh**

To create normalization_constants, which are used during training to min-max noramlize the data, run 

**PrepareDataset/computeNormalization.sh**

### Deep learning system

The deep learning system is contained in the file

**SimpleUnet/unet.py**

The deep learning system which utilizes cumulative contours is the *MultiOutputUNET*

The deep learning system is initiated using 

**SimpleUnet/RunModel/run_model.sh**

### Verification metrics

The developed verification metrics are located in

**verification_metrics/verification_metrics.py**

They include

* find_ice_edge
* find_ice_edge_from_fraction
* calculate_distance
* IIEE
* contourAreaDistribution
* minimumDistanceToIceEdge
* sea_ice_extent
  
### Intermodel-comparisson

Scripts for setting up the intermodel-comparisson are located in 

**PhysicalModels**

The directory contains several fetch-xxxxx scripts which perform preparatory work. *compareProducts.sh* can be run to perform the comparisson when all data is prepared.

---

All source code is supplied as is, thanks for your consideration!