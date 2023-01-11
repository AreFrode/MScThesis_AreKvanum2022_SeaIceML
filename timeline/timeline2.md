# thesis timeline ver 2.

### I want to answer the following research questions

* Can a deep learning system resolve regional sea ice concentration for high resolution, short lead time forecasts

* How does a high resolution, short lead time unet forecasting system resolve the translation and accumulation of sea ice compared to a physical based model

* In what sense can a deep learning model be explainable / made transparent to explain the statistical reasoning behind the physical decision-making

## Nov
* [x] 1.Nov submit iicwg poster abstract 

* Continue writing the methodology
    + Write about image segmentation, image to image prediction, CNNs -> unet
    + Model development,
    + History of CNNs in Sea Ice forecasting, advances in the field. 

* Develop and write about verification metrics
    + [x] Ice edge length
    + [x] IIEE
    + SPS / SPS-length (Would the SPS make sense to compute for the multi-output model?)
    + [x] minimumDIstanceToIceEdge
    + Fractions Skill Score (Derived in [Melsom, 2019], applied along the ice edge)
    + Fractional Brier Skill Score (As learned in GEO4902) (Spatial verification scheme, threshold SIC hits and misses)


* Write about the datasets
    + AROME Arctic
    + Sea Ice Charts
    + OsiSAF
    + (NextSIM)
    
    + Explain why the datasets are used, what are their pros and cons, how are the data gathered, structured, (analyzed) to create the final raw product
    + both pre and post processed
    + write about the datapipeline

    + Especially for the Ice Charts, explain why they are chosen as the target. Note that they are a high resolution product, with a high certainty as a consequence of manual interpretation. Relevant for operational usage. (Statistical downscaling)

* Start writing a preliminary intro

* Run experiments, generate results
    - When GPU starts working again

* [x] Retry getting mixed precision to work, opens up possibility by reducing memory footprint

* Experiment with hyperparameter tuning

* Develop new verification metrics (e.g. # of times forecast beat target for IIEE)
    + [x] Count # times model better than persistance implemented

* Try to include date as a predictor for the unet, similarly to [Grigoryev, 2022], fear that including a non-convolutional connection, (i.e. include date through some dense connection) will break the spatial inbvariance of the network. This is not a simple parameter to add, as the unet is a fully invariant image to image model.

## Des
* Generate results which are independent of the 2022 test period
  
* Continue writing the methodology, data, pipeline, intro, verification etc...
  
* Continue tuning the hyperparameters, running experiments

* Search the literature for unet explainability, e.g. ablation study
    + Ideally, this can be used to inspect which features contributes where

* Prepare results from a Physical Based Sea Ice model as well as a pipeline for the unet output for a comparison study (NextSIM)

* Return to the multi-class, single-output model, compare performance to single-class, multi-output model. 
    + Develop a variation of the CE-loss which is weighted according to the misclassified class distance to the target class.

* (Technical), load training data onto a RAMDISK on lustre, compare trainig time. (Inspect if loading data from disk is causing a bottleneck). 

* (moved this down to december, gpuback up and running) If during Nov (14 - 20) there is no response on the GPU - matter, start working on preparing the increased icechart data from Nick in preparation of training the model with more samples, but without T2M



## Jan
* generate results for 2022 test set
  
* Write discussion

* Run experiments
    + Example, how does the model resolve the seasonal dependance of sea ice? Is the model skill dependant on the season also? Would it be interesting to train the model on a certain season only to improve skill for that season?
  
* <mark> Send methodology and data to Cyril and Malte (maybe Jean) for review by mid January.</mark>

## Feb

* Write discussion
  
## March
* If poster is going smooth, more discussion
  
* Prepare poster (Summarize results, research poster formatting and contents)
  
* 21-22-23
   - Poster-session 11th International Workshop on Sea Ice Modelling, Assimilation, Observations, Predictions and Verification

* * <mark> Send discussion chapter to Cyril, Jean and Malte either after or before IICWG, based on poster progress </mark>

## April
* Work on loose ends discovered while working on the poster

* Start writing the conclusion, rweork introduction in light of thesis
    
* Revise based on feedback
  
* Finish introduction by April

* <mark> 2/3 of April send thesis to Cyril, Jean and Malte for revision </mark>

## May
  
* Revise revise REVISE
  
* Finish conclusion
  
* Submit thesis (15. May 2023)
