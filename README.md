# Adaptive Regularized Rnsemble for Evolving Data Stream Classification
This is the repository for the Adaptive Regularized Ensemble (ARE) algorithm. 

### Abstract
Extracting knowledge from data streams requires fast incremental algorithms that are able to handle unlimited
processing and ever-changing data with finite memory. A strategy for this challenge is the use of ensembles
owing to their ability to tackle concept drift and achieve highly accurate predictions. However, ensembles
often require a lot of computational resources. In this study, we propose a novel ensemble-based classification
algorithm for data streams, Adaptive Regularized Ensemble (ARE), with low demand for computational
resources. The algorithm combines strategies that contribute to high prediction accuracy using only incorrectly
classified instances into the training step, random-sized feature subspace for each ensemble element and
classifier selection for final ensemble voting. After an extensive experimental study, ARE exhibited high
predictive performance and outperformed state-of-the-art ensembles on data streams for real and synthetic
datasets while requiring a low processing time and memory consumption.

## Datasets
The datasets and synthetic data streams are available in the `\dataset` directory.

## Using ARTE
Download the pre-compiled JAR files and run:
```
java -javaagent:sizeofag-1.0.4.jar -cp moa-custom-ARE-1.0.jar moa.DoTask "EvaluatePrequential -l moa.classifiers.meta.AdaptiveRegularizedEnsemble -s generators.AgrawalGenerator -i 1000000 -f 10000"
```
Alternatively, import the project source code directly into [MOA](https://github.com/Waikato/moa).

## Citation
```
@article{paim2024adaptive,
title = {Adaptive regularized ensemble for evolving data stream classification},
author = {Aldo M. Paim and Fabrício Enembreck},
journal = {Pattern Recognition Letters},
volume = {180},
pages = {55-61},
year = {2024},
issn = {0167-8655},
doi = {https://doi.org/10.1016/j.patrec.2024.02.026}
}
```
