# vToxiNet
## Overview
This repository contains the code and processed data used in the study:  
**vToxiNet: biologically constrained deep learning for interpretable hepatotoxicity prediction**  

vToxiNet is a biologically constrained deep learning framework for predicting drug-induced hepatotoxicity by integrating
multiple data modalities, including chemical descriptors, molecular initiating event (MIE) assay responses, 
transcriptomic profiles, and the Reactome pathway hierarchy.  

The model is designed to support both predictive performance and mechanistic interpretation.

## Data
The repository includes processed datasets used for model development and evaluation, including:   
- Reactome pathway mapping files  
- CTD disease gene collection  
- modeling set and independent external validation set:  
  - chemical Saagar descriptors  
  - MIE assay profiles  
  - gene expression profiles  


All original source data were obtained from publicly available resources.  

## src
The src/ directory contains scripts for:  
- model construction  
- model training  
- external validation  
- layer-wise relevance propagation (LRP) analysis

## Environment
Python package requirements are provided in:  
- environment.yml
