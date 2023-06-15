# CoevolveML
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8035449.svg)](https://doi.org/10.5281/zenodo.8035449)  
**Data and Code for: Deploying synthetic coevolution and machine learning 5 to engineer protein-protein interactions**  

*Fine-tuning of protein-protein interactions occurs naturally through coevolution, but this process is difficult to recapitulate in the laboratory. We describe a synthetic platform for protein-protein coevolution that can isolate matched pairs of interacting muteins from complex libraries. This large dataset of coevolved complexes drove a systems-level analysis of molecular recognition between Z domain-affibody pairs spanning a wide range of structures, affinities, cross-reactivities, and orthogonalities, and captured a broad spectrum of coevolutionary networks. Furthermore, we harnessed pre-trained protein language models to expand, in silico, the amino acid diversity of our coevolution screen, predicting remodeled interfaces beyond the reach of the experimental library. The integration of these approaches provides a means of generating protein complexes with diverse molecular recognition properties as tools for biotechnology and synthetic biology.*

Paper Link [TBD]



<p align='center'>
<img src="https://github.com/akds/CoevolveML/blob/main/img/Fig.png" width="75%" >
 </p> 


## Dependencies
python  >= 3.8  
pytorch >= 1.11.0  
CUDA >= 11.6  


## Inference
1. A [notebook](https://github.com/akds/CoevolveML/blob/main/examples/Model_Inference.ipynb) is provided for inference using our pre-trained model and pre-processed data for results shown in the manuscript.  

2. For inference from sequence pairs, you can follow [this notebook](https://github.com/akds/CoevolveML/blob/main/examples/Sequence_Inference.ipynb). please see [ESM](https://github.com/facebookresearch/esm) for detailed installation instruction of the ESM-1b model. 

## pre-trained model & processed data
You can also find them [here](https://drive.google.com/drive/folders/1Jgi4gWmv3jszj244YSmhLOv05PZwXXXg?usp=sharing)
