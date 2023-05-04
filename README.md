# ESFPNet
Official Implementation of "ESFPNet: efficient deep learning architecture for real-time lesion segmentation in autofluorescence bronchoscopic video"

**:fire: NEWS :fire:**
**The full paper is available:** [The complete paper of ESFPNet](https://arxiv.org/pdf/2207.07759v3.pdf)

**The polyp datasets' results is available:** [Polyp Datasets' ESFPNet Models and Image Results](https://drive.google.com/drive/folders/1I4vsts-dfyUgrnbKi-Z8XQYVhVICYpOs?usp=share_link).

**:fire: CHEERS! :fire:** 
**This paper is selected as a [finalist of the Robert F. Wagner All-Conference Best Student Paper Award at SPIE Medical Imaging 2023](https://drive.google.com/file/d/1974ALKd6X0EUzm4nuR3kBWr2Gj0ctpdz/view?usp=share_link)**


## Global Rank

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/esfpnet-efficient-deep-learning-architecture/medical-image-segmentation-on-cvc-clinicdb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-clinicdb?p=esfpnet-efficient-deep-learning-architecture)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/esfpnet-efficient-deep-learning-architecture/medical-image-segmentation-on-etis)](https://paperswithcode.com/sota/medical-image-segmentation-on-etis?p=esfpnet-efficient-deep-learning-architecture)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/esfpnet-efficient-deep-learning-architecture/medical-image-segmentation-on-kvasir-seg)](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg?p=esfpnet-efficient-deep-learning-architecture)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/esfpnet-efficient-deep-learning-architecture/medical-image-segmentation-on-cvc-colondb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-colondb?p=esfpnet-efficient-deep-learning-architecture)

## Architecture of ESFPNet

<div align=center><img src="https://github.com/dumyCq/ESFPNet/blob/main/Figures/Network.jpg" width="1000" height="500" alt="Result"/></div>

## Installation & Usage
### Enviroment (Python 3.8)
- Install Pytorch (version 1.11.0, torchvision == 0.12.0):
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
- Install image reading and writting library (version 2.21.2):
```
conda install -c conda-forge imageio
```
- Install image processing library:
```
pip install scikit-image
```
- Install unofficial pytorch image model:
```
pip install timm
```
- Install OpenMMLab computer vision foundation:
```
pip install mmcv
```
- Install library for parsing and emitting YAML:
```
pip install pyyaml
```
- Install other packages:
```
conda install pillow numpy matplotlib
```
- Install Jupyter-Notebook to run .ipynb file
```
conda install -c anaconda jupyter
```
### Dataset
- Download the training and testing dataset from this link: [Experiment Dataset](https://drive.google.com/drive/folders/1FneOIY5OC0gaIHceBqYXqj5GCdutcLfv?usp=sharing)
- Extract the folders and copy them under "Endoscope-WL" folder
- The datasets are ordered as follows in "Endoscope-WL" folder:
- TrainDataset and TestDataset are used in Table 4 & 5.
- CVC-ClinicDB_Splited and Kvasir_Splited are used in Table 3.
```
|-- TrainDataset
|   |-- CVC-ClinicDB
|   |   |-- images
|   |   |-- masks
|   |-- Kvasir
|       |-- images
|       |-- masks

|-- TestDataset
|   |-- CVC-300
|   |   |-- images
|   |   |-- masks
|   |-- CVC-ClinicDB
|   |   |-- images
|   |   |-- masks
|   |-- CVC-ColonDB
|   |   |-- images
|   |   |-- masks
|   |-- ETIS-LaribPolypDB
|   |   |-- images
|   |   |-- masks
|   |-- Kvasir
|       |-- images
|       |-- masks

|-- CVC-ClinicDB_Splited
|   |-- testSplited
|   |   |-- images
|   |   |-- masks
|   |-- trainSplited
|   |   |-- images
|   |   |-- masks
|   |-- validationSplited
|   |   |-- images
|   |   |-- masks

|-- Kvasir_Splited
|   |-- testSplited
|   |   |-- images
|   |   |-- masks
|   |-- trainSplited
|   |   |-- images
|   |   |-- masks
|   |-- validationSplited
|   |   |-- images
|   |   |-- masks
```
- The default dataset paths can be changed in "Configure.yaml"
- To randomly split the CVC-ClincDB or Kvasir dataset, set "if_renew = True" in "ESFPNet_Endoscope_Learning_Ability.ipynb"
- To repeat generate the splitting dataset, previous generated folder shold be detelted first
- To reuse the splitting dataset without generating a new dataset, set "if_renew = False"
### Pretrained Model
- Download the pretrained Mixtransformer from this link: [Pretrained Model](https://drive.google.com/drive/folders/1FLtIfDHDaowqyF_HhmORFMlRzCpB94hV?usp=sharing)
- Put the pretrained models under "Pretrained" folder
## Evaluation
We computed all metrics using the freely available ParaNet Matlab tool with `./eval/main_GA.m` or `./eval/main_LA.m`.

One can download [saved ESFPNet model](https://drive.google.com/drive/folders/1I4vsts-dfyUgrnbKi-Z8XQYVhVICYpOs?usp=share_link) and then use `ESFPNet_Endoscope_ImageWrite.ipynb` to generate results for checking.


### Autofluorescence Bronchoscopic Segmentation Results
Quantitative Comparsion (Table 2):
<div align=left><img src="https://github.com/dumyCq/ESFPNet/blob/main/results/QuantitativeComparsion.JPG" width="600" alt="Result"/></div>

Sample of AFB Segmentation Results (Figure 3):
<div align=left><img src="https://github.com/dumyCq/ESFPNet/blob/main/results/ImageResult.jpg" width="600" alt="Result"/></div>

### Autofluorescence Bronchoscopic Video Clip Analysis
ESFPNet detection and segmentation test on a video clip (case 21405_198, Figure 4)
<div align=left><img src="https://github.com/dumyCq/ESFPNet/blob/main/results/VideoTest.jpg" width="600" alt="Result"/></div>


Animated GIF Result View 

| AFB video Clip | Ground Truth |Segmentation Result |
| :---: | :---: | :---: |
|<div align=center><img src="https://github.com/dumyCq/ESFPNet/blob/main/results/OriginalFrames_10FPS_360.gif" width="180" alt="Result"/></div>|<div align=center><img src="https://github.com/dumyCq/ESFPNet/blob/main/results/GroundTruth_10FPS_360.gif" width="180" alt="Result"/></div>| <div align=center><img src="https://github.com/dumyCq/ESFPNet/blob/main/results/Prediction_10FPS_360.gif" width="180" alt="Result"/></div>|

### Polyp Segmentation Figure Results
Quantitative Comparison in Learning Ability (Table 3)
<div align=left><img src="https://github.com/dumyCq/ESFPNet/blob/main/results/LABest.JPG" width="400" alt="Result"/></div>

Quantitative Comparison in Power balance between Learning Ability and Generalizability Capbility (Table 5)
<div align=left><img src="https://github.com/dumyCq/ESFPNet/blob/main/results/EITSBest.JPG" width="800" alt="Result"/></div>

Sample of Polyp Segmentation Results (Figure 5)
<div align=left><img src="https://github.com/dumyCq/ESFPNet/blob/main/results/PublicResult.jpg" width="600" alt="Result"/></div>

### Saved ESFPNet Model And Image Results For Polyp Dataset
The saved ESFPNet model and generated image results for Table 3, 4, and 5 are stored [here (Polyp Datasets' Models and Image Results)](https://drive.google.com/drive/folders/1I4vsts-dfyUgrnbKi-Z8XQYVhVICYpOs?usp=share_link).

In addtion, the evaluation results are stored in each folder result.txt

### Citation
If you think this paper helps, please cite:
```
@inproceedings{chang2023esfpnet,
  title={ESFPNet: efficient deep learning architecture for real-time lesion segmentation in autofluorescence bronchoscopic video},
  author={Chang, Qi and Ahmad, Danish and Toth, Jennifer and Bascom, Rebecca and Higgins, William E},
  booktitle={Medical Imaging 2023: Biomedical Applications in Molecular, Structural, and Functional Imaging},
  volume={12468},
  pages={1246803},
  year={2023},
  organization={SPIE}
}
```
Since the training of MixTransformer based network requires a good GPU.
One helpful state-of-the-art work compared in this paper without using MixTransformer backbone is [CARANet](https://github.com/AngeLouCN/CaraNet)
If you also think this work helps, please cite:
```
@inproceedings{lou2021caranet,
author = {Ange Lou and Shuyue Guan and Hanseok Ko and Murray H. Loew},
title = {{CaraNet: context axial reverse attention network for segmentation of small medical objects}},
volume = {12032},
booktitle = {Medical Imaging 2022: Image Processing},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {81 -- 92},
year = {2022},
doi = {10.1117/12.2611802}}
```
