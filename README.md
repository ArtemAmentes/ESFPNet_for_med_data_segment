# ESFPNet
Official Implementation of "ESFPNet: efficient deep learning architecture for real-time lesion segmentation in autofluorescence bronchoscopic video"

## Global Rank

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/esfpnet-efficient-deep-learning-architecture/medical-image-segmentation-on-cvc-clinicdb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-clinicdb?p=esfpnet-efficient-deep-learning-architecture)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/esfpnet-efficient-deep-learning-architecture/medical-image-segmentation-on-etis)](https://paperswithcode.com/sota/medical-image-segmentation-on-etis?p=esfpnet-efficient-deep-learning-architecture)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/esfpnet-efficient-deep-learning-architecture/medical-image-segmentation-on-kvasir-seg)](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg?p=esfpnet-efficient-deep-learning-architecture)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/esfpnet-efficient-deep-learning-architecture/medical-image-segmentation-on-cvc-colondb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-colondb?p=esfpnet-efficient-deep-learning-architecture)

## Architecture of ESFPNet

<div align=center><img src="https://github.com/dumyCq/ESFPNet/blob/main/Figures/Network.jpg" width="1000" height="550" alt="Result"/></div>

## Installation & Usage
### Enviroment (Python 3.8)
- Install Pytorch:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
- Install image reading and writting library:
```
conda install -c conda-forge imageio
```
- Install image processing library:
```
pip install scikit-image
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
