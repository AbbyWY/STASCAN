# STASCAN
![](./STASCAN_overview.png)


## Overview
STASCAN is an AI-driven method that simultaneously utilizes the gene expression and the morphological information to depict fine-resolved cell distribution of the captured domains and even gap regions. 

STASCAN can also enable cell-type predictions at subdivide-spot resolution and predict cell distribution in 3D space from histology images alone. 


## Hardware dependencies
STASCAN can run on CPU-only hardware, but training new models will take exceedingly long.

We recommend running STASCAN on a GPU hardware.


## Software dependencies
tensorflow==2.9.0

numpy>=1.21.6

matplotlib>=3.5.3

cv2>=4.6.0

PIL>=6.2.0

tensorflow==2.9.0

sklearn>=1.0.2

skimage>=0.19.0


## Installation
python setup.py build


## Getting started

A guide to getting quickly started with STASCAN can be found [here.](./Demo/Demo1/demo1.ipynb)

A complete guide to run a multi-sections model with optional processes can be found [here.](./Demo/Demo2/demo2.ipynb)

## License
[MIT © Ying Wu.](./LICENSE.txt)

## Citation
