# STASCAN
![](./STASCAN_overview.png)


## Overview
The development of spatial transcriptomics (ST) technologies has enabled the three-dimensional presentation of cellular distribution with differential gene expression profiles in tissue, but the limited spatial resolution still poses a challenge to acquire a fine-resolved cell map. Here we propose STASCAN, which both gene expression and the morphological information are simultaneously utilized to improve the cellular resolution of captured domains and even gap regions. Besides, STASCAN is further designed to enable cell-type predictions at subdivide-spot resolution and construction of the 3D spatial cell map from histology images alone. 


## Hardware dependencies
STASCAN can run on CPU-only hardware, but training new models will take exceedingly long.

We recommend running STASCAN on a GPU hardware.


## Software dependencies
tensorflow==2.9.0

numpy>=1.21.6

matplotlib>=3.5.3

opencv-python>=4.6.0

pillow>=6.2.0

scikit-learn>=1.0.2

scikit-image>=0.19.0


## Installation
pip install STASCAN==1.1.0


## Getting started

See [Documentation and Tutorials](https://stascan-tutorial.readthedocs.io/en/latest/).


## License
[MIT © Ying Wu.](./LICENSE.txt)


## Citation