
# Snow Depth Prediction using L-Band InSAR Products

## Overview

This work serves as a proof-of-concept that InSAR products can be used to estimate total snow depth through Machine Learning. This repository contains the code used to generate all results in the our paper (link to publication will follow shortly).

## Dataset

This work uses dataset from the 2017 NASA SnowEx Campaign. The InSAR data is from NASA JPL's UAVSAR sensor, and the Lidar Depth is from ASO. 

* Link to the remote sensing data we used: [link](https://drive.google.com/drive/folders/1Oalh_TooAk3PzQaBbyIwcWffLVaIEkuR?usp=sharing).
* Link to the ground truth data: [NSIDC](https://nsidc.org/data/snex17_sd/versions/1).


## Software and hardware list

| Software used | Link to the software  | Hardware specifications  | OS required |
|:---:  |:---:  |:---:  |:---:  |
| Python 3.11.5  | [https://github.com/pyenv/pyenv](https://github.com/pyenv/pyenv) | This code should work on any recent PC/Laptop | Linux (any), MacOS, Windows|

## Usage

To use the codes in this repository, you need to have Python installed. There are several ways to install Pyton. I will take you through how to install Python using the [pyenv](https://github.com/pyenv/pyenv) utility program. `pyenv` allows you to install different versions of Python and to seamlessly switch between them.


### 1. Install `pyenv``

Please follow the instructions [here](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation) to install `pyenv` for your operation system (OS).

### 2. Setup the environment

#### 2.1 Mac and Linux

Open the terminal and type the following to setup your virtual environment.

```{bash}
mkdir insar_project
cd insar_project
pyenv local 3.11.5
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## Methodology
- Explain the methodology and any models used.

## Results
- Discuss the results obtained (link to paper).

## Contributing
- Guidelines for contributing to the project.

## License
- License information.

## Acknowledgements
- Many thanks to SnowEX organization
