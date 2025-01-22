
# Snow Depth Prediction using L-Band InSAR Products

[![DOI](https://zenodo.org/badge/DOI/10.3389/frsen.2024.1481848.svg)](https://doi.org/10.3389/frsen.2024.1481848)


## Overview

This work serves as a proof-of-concept that InSAR products can be used to estimate total snow depth through Machine Learning. This repository contains the code used to generate all results in our paper "*Advancing Terrestrial Snow Depth Monitoring with Machine Learning and L-band InSAR data: A Case Study Using SnowEx 2017 Data*" (doi: https://doi.org/10.3389/frsen.2024.1481848).

## Dataset

This work uses dataset from the 2017 NASA SnowEx Campaign. The InSAR data is from NASA JPL's UAVSAR sensor, and the Lidar Depth is from ASO. 

* Link to the remote sensing data we used: [link](https://drive.google.com/drive/folders/1Oalh_TooAk3PzQaBbyIwcWffLVaIEkuR?usp=sharing).
* Link to the *in situ* depth measurements: [NSIDC](https://nsidc.org/data/snex17_sd/versions/1).


## Software and hardware list

| Software used | Link to the software  | Hardware specifications  | OS required |
|:---:  |:---:  |:---:  |:---:  |
| Python 3.11.5  | [https://github.com/pyenv/pyenv](https://github.com/pyenv/pyenv) | This code should work on any recent PC/Laptop | Linux (any), MacOS, Windows|

## Usage

To use the codes in this repository, you need to have Python installed. There are several ways to install Pyton. I will take you through how to install Python using the [pyenv](https://github.com/pyenv/pyenv) utility program. `pyenv` allows you to install different versions of Python and to seamlessly switch between them.


### 1. Install `pyenv`

Please follow the instructions [here](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation) to install `pyenv` for your operation system (OS).
### 2. Setup the environment

#### 2.1 Mac and Linux

Open the terminal and type the following commands to setup your virtual environment.

```bash
pyenv install 3.11.5
mkdir insar_project
cd insar_project
pyenv local 3.11.5
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

#### 2.2 Windows

`pyenv` does not officially support Windows OS. Hence, you should install Python directly for your own version of Windows either by using the Microsoft Store or by navigating to [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/) in your browser and following the download link. During installation, be sure to check the box that adds Python to your `PATH` environment variables. To run Python in a virtual environment, open the PowerShell program and type the follow commands.

```bash
mkdir insar_project
cd insar_project
python -m venv .venv
.venv/Scripts/activate
pip install --upgrade pip
```

If these instructions doesn't work for you (Windows users), you may need change the execution policy of your computer. To do this, run the PowerShell program as an administrator and run `Set ExecutionPolicy Unrestricted`. Afterwards, exit and reopen the PowerShell program (not as an administrator) then type commands in 2.2. 

**Note: the Windows installation guide was not tested and the entire codebase was not tested on a Windows machine!** 

### 3. Clone the Repository

```bash
git clone git@github.com:cryogars/uavsar-lidar-ml-project.git
cd uavsar-lidar-ml-project
```

### 4. Install Source Code

```bash
pip install .
```

### 5. Directory Setup

Create the data folder using the command below.

```bash
mkdir data
```

### 6. Download Data

1. Download the remote sensing dataset into the `data` folder using this [link](https://drive.google.com/drive/folders/1Oalh_TooAk3PzQaBbyIwcWffLVaIEkuR?usp=sharing).

2. Download the *in situ* data following these steps:

    * Create the folder to put the *in situ* measurement and change directory:
    ```bash
    mkdir data/SNOWEX
    cd data/SNOWEX
    ```

    * Dowlond the *in situ* measurement from [NSIDC](https://nsidc.org/data/snex17_sd/versions/1). When you land at the page navigate to `Data Access Tool` -> `Get Data` and click `Download Script`. At the time of writing,  `Download Script` is on the botton right of the page and the downloaded script is named `nsidc-download_SNEX17_SD.001_2025-01-07.py`. Note that you will need  Earthdata username and password to download the *in situ* measurements. Click [here](https://urs.earthdata.nasa.gov/users/new) to register.

    * Run the Python script to download the *in situ* data using:
    ```bash
    python nsidc-download_SNEX17_SD.001_2025-01-07.py
    ```

    * Move back to the parent directory and run the codes. Note that you will need `Jupyter Notebook` installed.
    ```bash
    cd ../..
    ```


### 7. Deactivate Environment

After running the experiments, you can deactivate the virtual environment by running the command below.

```bash
deactivate
```

## Acknowledgements

- We would like to express our gratitude to the NASA Terrestrial Hydrology Program and all participants of the SnowEx campaign for providing the Lidar and *in situ* data. 

- We also thank Yunling Lou - the UAVSAR Project Manager at NASA JPL - for the UAVSAR data. 

- This work was partially funded by the U.S. Army Cold Regions Research and Engineering Laboratory (CRREL) under contract W913E523C0002.


## Citation

We would appreciate citations if you use our code in your work.

```bash
@ARTICLE{alabi5advancing,
    AUTHOR={Alabi, Ibrahim Olalekan  and Marshall, Hans-Peter  and Mead, Jodi  and Trujillo, Ernesto },
    TITLE={Advancing terrestrial snow depth monitoring with machine learning and L-band InSAR data: a case study using NASA’s SnowEx 2017 data},
    JOURNAL={Frontiers in Remote Sensing},
    VOLUME={5},
    YEAR={2025},
    URL={https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2024.1481848},
    DOI={10.3389/frsen.2024.1481848},
    ISSN={2673-6187}
}
```

## Questions?

Ibrahim Alabi (Email: ibrahimolalekana@u.boisestate.edu)