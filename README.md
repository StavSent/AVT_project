# AVT_project
In this project we aim to distinguish spoofed from original audio clips.

WOW

## Dataset
The dataset used in this project is provided by [ASVspoof](https://datashare.ed.ac.uk/handle/10283/3336)

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required dependencies

```bash
pip install sklearn, numpy, pandas, scipy, librosa, seaborn
```

## How to run
First create GMM models

```bash
python GMM.py
```

Produce results of training and evaluation datasets separately

```bash
python results.py
```

Create SVM model

```bash
python SVM.py
```

Finally test the whole pipeline by running

```bash
python test.py
```
