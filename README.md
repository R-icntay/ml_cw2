# ML in Medical Imaging. Multi-Task Learning 

## 1. Downloading the data
1. Clone/download the code file
2. Download the image dataset from https://zenodo.org/record/7013610#.ZEgRROzP23I


## 2. Set up the environment
Run the following script to set up a new Conda environment and install the necessary packages:
```
conda create --name mphy0041-cw2 -c conda-forge numpy nibabel matplotlib tensorflow=2.10 pytorch=1.12 monai
conda install -c anaconda pathlib
```

## 3. Train & test the models 
Modify the first few lines of code in the file `main.py`:
```
# Choose whether to train and/or test model(s)
TRAIN           = 1
TEST            = 1

# Choose which models to test
BASE_CASE       = 1
AUX_SEGMENT_3   = 1
AUX_SEGMENT_6   = 1
AUX_RECONSTRUCT = 1
```

Run the following script on the command line:
```
python main.py
```

