from pathlib import Path
#!pip install monai
from monai.utils import set_determinism 
from split_data import split_data
from transforms import get_transforms
import torch
from model import MTLResidualAttention3DUnet
from train_model import train_model


# Set deterministic training for reproducibility
set_determinism(seed = 2056)

# Read all files ending with _img.nii
img_path = Path("data/data")
train_files, val_files, test_files = split_data(img_path)

# Create transforms for training
train_transforms, val_transforms, pred_main, label_main, pred_aux, label_aux = get_transforms()

# Create an index dictionary for the organs
organs      = ["Background", "Bladder", "Bone", "Obturator internus", "Transition zone", "Central gland", "Rectum", "Seminal vesicle", "Neurovascular bundle"]
organs_dict = {organ: idx for idx, organ in enumerate(organs)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MTLResidualAttention3DUnet(in_channels = 1, main_out_channels = 3, aux_out_channels = 4).to(device)#Main: 2 structures + background, Aux: 3 structures + background

train_model(model, device, train_files, train_transforms, val_files, val_transforms, organs_dict, pred_main, label_main, pred_aux, label_aux)