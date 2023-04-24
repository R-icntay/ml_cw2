import torch
import argparse
from pathlib import Path
from monai.utils    import set_determinism  #!pip install monai
from split_data     import split_data
from transforms     import get_transforms
from model          import MTLResidualAttention3DUnet
from train_model    import train_model
from test_model     import test_model

###### INPUTS
parser = argparse.ArgumentParser(description='Parse data')
parser.add_argument('-t','--test',  action='store_true', help='Evaluate test data')

opt     = parser.parse_args()
TEST    = opt.test
print('Evaluate test data: ', TEST)

# Set deterministic training for reproducibility
set_determinism(seed = 2056)

img_path = Path("../data")
train_files, val_files, test_files = split_data(img_path)

# Create transforms for training
train_transforms, val_transforms, pred_main, label_main, pred_aux, label_aux = get_transforms()

# Create an index dictionary for the organs
organs      = ["Background", "Bladder", "Bone", "Obturator internus", "Transition zone", "Central gland", "Rectum", "Seminal vesicle", "Neurovascular bundle"]
organs_dict = {organ: idx for idx, organ in enumerate(organs)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MTLResidualAttention3DUnet(in_channels = 1, main_out_channels = 3, aux_out_channels = 4).to(device) #Main: 2 structures + background, Aux: 3 structures + background

print(model)

#train_model(model, device, train_files, train_transforms, val_files, val_transforms, organs_dict, pred_main, label_main, pred_aux, label_aux)
#if TEST:
#    test_model(model, device, test_files, val_transforms, organs_dict, pred_main, label_main, pred_aux, label_aux)
