import torch
from pathlib import Path
from monai.utils    import set_determinism  
from split_data     import split_data
from transforms     import get_transforms
from model          import MTLResidualAttention3DUnet
from train_model    import train_model
from test_model     import test_model

# If TRUE, evaluate test data
TEST = 1

# Set deterministic training for reproducibility
set_determinism(seed = 2056)

img_path = Path("../data")
train_files, val_files, test_files = split_data(img_path, scale=28)

# Create transforms for training
train_transforms, val_transforms, pred_main, label_main, pred_aux, label_aux = get_transforms()

# Create an index dictionary for the organs
all_organs =  ["Background", "Bladder", "Bone", "Obturator internus", "Transition zone", "Central gland", "Rectum", "Seminal vesicle", "Neurovascular bundle"]
organs = {
    'all': all_organs,
    'main': ["Transition zone", "Central gland"],
    'aux':  ["Rectum", "Seminal vesicle", "Neurovascular bundle"],
    'dict': {organ: idx for idx, organ in enumerate(all_organs)}
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MTLResidualAttention3DUnet(in_channels = 1, main_out_channels = len(organs['main'])+1, aux_out_channels = len(organs['aux'])+1).to(device) #Main: 2 structures + background, Aux: 3 structures + background


train_model(model, device, train_files, train_transforms, val_files, val_transforms, organs, pred_main, label_main, pred_aux, label_aux)
if TEST:
    test_model(model, device, test_files, val_transforms, organs, pred_main, label_main, pred_aux, label_aux)
