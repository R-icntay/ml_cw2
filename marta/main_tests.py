import torch
from pathlib import Path
from monai.utils        import set_determinism  
from split_data         import split_data
from transforms         import get_transforms
from model              import ResidualAttention3DUnet, MTLResidualAttention3DUnet
from train_model        import train_model
from test_model         import test_model
from train_model_base   import train_model_base
from test_model_base    import test_model_base

# Choose whether to train and/or test model(s)
TRAIN           = 1
TEST            = 1

# Choose which models to test
BASE_CASE       = 1
AUX_SEGMENT     = 1
AUX_RECONSTRUCT = 0

# Set deterministic training for reproducibility
set_determinism(seed = 2056)

# Path to data
img_path = Path("../data")
train_files, val_files, test_files = split_data(img_path)

# Create transforms for training
train_transforms, val_transforms, pred_main, label_main, pred_aux, label_aux = get_transforms()

# Use CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define organ names in the segmentation task
all_organs =  ["Background", "Bladder", "Bone", "Obturator internus", "Transition zone", "Central gland", "Rectum", "Seminal vesicle", "Neurovascular bundle"]
organs = {
    'all': all_organs,
    'main': ["Transition zone", "Central gland"],
    'aux': [],
    'dict': {organ: idx for idx, organ in enumerate(all_organs)}
    }

############# BASE CASE #############
if BASE_CASE:
    torch.cuda.empty_cache()
    model  = ResidualAttention3DUnet(in_channels = 1, out_channels = len(organs['main'])+1).to(device) 
    
    if TRAIN:
        train_model_base(model, device, train_files, train_transforms, val_files, val_transforms, organs, pred_main, label_main)
    if TEST:
        test_model_base(model, device, test_files, val_transforms, organs, pred_main, label_main)


############# AUXILIARY TASK - SEGMENT 3 EXTRA STRUCTURES #############
if AUX_SEGMENT:
    torch.cuda.empty_cache()
    organs['aux'] = ["Rectum", "Seminal vesicle", "Neurovascular bundle"]
    model = MTLResidualAttention3DUnet(in_channels = 1, main_out_channels = len(organs['main'])+1, aux_out_channels = len(organs['aux'])+1).to(device) 
    
    if TRAIN:
        train_model(model, device, train_files, train_transforms, val_files, val_transforms, organs, pred_main, label_main, pred_aux, label_aux)
    if TEST:
        test_model(model, device, test_files, val_transforms, organs, pred_main, label_main, pred_aux, label_aux)
    
    
############# AUXILIARY TASK - RECONSTRUCTION #############
if AUX_RECONSTRUCT:
    torch.cuda.empty_cache()
    organs['aux'] = []
