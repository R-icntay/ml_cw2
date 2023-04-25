import torch
from pathlib import Path
from monai.utils        import set_determinism  
from split_data         import split_data
from transforms         import get_transforms
from model              import ResidualAttention3DUnet, MTLResidualAttention3DUnet, MTLResidualAttentionRecon3DUnet
from train_model        import train_model
from test_model         import test_model
from train_model_base   import train_model_base
from test_model_base    import test_model_base

# Choose whether to train and/or test model(s)
TRAIN           = 1
TEST            = 1

# Choose which models to test
BASE_CASE       = 1
AUX_SEGMENT_3   = 1
AUX_SEGMENT_6   = 1
AUX_RECONSTRUCT = 1

# Parameters
params = {
    'BATCH_SIZE':       2,
    'MAX_EPOCHS':       100,
    'VAL_INT':          10,
    'PRINT_INT':        10
}

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
    'dict': {organ: idx for idx, organ in enumerate(all_organs)}
    }

############# BASE CASE #############
if BASE_CASE:
    organs['aux']  = []
    params['TASK'] = 'BASE_CASE'
    model_name     = 'base_case'
    model  = ResidualAttention3DUnet(in_channels = 1, out_channels = len(organs['main'])+1, device=device).to(device) 
    
    if TRAIN:
        torch.cuda.empty_cache()
        train_model_base(model, device, params, train_files, train_transforms, val_files, val_transforms, organs, pred_main, label_main, model_name)
    if TEST:
        torch.cuda.empty_cache()
        test_model_base(model, device, params, test_files, val_transforms, organs, pred_main, label_main, model_name)


############# AUXILIARY TASK - SEGMENT 3 EXTRA STRUCTURES #############
if AUX_SEGMENT_3:
    organs['aux']  = ["Rectum", "Seminal vesicle", "Neurovascular bundle"]
    params['TASK'] = 'SEGMENT'
    model_name     = 'auxiliary_segment_3'
    model = MTLResidualAttention3DUnet(in_channels = 1, main_out_channels = len(organs['main'])+1, aux_out_channels = len(organs['aux'])+1, device=device).to(device) 
    
    if TRAIN:
        torch.cuda.empty_cache()
        train_model(model, device, params, train_files, train_transforms, val_files, val_transforms, organs, pred_main, label_main, pred_aux, label_aux, model_name)
    if TEST:
        torch.cuda.empty_cache()
        test_model(model, device, params, test_files, val_transforms, organs, pred_main, label_main, pred_aux, label_aux, model_name)
        
        
############# AUXILIARY TASK - SEGMENT 6 EXTRA STRUCTURES #############
if AUX_SEGMENT_6:
    organs['aux']  = ["Rectum", "Seminal vesicle", "Neurovascular bundle", "Bladder", "Bone", "Obturator internus"]
    params['TASK'] = 'SEGMENT'
    model_name     = 'auxiliary_segment_6'
    model = MTLResidualAttention3DUnet(in_channels = 1, main_out_channels = len(organs['main'])+1, aux_out_channels = len(organs['aux'])+1, device=device).to(device) 
    
    if TRAIN:
        torch.cuda.empty_cache()
        train_model(model, device, params, train_files, train_transforms, val_files, val_transforms, organs, pred_main, label_main, pred_aux, label_aux, model_name)
    if TEST:
        torch.cuda.empty_cache()
        test_model(model, device, params, test_files, val_transforms, organs, pred_main, label_main, pred_aux, label_aux, model_name)
    
    
############# AUXILIARY TASK - RECONSTRUCTION #############
if AUX_RECONSTRUCT:
    organs['aux']   = []
    params['TASK'] = 'RECONSTRUCT'
    model_name     = 'auxiliary_reconstruct'
    model = MTLResidualAttentionRecon3DUnet(in_channels = 1, out_channels = len(organs['main'])+1, device=device).to(device) 
    
    if TRAIN:
        torch.cuda.empty_cache()
        train_model(model, device, params, train_files, train_transforms, val_files, val_transforms, organs, pred_main, label_main, pred_aux, label_aux, model_name)
    if TEST:
        torch.cuda.empty_cache()
        test_model(model, device, params, test_files, val_transforms, organs, pred_main, label_main, pred_aux, label_aux, model_name)
    
