import torch
import pickle
from monai.data             import DataLoader, Dataset, decollate_batch
from monai.metrics          import DiceMetric
from pathlib                import Path
from labels                 import modify_labels

def set_data(val_files, val_transforms, BATCH_SIZE):
    """
    Create dataloader for test set.
    """
    
    torch.cuda.empty_cache()
    val_ds = Dataset(data = val_files, transform = val_transforms)
    val_dl = DataLoader(dataset = val_ds, batch_size = BATCH_SIZE, num_workers = 4, shuffle = False)
    
    return val_dl


def set_model_params():
    """
    Set metrics for evaluation.
    """
    
    # Input image has eight anatomical structures of planning interest
    dice_metric_main    = DiceMetric(include_background=False, reduction="mean") # Collect the loss and metric values for every iteration
    
    return dice_metric_main


def save_results(MODEL_NAME, MODEL_PATH, main_metric_values):
    """
    Save performance metrics.
    """
    
    # Save metric values
    pref = f"{MODEL_NAME.split('.')[0]}"
    with open(MODEL_PATH/f"{pref}_test.pkl", "wb") as f:
        pickle.dump(main_metric_values, f)


def test_model_base(model, device, params, val_files, val_transforms, organs_dict, pred_main, label_main, model_name):
    """
    Evaluate the test dataset
    """
    BATCH_SIZE = params['BATCH_SIZE']
    
    val_dl              = set_data(val_files, val_transforms, BATCH_SIZE)
    dice_metric_main    = set_model_params()
    
    # Model save path
    MODEL_PATH = Path("models")
    MODEL_NAME = model_name + ".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    print("-" * 40)
    print("Starting model testing...")
    
    # Disable gradient calculation
    with torch.inference_mode():
        # Loop through the validation data
        for val_data in val_dl:
            val_inputs, val_labels = val_data["image"].permute(0, 1, 4, 2, 3).to(device), val_data["mask"].to(device)
            val_main_labels, _     = modify_labels(val_labels, organs_dict)

            # Forward pass
            val_main_outputs = model(val_inputs)
            val_main_outputs = val_main_outputs.permute(0, 1, 3, 4, 2)
            
            # Transform main outputs and labels to calculate inference loss
            val_main_outputs    = [pred_main(i) for i in decollate_batch(val_main_outputs)]
            val_main_labels     = [label_main(i) for i in decollate_batch(val_main_labels)]

            # Compute dice metric for current iteration
            dice_metric_main(y_pred = val_main_outputs, y = val_main_labels)
            
        # Compute the average metric value across all iterations
        main_metric = dice_metric_main.aggregate().item()
        
    print(
        f"\nMean dice for main task: {main_metric:.4f}"
        )
    
    save_results(MODEL_NAME, MODEL_PATH, main_metric)
    

                    

        
        
        