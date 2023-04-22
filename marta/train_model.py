import torch
import pickle
from monai.data             import DataLoader, Dataset, decollate_batch
from monai.losses           import DiceLoss
from monai.metrics          import DiceMetric
from pathlib                import Path
from labels                 import modify_labels

BATCH_SIZE      = 2
max_epochs      = 60
val_interval    = 10

def set_data(train_files, train_transforms, val_files, val_transforms):
    torch.cuda.empty_cache()
    train_ds = Dataset(data = train_files, transform = train_transforms)
    train_dl = DataLoader(dataset = train_ds, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

    val_ds = Dataset(data = val_files, transform = val_transforms)
    val_dl = DataLoader(dataset = val_ds, batch_size = BATCH_SIZE, num_workers = 4, shuffle = False)
    
    return train_dl, val_dl


def set_model_params(model):
    # Input image has eight anatomical structures of planning interest
    loss_function       = DiceLoss(to_onehot_y = True, softmax = True, include_background=False) # For segmentation Expects BNHW[D] input i.e. batch, channel, height, width, depth, performs softmax on the channel dimension to get a probability distribution
    optimizer           = torch.optim.Adam(model.parameters(), (1e-3)/4) # Decreased the loss after getting a somewhat good model
    dice_metric_main    = DiceMetric(include_background=False, reduction="mean")# Collect the loss and metric values for every iteration
    dice_metric_aux     = DiceMetric(include_background=False, reduction="mean")
    scheduler           = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 60, eta_min = 1e-6) #** Adopt a cosine annealing learning rate schedule which reduces the learning rate as the training progresses
    
    return loss_function, optimizer, dice_metric_main, dice_metric_aux, scheduler


def save_results(MODEL_NAME, MODEL_PATH, epoch_loss_values, epoch_aux_loss_values, epoch_total_loss_values, main_metric_values, aux_metric_values):
    # Save epoch loss and metric values based on the model name
    pref = f"{MODEL_NAME.split('.')[0]}"
    with open(MODEL_PATH/f"{pref}_epoch_loss_values.pkl", "wb") as f:
        pickle.dump(epoch_loss_values, f)
    with open(MODEL_PATH/f"{pref}_epoch_aux_loss_values.pkl", "wb") as f:
        pickle.dump(epoch_aux_loss_values, f)
    with open(MODEL_PATH/f"{pref}_epoch_total_loss_values.pkl", "wb") as f:
        pickle.dump(epoch_total_loss_values, f)
    with open(MODEL_PATH/f"{pref}_main_metric_values.pkl", "wb") as f:
        pickle.dump(main_metric_values, f)
    with open(MODEL_PATH/f"{pref}_aux_metric_values.pkl", "wb") as f:
        pickle.dump(aux_metric_values, f)


def train_model(model, device, train_files, train_transforms, val_files, val_transforms, organs_dict, pred_main, label_main, pred_aux, label_aux):
    train_dl, val_dl = set_data(train_files, train_transforms, val_files, val_transforms)
    loss_function, optimizer, dice_metric_main, dice_metric_aux, scheduler = set_model_params(model)
    
    # Create model directory
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Create model save path
    MODEL_NAME = "nn_MTL_pytorch_male_pelvic_segmentation_model_3.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    best_metric             = -1
    best_metric_epoch       = -1
    epoch_loss_values       = []
    epoch_aux_loss_values   = []
    epoch_total_loss_values = []
    main_metric_values      = []
    aux_metric_values       = []

    # Loss weights
    main_weight = 1.1
    aux_weight  = 1.5
    
    for epoch in range(max_epochs):
        print("-" * 20)
        print(f"Epoch {epoch + 1} / {max_epochs}")
        
        # Put the model into training mode
        model.train()
        epoch_loss = 0
        epoch_aux_loss = 0
        epoch_total_loss = 0
        step = 0
        
        for batch in train_dl:
            step = step + 1
            inputs = batch["image"].permute(0, 1, 4, 2, 3).to(device)
            labels = batch["mask"].to(device) # Permute beccause of torch upsample
            
            main_labels, aux_labels = modify_labels(labels, organs_dict)

            # Forward pass
            main_seg, aux_seg = model(inputs) 
            main_seg, aux_seg = main_seg.permute(0, 1, 3, 4, 2), aux_seg.permute(0, 1, 3, 4, 2) # Permute back to BNHWD

            # Compute the loss functions
            main_seg_loss = loss_function(main_seg, main_labels)
            aux_seg_loss = loss_function(aux_seg, aux_labels)

            # Compute the total loss
            loss = main_weight * main_seg_loss + aux_weight * aux_seg_loss

            # Zero the gradients
            optimizer.zero_grad()

            # Find the gradients of the loss w.r.t the model parameters
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Add the loss to the epoch loss
            epoch_loss = epoch_loss + main_seg_loss.item()
            epoch_aux_loss = epoch_aux_loss + aux_seg_loss.item()
            epoch_total_loss = epoch_total_loss + loss.item()
        
        # Compute the average loss of the epoch
        epoch_loss          = epoch_loss        / step
        epoch_aux_loss      = epoch_aux_loss    / step
        epoch_total_loss    = epoch_total_loss  / step
        epoch_loss_values.append(epoch_loss)
        epoch_total_loss_values.append(epoch_total_loss)
        epoch_aux_loss_values.append(epoch_aux_loss)

        # Print the average loss of the epoch
        print(f"\nEpoch {epoch + 1} average dice loss for main task: {epoch_loss:.4f}")
        print(f"\nEpoch {epoch + 1} average dice loss for aux task: {epoch_aux_loss:.4f}")
        print(f"\nEpoch {epoch + 1} average total loss for both tasks: {epoch_total_loss:.4f}")

        # Step the scheduler after every epoch
        scheduler.step()

        # Print loss and evaluate model when epoch is divisible by val_interval
        if (epoch + 1) % val_interval == 0:
            # Put the model into evaluation mode
            model.eval()
            # Disable gradient calculation
            with torch.inference_mode():
                # Loop through the validation data
                for val_data in val_dl:
                    val_inputs, val_labels = val_data["image"].permute(0, 1, 4, 2, 3).to(device), val_data["mask"].to(device)
                    val_main_labels, val_aux_labels = modify_labels(val_labels, organs_dict)

                    # Forward pass
                    val_main_outputs, val_aux_outputs = model(val_inputs)
                    val_main_outputs, val_aux_outputs = val_main_outputs.permute(0, 1, 3, 4, 2), val_aux_outputs.permute(0, 1, 3, 4, 2)

                    # Transform main outputs and labels to calculate inference loss
                    val_main_outputs    = [pred_main(i) for i in decollate_batch(val_main_outputs)]
                    val_main_labels     = [label_main(i) for i in decollate_batch(val_main_labels)]

                    # Transform aux outputs and labels to calculate inference loss
                    val_aux_outputs     = [pred_aux(i) for i in decollate_batch(val_aux_outputs)]
                    val_aux_labels      = [label_aux(i) for i in decollate_batch(val_aux_labels)]

                    # Compute dice metric for current iteration
                    dice_metric_main(y_pred = val_main_outputs, y = val_main_labels)
                    dice_metric_aux(y_pred = val_aux_outputs, y = val_aux_labels)

                # Compute the average metric value across all iterations
                main_metric = dice_metric_main.aggregate().item()
                aux_metric = dice_metric_aux.aggregate().item()
                main_metric_values.append(main_metric)
                aux_metric_values.append(aux_metric)
                
                # Reset the metric for next validation run
                dice_metric_main.reset()
                dice_metric_aux.reset()

                # If the metric is better than the best seen so far, save the model
                if main_metric > best_metric:
                    best_metric = main_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print("saved new best metric model")
                
                print(
                    f"\nCurrent epoch: {epoch + 1} current mean dice for main task: {main_metric:.4f}"
                    f"\nBest mean dice for main task: {best_metric:.4f} at epoch: {best_metric_epoch}"
                    f"\nCurrent epoch: {epoch + 1} current mean dice for aux task: {aux_metric:.4f}"
                    )
                
    # When training is complete:
    print(f"Done training! Best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
    save_results(MODEL_NAME, MODEL_PATH, epoch_loss_values, epoch_aux_loss_values, epoch_total_loss_values, main_metric_values, aux_metric_values)
    

                    

        
        
        