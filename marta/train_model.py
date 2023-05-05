import torch
import torch.nn as nn
import pickle
from monai.data                 import DataLoader, Dataset, decollate_batch
from monai.losses               import DiceLoss
from monai.losses.ssim_loss     import SSIMLoss
from monai.metrics              import DiceMetric, MSEMetric
from monai.metrics.regression   import SSIMMetric
from pathlib                    import Path
from labels                     import modify_labels


def set_data(train_files, train_transforms, val_files, val_transforms, BATCH_SIZE):
    """
    Create dataloader for test set.
    """
    
    torch.cuda.empty_cache()
    train_ds = Dataset(data = train_files, transform = train_transforms)
    train_dl = DataLoader(dataset = train_ds, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

    val_ds = Dataset(data = val_files, transform = val_transforms)
    val_dl = DataLoader(dataset = val_ds, batch_size = BATCH_SIZE, num_workers = 4, shuffle = False)
    
    return train_dl, val_dl


def set_model_params(model, TASK):
    """
    Set model parameters and metrics for evaluation.
    """
    
    # Input image has eight anatomical structures of planning interest
    loss_main       = DiceLoss(to_onehot_y = True, softmax = True, include_background=False) 
    metric_main     = DiceMetric(include_background=False, reduction="mean")
    
    if TASK == 'SEGMENT':
        loss_aux    = DiceLoss(to_onehot_y = True, softmax = True, include_background=False) 
        metric_aux  = DiceMetric(include_background=False, reduction="mean")
    else:
        #loss_aux    = SSIMLoss()
        loss_aux    = torch.nn.L1Loss()
        metric_aux  = MSEMetric() # Do we use MSE or SSIM?
        
    optimizer       = torch.optim.Adam(model.parameters(), (1e-3)/1) # Decreased the loss after getting a somewhat good model
    scheduler       = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min = 1e-6) #** Adopt a cosine annealing learning rate schedule which reduces the learning rate as the training progresses
    
    return loss_main, loss_aux, metric_main, metric_aux, optimizer, scheduler


def save_results(MODEL_NAME, MODEL_PATH, epoch_loss_values, epoch_aux_loss_values, epoch_total_loss_values, main_metric_values, aux_metric_values):
    """
    Save performance metrics.
    """
    
    # Save epoch loss and metric values
    pref = f"{MODEL_NAME.split('.')[0]}"
    with open(MODEL_PATH/f"{pref}_epoch_loss.pkl", "wb") as f:
        pickle.dump(epoch_loss_values, f)
    with open(MODEL_PATH/f"{pref}_epoch_aux_loss.pkl", "wb") as f:
        pickle.dump(epoch_aux_loss_values, f)
    with open(MODEL_PATH/f"{pref}_epoch_total_loss.pkl", "wb") as f:
        pickle.dump(epoch_total_loss_values, f)
    with open(MODEL_PATH/f"{pref}_main_validation.pkl", "wb") as f:
        pickle.dump(main_metric_values, f)
    with open(MODEL_PATH/f"{pref}_aux_validation.pkl", "wb") as f:
        pickle.dump(aux_metric_values, f)


def train_model(model, device, params, train_files, train_transforms, val_files, val_transforms, organs, pred_main, label_main, pred_aux, label_aux, model_name):
    """
    Train the model on the training dataset and evaluate the validation dataset.
    """
    BATCH_SIZE      = params['BATCH_SIZE']
    MAX_EPOCHS      = params['MAX_EPOCHS']
    VAL_INTERVAL    = params['VAL_INTERVAL']
    PRINT_INTERVAL  = params['PRINT_INTERVAL']
    TASK            = params['TASK']
    
    train_dl, val_dl = set_data(train_files, train_transforms, val_files, val_transforms, BATCH_SIZE)
    loss_main, loss_aux, metric_main, metric_aux, optimizer, scheduler = set_model_params(model, TASK)
    
    # Create model directory
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Create model save path
    MODEL_NAME = model_name + ".pth"
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
    
    print("-" * 20)
    print("Starting model training...")
    
    for epoch in range(1,MAX_EPOCHS):
        if epoch % PRINT_INTERVAL == 0:
            print("-" * 20)
            print(f"Epoch {epoch} / {MAX_EPOCHS}")
        
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
            
            # Modify the main labels to match the output of the main decoder
            main_labels, aux_labels = modify_labels(labels, organs)

            # Forward pass
            main_seg, aux_seg = model(inputs) 
            main_seg, aux_seg = main_seg.permute(0, 1, 3, 4, 2), aux_seg.permute(0, 1, 3, 4, 2) # Permute back to BNHWD

            # Compute the loss functions
            main_seg_loss = loss_main(main_seg, main_labels)
            if TASK == 'SEGMENT':
                aux_seg_loss = loss_aux(aux_seg, aux_labels)
            else:
                aux_seg_loss = loss_aux(aux_seg, inputs.permute(0, 1, 3, 4, 2))
                
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

        if epoch % PRINT_INTERVAL == 0:
            # Print the average loss of the epoch
            print(f"\nEpoch {epoch} average loss for main task: {epoch_loss:.4f}")
            print(f"\nEpoch {epoch} average loss for aux task: {epoch_aux_loss:.4f}")
            print(f"\nEpoch {epoch} average total loss for both tasks: {epoch_total_loss:.4f}")

        # Step the scheduler after every epoch
        scheduler.step()

        # Print loss and evaluate model when epoch is divisible by val_interval
        if epoch % VAL_INTERVAL == 0:
            print("-" * 40)
            print("Testing on validation data...")
            
            # Put the model into evaluation mode
            model.eval()
            # Disable gradient calculation
            with torch.inference_mode():
                # Loop through the validation data
                for val_data in val_dl:
                    val_inputs = val_data["image"].permute(0, 1, 4, 2, 3).to(device)
                    val_labels = val_data["mask"].to(device)
                    
                    val_main_labels, val_aux_labels = modify_labels(val_labels, organs)

                    # Forward pass
                    val_main_outputs, val_aux_outputs = model(val_inputs)
                    val_main_outputs, val_aux_outputs = val_main_outputs.permute(0, 1, 3, 4, 2), val_aux_outputs.permute(0, 1, 3, 4, 2)

                    # Transform main outputs and labels to calculate inference loss
                    val_main_outputs    = [pred_main(i) for i in decollate_batch(val_main_outputs)]
                    val_main_labels     = [label_main(i) for i in decollate_batch(val_main_labels)]

                    # Compute metric for current iteration
                    metric_main(y_pred = val_main_outputs, y = val_main_labels)
                    if TASK == 'SEGMENT':
                        # Transform aux outputs and labels to calculate inference loss
                        val_aux_outputs     = [pred_aux(i) for i in decollate_batch(val_aux_outputs)]
                        val_aux_labels      = [label_aux(i) for i in decollate_batch(val_aux_labels)]
                        
                        metric_aux(y_pred = val_aux_outputs, y = val_aux_labels)
                    else:
                        metric_aux(y_pred = val_aux_outputs, y = inputs.permute(0, 1, 3, 4, 2)) # Does this need to be deocllated?
                        
                # Compute the average metric value across all iterations
                main_metric = metric_main.aggregate().item()
                aux_metric  = metric_aux.aggregate().item()
                main_metric_values.append(main_metric)
                aux_metric_values.append(aux_metric)
                
                # Reset the metric for next validation run
                metric_main.reset()
                metric_aux.reset()

                # If the metric is better than the best seen so far, save the model
                if main_metric > best_metric:
                    best_metric = main_metric
                    best_metric_epoch = epoch
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print("saved new best metric model")
                
                print(
                    f"\nCurrent epoch: {epoch} current mean dice for main task: {main_metric:.4f}"
                    f"\nBest mean dice for main task: {best_metric:.4f} at epoch: {best_metric_epoch}"
                    f"\nCurrent epoch: {epoch} current mean metric for aux task: {aux_metric:.4f}"
                    )
                
    # When training is complete:
    print(f"Done training! Best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
    save_results(MODEL_NAME, MODEL_PATH, epoch_loss_values, epoch_aux_loss_values, epoch_total_loss_values, main_metric_values, aux_metric_values)
    

                    

        
        
        