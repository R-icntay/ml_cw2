import numpy as np

from monai.transforms import (
    EnsureChannelFirstd, # Adjust or add the channel dimension of input data to ensure channel_first shape.
    CenterSpatialCropd,
    Compose,
    AsDiscrete,
    LoadImaged,
    ScaleIntensityd,
    Spacingd,
    SpatialPadd,
    RandAffined, 
    CropForegroundd, # Crop the foreground region of the input image based on the provided mask to help training and evaluation if the valid part is small in the whole medical image
    RandGaussianNoised, # Randomly add Gaussian noise to image.
    RandGaussianSmoothd, # Randomly smooth image with Gaussian filter.
    AdjustContrastd, # Adjust image contrast by gamma value.
)

def get_transforms():
    print("-" * 40)
    print("Creating transformations...")
    
    # Create transforms for training
    train_transforms = Compose(
        [
            LoadImaged(keys = ["image", "mask"]),
            EnsureChannelFirstd(keys = ["image", "mask"]),
            ScaleIntensityd(keys = "image"),
            CropForegroundd(keys = ["image", "mask"], source_key = "image"),
            Spacingd(
                keys = ["image", "mask"],
                pixdim = [0.75, 0.75, 2.5],
                mode = ("bilinear", "nearest"), # Interpolation mode for image and mask
            ),
            RandAffined(
                keys = ["image", "mask"],
                mode = ("bilinear", "nearest"),
                prob = 1.0,
                spatial_size = (256, 256, 40), # Output size of the image [height, width, depth]
                rotate_range = (np.pi / 36, np.pi / 36, np.pi / 36), # Rotation range
                scale_range = (0.1, 0.1, 0.1), # will do [-0.1, 0.1] scaling then add 1 so a scaling in the range [0.9, 1.1]
                padding_mode="zeros", # This means that the image will be padded with zeros, some images are smaller than 256x256x40
            ),
            RandGaussianNoised(
                keys = "image",
                prob = 0.15,
                mean = 0.0,
                std = 0.1
            ),
            RandGaussianSmoothd(
                keys = "image",
                prob = 0.1,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5)
            ),
            AdjustContrastd(
                keys = "image",
                gamma = 1.3
            )
        ]
    )

    # Create transforms for validation
    val_transforms = Compose(
        [
            LoadImaged(keys = ["image", "mask"]),
            EnsureChannelFirstd(keys = ["image", "mask"]),
            ScaleIntensityd(keys = "image"),
            Spacingd(
                keys = ["image", "mask"],
                pixdim = [0.75, 0.75, 2.5],
                mode = ("bilinear", "nearest"),
            ),
            # since we are not doing data augmentation during validation,
            #we simply center crop the image and mask to the specified size of [256, 256, 40]
            CenterSpatialCropd(keys = ["image", "mask"], roi_size = (256, 256, 40)), 
            SpatialPadd(keys = ["image", "mask"], spatial_size= (256, 256, 40)) # Some images are smaller than 256x256x40, so we pad them to this size
        ]
    )
    
    # Post transforms for the main prostate zones: 2 classes + background
    post_pred_transform_main    = Compose([AsDiscrete(argmax = True, to_onehot = 3)])
    post_label_transform_main   = Compose([AsDiscrete(to_onehot = 3)])

    # Post transforms for the auxilliary prostate zones: 3 classes + background
    post_pred_transform_aux_3   = Compose([AsDiscrete(argmax = True, to_onehot = 4)])
    post_label_transform_aux_3  = Compose([AsDiscrete(to_onehot = 4)])
    
    # Post transforms for the auxilliary prostate zones: 6 classes + background
    post_pred_transform_aux_6   = Compose([AsDiscrete(argmax = True, to_onehot = 7)])
    post_label_transform_aux_6  = Compose([AsDiscrete(to_onehot = 7)])
    
    print('Transforms have been defined.')
    
    return train_transforms, val_transforms, post_pred_transform_main, post_label_transform_main, post_pred_transform_aux_3, post_label_transform_aux_3, post_pred_transform_aux_6, post_label_transform_aux_6