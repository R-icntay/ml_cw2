from pathlib import Path
import random
from shutil import copyfile
import os

def split_data(img_path, scale=1):
    """
    Read all images and divide into training, validation and test sets.
    Scale to test models on fewer data.
    """
    
    print("-" * 40)
    print("Splitting data into train-validate-test sets...")
    
    # Delete two images that do not have segmentation masks
    for file in ['001001_img.nii', '005057_img.nii']:
        if os.path.exists(Path(img_path / file)):
            os.remove(Path(img_path / file))
        else:
            print("The file does not exist")
        
    # Read all files ending with _img.nii
    img_files   = list(img_path.glob("*_img.nii")) # Image and mask are in the same folder
    num_images  = len(img_files)

    # Create train, validation and test splits
    train_split = int(0.8 * num_images / scale)
    val_split   = int(0.1 * num_images / scale)
    test_split  = int((num_images - (train_split + val_split)* scale) / scale)

    # Set the random seed for reproducibility
    random.seed(2022)
    
    # Shuffle the image files
    random.shuffle(img_files)

    # Split the dataset
    train_images    = img_files[:train_split]
    val_images      = img_files[train_split:(train_split + val_split)]
    test_images     = img_files[(train_split + val_split): int(num_images/scale)]

    # Create train, validation and test directories
    train_image_dir     = Path(img_path / "train_images")
    train_mask_dir      = Path(img_path / "train_masks")
    val_image_dir       = Path(img_path / "val_images")
    val_mask_dir        = Path(img_path / "val_masks")
    test_image_dir      = Path(img_path / "test_images")
    test_mask_dir       = Path(img_path / "test_masks")

    # Create the directories if they don't exist
    if not os.path.exists(train_image_dir) and not os.path.exists(train_mask_dir) and not os.path.exists(val_image_dir) and not os.path.exists(val_mask_dir) and not os.path.exists(test_image_dir) and not os.path.exists(test_mask_dir):
        for directory in [train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, test_image_dir, test_mask_dir]:
            directory.mkdir(exist_ok = True, parents = True)

        # Copy the images and their corresponding segmentation masks to their respective directories
        for directory, images in zip([train_image_dir, val_image_dir, test_image_dir], [train_images, val_images, test_images]):
            for image in images:
                # Copy image
                copyfile(image, directory / image.name)

                # Get corresponding segmentation mask
                mask = image.name.replace("_img.nii", "_mask.nii")

                # Copy segmentation mask
                copyfile(image.parent / mask, image.parent / directory.name.replace("images", "masks") / mask)

    # Put the train images and masks in a dictionary
    train_images    = sorted(train_image_dir.glob("*"))
    train_masks     = sorted(train_mask_dir.glob("*"))
    train_files     = [{"image": image_name, "mask": mask_name} for image_name, mask_name in zip(train_images, train_masks)]
    
    # Put the validation images and masks in a dictionary
    val_images      = sorted(val_image_dir.glob("*"))
    val_masks       = sorted(val_mask_dir.glob("*"))
    val_files       = [{"image": image_name, "mask": mask_name} for image_name, mask_name in zip(val_images, val_masks)]
    
    # Put the test images and masks in a dictionary
    test_images     = sorted(test_image_dir.glob("*"))
    test_masks      = sorted(test_mask_dir.glob("*"))
    test_files      = [{"image": image_name, "mask": mask_name} for image_name, mask_name in zip(test_images, test_masks)]
        
    print('Images have been divided into train-validate-test sets.')
    print('Total number of images: ', num_images)
    print('Number of images train-validate-test: ', train_split, '-', val_split, '-', test_split)

    return train_files, val_files, test_files
    