## Multi-task learning

Multi-task learning has been motivated by an intuition that learning a good representation during one
or more auxiliary tasks may benefit a different task, the main task. This is particularly interesting when
learning the main task is challenging due to, for example, data scarcity and labels with high variance.
Medical image segmentation is such an example. In this project, segmentation of a particular set of
regions of interest (ROIs) in T2-weighted MR images of pelvic is useful for imaging-based surgical
planning for prostate cancer intervention. This main task may be assisted by segmenting a number of
other surrounding structures or classification tasks to discriminating whether certain ROIs are present.

This project aims

- to motivate and justify a main segmentation task, including understanding the
data set and its processing;

- to hypothesize what can be a viable set of auxiliary tasks that improve
the main task; 

- to design experiments to test the hypothesis;

- to develop deep neural networks for multi-task learning for these experiments; 

- to summarise and report relevant findings.

## What was done

In this project, four 3D unets were created:

- single decoder unet for the main task (segmentation of the prostate (Transition zone and Central gland))
- double decoder unet for the main task (segmentation of the prostate (Transition zone and Central gland)) and auxiliary task (segmentation of the bladder, seminal vesicle and rectum)
- double decoder unet for the main task (segmentation of the prostate (Transition zone and Central gland)) and auxiliary task (segmentation of the Bladder, Bone, Obturator internus, Transition zone, Central gland, Rectum, Seminal vesicle, Neurovascular bundle)
- double decoder unet for the main task (segmentation of the prostate (Transition zone and Central gland)) and auxiliary task (reconstruction of 3D input image)

The steps for creating, training and making inference for these models can be found in the folder **mtl_unet_models**.


![Multi-task learning 3D U-Net](images\architecture.png "Multi-task learning 3D U-Net")
