
kamyczki - v2 2025-11-16 8:50pm
==============================

This dataset was exported via roboflow.com on November 16, 2025 at 7:51 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 339 images.
Kamienie-kurwa are annotated in YOLOv8 format.

The following pre-processing was applied to each image:

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Salt and pepper noise was applied to 0.3 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* Random brigthness adjustment of between -15 and +15 percent
* Random exposure adjustment of between -10 and +10 percent


