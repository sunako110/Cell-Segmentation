# Cell Segmentation - UNET
Keras implementation of U-net for stem cell segmentation. This trained model is used for my graduate project. 
## Datasets
The dataset (provided by Tsinghua Yangtze Delta Research Institute) consists of 38 microscopy images of stem cells at various passages (P2, P6 and P7). The sizes of these images are all {$1024\times768$}. 
## Process Data
The U-Net architecture we use for this project takes input images with sizes equal to {$256\times256$}. In order to obtain more accurate cell segmentation results, the images in original dataset are cropped into {$256\times256$} patches. This is also to prevent distortion of the cell images during data augmentation. During augmentation, if the images were in their original sizes, they would be resized into {$256\times256$} hence producing a result dataset with lower qualities and distorted shapes which could not match with original dataset. 
