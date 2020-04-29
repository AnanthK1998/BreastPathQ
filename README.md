# Cancer Cellularity Prediction using Deep Learning
This project was done towards providing a solution for the BreastPathQ challenge held as part of the SPIE Medical Imaging 2019 conference.
The BreastpathQ is a challenge which asks us to predict the cancer cellularity of the tissue in the image.
Cancer Cellularity can be defined as the ratio of volume of cancer cells present in the tissue to the volume of the whole tissue.
## Data
The dataset provided, consisted of 2394 images for training and 185 images for validation. The images were of the size (256x256). The output for each image was the cellularity between 0 and 1. 
Given below is an example of the data.
![image 1][/images/99788_1.tif]
The cellularity of this image is 0.97

## Training 
I treated this problem as an Image Regression one. I used the ResNet-18,34,50,101,152 architectures for training with batch size of 4 and a learning rate of 0.0001 (using adam optimizer) for 100 epochs. I used an ensemble of the best models for these architectures by averaging the result in the end.

I replaced the last activation layer of the ResNet models, which is a softmax layer, with a sigmoid activation layer to give us regression values between 0 and 1.

