# End-to-End Unsupervised Learning for Sonar Image Registration
In this project, we analyze state-of-the-art unsupervised learning architecture of the image registration technique such as DLIRNet [1], which can be applied to perform registration on unseen image pairs in one pass. The neural network architecture obtains the parameters from different forms of ConvNet by inputting a pair of moving and fixed images. The output parameters (affine transformation) will be passed to a spatial transformer [2] and a resampler which wraps the moving image into the fixed image. The performance will be evaluated by a similarity metric (NCC) between the wrapped and fixed images. Besides, we replace the ConvNet in DLIRNet with Inception blocks [4] and Dense blocks [3] in this project.

## Data
To train a neural network effectively, a substantial volume of datasets is indispensable. Specifically, our study necessitates a significant quantity of sonar image pairs. We have compiled tailored datasets designated for training, validation, and testing purposes. The data was acquired within a water tank environment by maneuvering a forward-looking sonar across a scene.

## Architecture
| <img src="./figures/AIRNet.png" width="80%">| 
|:--:| 
| *Architecture* |

The figure above shows our architecture of the model that follows the DLIRNet baseline [1]. We use the regression model to encode 2 patches of images to extract spatial information. Then, flatten and concatenate the Inception output feature map and pass to fully connected layers that produce the 1 by 6 vector which are predicted affine transformation parameters. Last, apply a spatial transformation layer in our baseline which inputs moving and predicted affine parameters and wraps the moving sonar image. 

The negative normalized cross-correlation (NCC) is chosen to train the network. The NCC measures the similarity between two images based on their intensity values. The negative NCC is often used as a loss function because optimization algorithms typically minimize a loss function to find the optimal transformation parameters for image registration.

## Image Preprocessing
To effectively train the image registration model, it is essential that the input images possess uniform width and height dimensions. To maximize the field of view offered by sonar images, we intend to retain the entire scope of the sonar image without cropping specific regions. This involves converting the recorded sonar images into Range-azimuth images. 

According to the sonar calibration, we have determined that the minimum bottom range in the original Oculus sonar image corresponds to the image coordinate (959, 1025). Additionally, the Range value at this coordinate is 971. Utilizing this contextual information, we can employ specific formulas to convert each pixel value within the image plane to its corresponding Range-azimuth representation.

The Range resolution of the Oculus sonar image is 0.0025 meters in 0 to 3.5 meters. The azimuth resolution is 0.25 degrees in -30 to 30 degrees. In sum, we can have a 1400 x 267 Range-azimuth image. The original Oculus sonar image and Range-Azimuth image are shown below.

Original |  Range-Azimuth |
:-------------------------:|:-------------------------:
<img src="./figures/Oculus3-6.jpg" width="45%">  |  <img src="./figures/RangeAzimuth.png"> 

## Usage
To train the model, use the following command. The code will save the checkpoints of every epoch.
```
python3 Train.py --DataPath {Directory} --NumEpochs{default 50} --MiniBatchSize {Default 32} --LearningRate {Default 0.001} --LrDecayStep {Defualt 0.5} --CheckPointPath {Directory} --LoadCheckPoint {Default 0}
```
* **DataPath**: Base path of images
* **NumEpochs**: Number of Epochs to Train for
* **MiniBatchSize**: Size of the MiniBatch to use
* **LearningRate**: Training Learning rate
* **LrDecayStep**: Period of learning rate decay
* **CheckPointPath**: Path to save Checkpoints
* **LoadCheckPoint**: Load Model from latest Checkpoint from CheckPointsPath for continuous training


To test the model, use the command below. The testing script will show the NCC score of each epoch
```
python3 Test.py --NumEpochs {Number of training epoch} --CkptsPath {Directory of saved checkpoints} --BasePath {Directory of testing data}
```


## Visualization & Performance
The figures below demonstrate the result of DLIRNets. 

| <img src="./figures/demo_0.png">| <img src="./figures/demo_7.png">| <img src="./figures/demo_5.png">| 
|:--:|:--:|:--:| 
| *DLIRNet* | *Inception* | *Dense* |

In addition, we have shown the NCC score of each architecture.
| NCC \ Model       | DLIRNet     | Inception    |  Dense        |
| :---        |    :----:       |     :----:       |    :----:       |
| Training | 0.737 | **0.837**  | 0.755 |
| Testing | 0.716 | **0.815**  | 0.738 |

## Reference
1. De Vos, B. D., Berendsen, F. F., Viergever, M. A., Staring, M., & Išgum, I. (2017). End-to-end unsupervised deformable image registration with a convolutional neural network. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support: Third International Workshop, DLMIA 2017, and 7th International Workshop, ML-CDS 2017, Held in Conjunction with MICCAI 2017, Québec City, QC, Canada, September 14, Proceedings 3 (pp. 204-212). Springer International Publishing.
2. Jaderberg, M., Simonyan, K., & Zisserman, A. (2015). Spatial transformer networks. Advances in neural information processing systems, 28.
3. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4700-4708).
4. Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
