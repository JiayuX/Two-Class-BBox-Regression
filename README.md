# Two-Class-BBox-Regression
 This object detection project builds a two-class bounding box regressor to detect and classify dogs and cats. Due to the simplicity of the model, it requires that only one object appear in a picture belonging to one of the included classes. To achieve multi-object & multi-class detection, more complex models are expected such as those in R-CNN and YOLO families.

 To run the code, download the original dataset from https://www.kaggle.com/andrewmvd/dog-and-cat-detection?select=images and place the 'annotations' and 'images' folders in the dataset folder. 
 
 The .h5 file containing the pretrained weights of VGG16 (https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5) also need to be downloaded and placed in the 'VGG16' folder.
