# Densenet Prediction

This module is intended for running the winning model from the HPA Kaggle
challenge to generate feature files and prediction files.

## Before using this module

1. The repository does not include the model files themselves, which can
   instead be downloaded from here.

2. The repository does not include any images to predict on. They can for
   example be downloaded from the [Human_Protein Atlas](https://www.proteinatlas.org).


## Preparing image data for prediction

In the `predict` folder, there is a script named `resize_image.py`. The script,
as the name suggests, will resize the input images into a format that works
well for the winning challenge model.

To run the script, run the following

```bash
python resize_image.py --src_dir ${src_dir} --dst_dir ${dst_dir} --size ${size}
```

Where `${src_dir}` is the directory where your input images are located,
`${dst_dir}` is the directory where you would like the resized images to end
up, and `${size}` is the size which the images should be resized into. After
the script has finished, there will be a folder `images_${size}` in the
`${dst_dir}`.

## Densenet predictions / feature extraction

### Model configuration

Also in the predict folder, there is a script named `predict_d121.py` which will run the
winning densenet model on the images.  These scripts contain a python class
called `Config` which will need to be edited to work for your setup.


Following is the `Config` class description, as found in the code. You should edit the
source code file to modify the relevant parts for you.

```python
class Config(object):
"""This class acts as the configuration for all model functions.

This class essentially acts as a python dict with information for the
running of model predictions.

The following fields are in the class, and should be changed according
to your system and preferences:

num_workers: The number of threads that runs prediction, as an integer.
             Note that the number of available GPUs may impact how
             many threads you can safely run.
batch_size: The size of each prediction batch run by the threads, as an integer.
            This should not be too large for your GPU to fit the images
            into memory.
num_classes: The number of classes that the model predicts, as an integer.
             The HPA Kaggle challenge winning model specifically requires 28 classes.
image_size: The image size as an integer, in pixels, that will be used as input for the network.
            Each image should be <image_size>x<image_size> large.
            The HPA Kaggle challenge winning model should specifically use 1536 pixels here.
crop_size: The image size as an integer, in pixels, that will be cropped from the input images.
            The HPA Kaggle challenge winning model should specifically use 1024 pixels here.
in_channels: The number of input channels for the network, as an integer, usually 3 or 4.
             When working with HPA data, use 3 for RBG and 4 for RGBY.
             The HPA Kaggle challenge winning model should use 4 here.
seeds: The random seeds to be used for random crops, as a list of integers.
       A value of [0] means no crop.
       To recreate the results from HPA papers using the HPA Kaggle challenge winning model,
       seeds should be equal to [0] (no crop).
augments: A list of strings describing with test time augmentations to use.
          To recreate the results from HPA papers using the HPA Kaggle challenge winning model,
          seeds should be equal to ["default"].
model_name: The name of the model to be used for predictions, as a string.
model_path: A file path, as a string, to the pytorch model file.
suffix: A string with the file ending of all input images, for example "jpg".
result_name: A string containing a name to be used as part of the output file names.
out_dir: A file path, as a string, to a folder in which prediction results will be stored.
features_dir: A file path, as a string, to a folder in which feature outputs will be stored.
# The following two fields are ignored in favor of command line parameters instead.
image_dir: This field is ignored and instead the corresponding command line argument is used.
gpus: This field is ignored and instead the command line parameter is used.
"""
```

### Running the model

To run the model, run the command `python predict_d121.py --gpus 0,1 --image_dir /path/to/your/resized/images_1536/`.

The argument to `--gpus` can be anything that is valid for the `CUDA_VISIBLE_DEVICES` environment variable.

The argument to `--image_dir` should be the path to your resized images, as resized by the resize script above.

The result files will end up in the folders specified in the `Config`
class during the previous step.

## A small example

`predict_d121.py` is configured to work with the example images found
in the folder `exampleimages`, assuming you run the command `python
predict_d121.py` from the `predict` folder.

Note that the default setup assumes you have at least two available
Nvidia GPU capable of Cuda calculations.

To run the example, run the following commands from the project root folder:

```bash
mkdir models
curl https://sandbox.zenodo.org/record/920709/files/final.pth -o models/model.pth
cd predict
# pip install is only needed if you haven't installed the required packages previously
pip install -r requirements.txt
python resize_image.py --src_dir ../exampleimages --dst_dir ../resized_images/ --size 1536
python predict_d121.py --gpus "0" --image_dir ../resized_images/images_1536 --out_dir ../results/
```

The results will end up in the folders `../features` and `../results`
for result features and predictions respectively.

If you use the default example images, you should see the following results:
```
         Id Predicted
0  800_A6_1        25
1   89_B5_1         6
```
