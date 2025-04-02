# AIPI 590 (Applied Computer Vision) - Project 1
### Author: Rajiv Raman
### Institution: Duke University
### Date: April 1st, 2025

## Overview

The project task was to train a convolutional neural network (CNN) that can predict at least 5 different types of "scenes" given an input image - a multi-class classification task. This repository features the code, the results, and the demo involved with building a modified VGG16-based CNN using transfer learning with an MLP classifier. 

You can view the Intel Image Classification dataset here: https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data. 

You can download the fully trained model here: https://huggingface.co/rajivsraman4/inteltorch/tree/main (the file is **best_model.pth**).

You can observe the model's predictions on a subset of the testing data here: https://inteltorchdemo.streamlit.app/.

## Description

The main branch contains many different files. I have detailed all their purposes here.

1. **app.py** - the Python script for the Streamlit demo.
2. **evalmodel.py** - the Python script for evaluating the model's performance on the labeled validation set.
3. **full_predictions.csv** - the model's predictions on the full set of testing data.
4. **predictions.csv** - the model's prediction on the small subset of testing data for the Streamlit demo.
5. **requirements.txt** - the package requirements for all the code in this repository.
6. **testmodel.py** - the Python script for generating a new "predictions.csv" file from the testing set.
7. **trainmodel.py** - the Python script for training the neural network from scratch.

To properly use the scripts, you need to follow the correct practices for importing the data. This will be described later in the writeup.

## Data

All image data is sourced from the Intel Image Classification dataset on Kaggle (https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data). Each image displays one of six possible scenes: buildings, forest, glacier, mountain, sea, or street. The model is trained to differentiate between these six classes.

It is worth noting that there are no ground truth labels provided for the testing set (seg_pred) in the dataset. Because labeling by hand would be inconvenient, we simply use the validation set (seg_test) for computing our evaluation metrics, especially because they are well-generalized from the testing set.

## Usage

The Streamlit demo (https://inteltorchdemo.streamlit.app/) can be accessed here, but it only operates based on the model that I personally trained. If you are interested in training the model or computing predictions yourself, you will need to follow these instructions.

1. Navigate to the directory where you want to store your code.
2. Clone this repository into your local machine via Terminal or Command Prompt: `git clone https://github.com/rajivsraman/intel-torch.git`
3. Install the requirements for the project: `pip install -r requirements.txt`
4. Delete the seg_pred folder found inside the data folder.
5. Download the Intel Image Classification dataset from Kaggle: https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data.
6. Extract the three folders (seg_pred, seg_test, and seg_train) and move them into the data folder for this project.
7. [Optional] Download my trained CNN from Hugging Face (https://huggingface.co/rajivsraman4/inteltorch/tree/main) and add it to your project folder.

Now, you are set up to run any of the scripts. However, if you choose to skip Step 7, then you will need to train the model yourself. You can accomplish this by running **trainmodel.py**.

## Model

We start by loading a VGG16 model, which is a classic deep CNN with 13 convolutional layers, 3 fully connected layers, and 16 weight layers in its architecture. This model is pre-trained on ImageNet (a large dataset of images), and although it is known for its deep architecture, it does not contain extra features like residual connections or attention mechanisms. Despite that, it is still extremely powerful, boasting over 130 million parameters.

After the VGG16 model is loaded, we modify the 3 fully connected layers by adding a custom classifier for the 6 classes in the dataset.
- Layer 1: this is a linear layer that takes 25,088 input features and outputs 4096 features. It is followed by a ReLU activation function and a dropout layer.
- Layer 2: this is a linear layer that takes 4096 input features and outputs 1024 features. It is followed by a ReLU activation function and a dropout layer.
- Layer 3: this is a linear layer that takes 1024 input features and outputs the 6 classes.
