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

This is a multi-level classification task, so the loss function was appropriately selected to be Cross Entropy. Additionally, the optimizer used stochastic gradient descent (SGD) with a learning rate of 0.001, a momentum of 0.9, and a weight decay of $10^{-4}$. On top of that, a learning rate scheduler is built into the training script to scale the learning rate by 0.1 every 5 epochs.

## Evaluation

Even though the prediction set was used for the Streamlit demo, it was unlabeled, so we do not have the ground truth when analyzing that data. Instead, the model performance was evaluated on the validation set. This is a valid decision, as the validation set is well-generalized from the provided testing data. Therefore, we can assume that the model would perform similarly across the validation set and the testing set.

After training the CNN and using it to make predictions on the validation set, the following confusion matrix was generated to characterize the performance on the multi-level classification task.

![confusion_matrix](https://github.com/user-attachments/assets/508f5522-e7a1-40b0-b9ed-05865b0e3528)

From this matrix, we may calculate the following metrics for each class:
- **Buildings**
  - Number of Samples = 437
  - Accuracy = 92.22%
  - Precision = 94.60%
  - Recall = 92.22%
  - F1 Score = 93.40%
- **Forest**
  - Number of Samples = 474
  - Accuracy = 99.16%
  - Precision = 99.16%
  - Recall = 99.16%
  - F1 Score = 99.16%
- **Glacier**
  - Number of Samples = 553
  - Accuracy = 87.88%
  - Precision = 91.18%
  - Recall = 87.88%
  - F1 Score = 89.50%
- **Mountain**
  - Number of Samples = 525
  - Accuracy = 89.71%
  - Precision = 89.54%
  - Recall = 89.71%
  - F1 Score = 89.63%
- **Sea**
  - Number of Samples = 510
  - Accuracy = 98.63%
  - Precision = 94.55%
  - Recall = 98.63%
  - F1 Score = 96.55%
- **Street**
  - Number of Samples = 501
  - Accuracy = 95.01%
  - Precision = 93.52%
  - Recall = 95.01%
  - F1 Score = 94.26%
 
We may also compute the metrics that characterize the overall performance of the model:
- **Total Number of Samples** = 3000
- **Overall Accuracy** = 93.63%
- **Weighted Average Precision** = 93.62%
- **Weighted Average Recall** = 93.63%
- **Weighted Average F1 Score** = 93.61%

No official benchmark was provided by the Kaggle dataset; however, numerous users have uploaded code to demonstrate the performance of PyTorch-based deep learning approaches. One user uploaded a PyTorch-based CNN that was not powered by a separate pre-trained model, and they yielded 91.47% accuracy over the validation set from the Intel dataset (https://www.kaggle.com/code/ihalil95/98-train-92-test-accuracy-intelimgs-pytorch). Our model yielded improved accuracy compared to this CNN, which demonstrates the power of relying on a pre-trained VGG16 model. However, other users experienced more success with other pre-trained models, such as a particular ResNet-based network that performed with 93.7% overall accuracy (https://www.kaggle.com/code/payamamanat/pytorch-94-pretrainedresnet50). So, while our CNN with a pre-trained VGG16 model performs well, there is clearly room for improvement.
