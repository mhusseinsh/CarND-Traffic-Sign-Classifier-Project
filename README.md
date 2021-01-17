# **Traffic Sign Recognition** 

## Report

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[allSigns]: ./images/allSigns.png "All Signs"
[trainDistribution]: ./images/trainDistribution.png "Train Set Distribution"
[testDistribution]: ./images/testDistribution.png "Test Set Distribution"
[validDistribution]: ./images/validDistribution.png "Valid Set Distribution"
[preProcess]: ./images/preprocess.png "Preprocessing"
[model1]: ./images/model1.png "Model 1"
[model2]: ./images/model2.png "Model 2"
[model3]: ./images/model3.png "Model 3"
[testData]: ./images/testData.png "Test Data"
[testDataProbabilities]: ./images/testDataProbabilities.png "Test Data Probabilities"
[testDataBars]: ./images/testDataBars.png "Test Data Bars"
[challengeData]: ./images/challengeData.png "Challenging Data"
[challengeDataProbabilities]: ./images/challengeDataProbabilities.png "Challenging DataProbabilities"
[challengeDataBars]: ./images/challengeDataBars.png "Challenging Data Bars"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mhusseinsh/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is <em>34799</em> samples
* The size of the validation set is <em>4410</em> samples
* The size of test set is <em>12630</em> samples
* The shape of a traffic sign image is <em>(32, 32, 3)</em>
* The number of unique classes/labels in the data set is <em>43</em>

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Below is an overview of all signs available in the dataset. Each image represents a random one from a corresponding sign.

![alt text][allSigns]

The below bar charts represent the data distribution for the train, test and validation sets. It shows the number of available examples per class.
![alt text][trainDistribution]

![alt text][testDistribution]

![alt text][validDistribution]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I had a look randomly on the images represented in the dataset, and it appears that most of them are dark. This shows that the only little values available in the color channels in the images will not really affect the results because they do not contain much information about the image.

Converting the images to grayscale will help reduces the amount of features from <em>3072</em> to <em>1024</em> per image, which will help in reducing the amount of training parameters as well as the execution time.

Additionally, a normalization of the images was done to convert the pixel int values of range <em>[0, 255]</em> to float values of range <em>[-1, 1]</em>.

Normalization normally helps since neural networks process inputs using small weight values, and inputs with large integer values can disrupt or slow down the learning process. In this process, we want or each feature to have a similar range so that our gradients don't go out of control. That's why normalization is quite important.

A visualization of a sample image before and after preprocessing is shown below:

![alt text][preProcess]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model architecture is based on the original [LeNet model architecture](https://www.researchgate.net/figure/Architecture-of-LeNet-5-a-Convolutional-Neural-Network-here-for-digits-recognition_fig1_2985446) which consists of 2 convolutional layers followed by RELU activation function and max pooling layer, then 3 fully connected layer and a softmax output of 10 classes in the end.

I modified the architecture by adding 3 dropout layers after the fully connected layers to avoid overfitting, as well as changing the output of the last layer to 43 classes instead of 10 based on the size of the labels available in the dataset I am using.

| Layer         		    |     Description	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Input         	    	| 32x32x1 Grayscale image   					|
| Convolution 5x5       	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					    |												|
| Max pooling	        	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5       	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					    |												|
| Max pooling	      	    | 2x2 stride,  outputs 5x5x16 				    |
| Flatten       		    | outputs 400        							|
| <strong>Dropout</strong>  |                   							|
| Fully Connected			| outputs 120        							|
| RELU					    |												|
| <strong>Dropout</strong>  |                   							|
| Fully Connected			| outputs 84        							|
| RELU					    |												|
| <strong>Dropout</strong>  |                   							|
| Fully Connected			| outputs 43        							|
| Softmax       			|                   							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, [Adam Optimizer](https://arxiv.org/abs/1412.6980) was used as the optimization method for the training operation to minimize the training loss where the [Cross-Entropy](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy) function was used as the network's loss function.

My final hyperparameters were:
* <strong>Epochs</strong>: 20
* <strong>Batch Size</strong>: 16 
* <strong>Learning Rate</strong>: 0.001 
* <strong>Dropout Probability</strong>: 0.6
In addition to initializing the model variables by using truncated normal distribution with mu = 0.0 and sigma = 0.1.

The model produced final accuracies as follows:
* <strong>Training Accuracy</strong>: 94.98%
* <strong>Validation Accuracy</strong>: 99.2%
* <strong>Test Accuracy</strong>: 95.04%

![alt text][model1]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

If an iterative approach was chosen:
* <strong>What was the first architecture that was tried and why was it chosen?</strong>
  
    Initially, the [LeNet model architecture](https://www.researchgate.net/figure/Architecture-of-LeNet-5-a-Convolutional-Neural-Network-here-for-digits-recognition_fig1_2985446) was used for the task of traffic sign classification. The model was adapted to change the input layer shape from <em>32x32x1</em> to <em>32x32x3</em> to accept our traffic sign images which have 3 colorchannels, also the output nodes were changed to 43 instead of 10, since we have 43 unique classes.
* <strong>What were some problems with the initial architecture?</strong>
  
    The model architecture did not perform well on RGB images reaching less than <strong>85%</strong> of accuracy. The images were explored and as I described above, the RGB images was not that helpful due to the few amount of information that the colors have, as well as the training time was taking so long.
* <strong>How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.</strong>
  
    It was then decided to go back to the original input layer of <em>32x32x1</em> and perform the preprocessing which was described above for grayscaling and normalizing the dataset, so now the images have a shape of <em>32x32x1</em>, and we are back to the original architecture. I started training with <strong><em>EPOCHS=25, LEARNING_RATE=0.001, BATCH_SIZE=128</em></strong>. Upon playing around and changing the parameters and retesting, it was noticed that the training accuracy was very high upon each epoch, reaching <strong>99+%</strong>, however the validation accuracy was not increasing that much.

    The model is definitely overfitting. In order to solve this problem, a dropout layer was added after each fully connected layer to reduce the overfitting.

    So a dropout probability of 0.5 was added during training, and it is set to 1.0 while validating and testing. The model performed much better and the validation accuracy was exceeding <strong>95%</strong>. The test accuracy afterwards was around <strong>90%</strong>. This is for sure dependent on the hyperparameters which I was playing around and trying different combinations, but what I am mentioning was more or less the average accuracy which I got.
* <strong>Which parameters were tuned? How were they adjusted and why?</strong>
    
    The task now was instead of randomly playing around with the parameters, it is time to search and choose the optimal hyperparameter values.


    To perform this task, a [Grid Search](https://medium.com/fintechexplained/what-is-grid-search-c01fe886ef0a) was done to obtain the optimal hyperparameter values with a search within 4 hyperparameters in total, and these are:
    * <strong>Epochs</strong>: [15, 20, 25, 30]
    * <strong>Batch Size</strong>: [16, 32, 64, 128, 256]
    * <strong>Learning Rate</strong>: [0.0005, 0.0007, 0.0009, 0.001]
    * <strong>Dropout Probability</strong>: [0.4, 0.5, 0.6, 0.7]

    During each training, the training and validation accuracies were calculated and logged, and after the training ends, the test accuracy is computed, and the model with the highest test accuracy was considered to be the best performing model.

    Since I covered most of <em>batch_size, learning_rate, and dropout</em> for 15 and 20 epochs, and I did not find a major increase in the results for the same hyperparameters in both 15 and 20 epochs, so I stopped searching, since increasing more epochs won't affect the results that much

    Below are the results of the top 3 models sorted by test accuracy:

    
    | Epochs        | Batch Size    | Learning Rate    | Dropout Probability    | Training Accuracy    | Validation Accuracy    | Test Accuracy    |
    |:-------------:|:-------------:|:-------------:| :-------------:| :-------------:| :-------------:| :-------------:|
    | 20 | 16 | 0.001 | 0.6 | 99.2% | 95.8% |95.04% |
    | 15 | 16 | 0.0009 | 0.7 | 99.47% | 95.85% |94.89% |
    | 20 | 16 | 0.0009 | 0.6 | 99.35% | 96.82% |94.7% |


    Model 1             |  Model 2 |  Model 3
    :-------------------------:|:-------------------------:|:-------------------------:
    ![alt text][model1] | ![alt text][model2] | ![alt text][model3]

* <strong>What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?</strong>
  
    Deep learning neural networks are likely to quickly overfit a training dataset with few examples.

    Dropout is considered to be a regularization technique that approximates training a large number of neural networks with different architectures in parallel.

    During training, some number of layer outputs are randomly ignored or “<em>dropped out.</em>” This has the effect of making the layer look-like and be treated-like a layer with a different number of nodes and connectivity to the prior layer. In effect, each update to a layer during training is performed with a different “view” of the configured layer.

    So dropout has the effect of making the training process noisy, forcing nodes within a layer to probabilistically take on more or less responsibility for the inputs.

    This conceptualization suggests that perhaps dropout breaks-up situations where network layers co-adapt to correct mistakes from prior layers, in turn making the model more robust.
    
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 5 German traffic signs that I found on the web:

![alt text][testData]

Also I included some challenging images which I will explain later why they were challenging

![alt text][challengeData]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Go straight or right      		| Go straight or right   									| 
| Right of way at the next intersection     			| Right of way at the next intersection 										|
| Keep right					| Keep right											|
| Slippery road	      		| Slippery road					 				|
| Roundabout mandatory			| Roundabout mandatory      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.04% which i got from the model.

However for the challenging images, the model did not perform well at all. The results as below:

| Image	 |  Prediction	 | 
|:---------------------:|:---------------------------------------------:| 
| Vehicles over 3.5 metric tons prohibited| No passing for vehicles over 3.5 metric tons| 
| Road work | Wild animals crossing |
| Speed limit (30km/h) | Speed limit (80km/h) |

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

A visulaization of the top 5 softmax probabilities of the predicted images is shown below:
![alttext][testDataProbabilities]

![alttext][testDataBars]

For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


