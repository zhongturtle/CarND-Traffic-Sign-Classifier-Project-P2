# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./newimage/1 "Traffic Sign 1"
[image5]: ./newimage/14 "Traffic Sign 2"
[image6]: ./newimage/18 "Traffic Sign 3"
[image7]: ./newimage/22 "Traffic Sign 4"
[image8]: ./newimage/3 "Traffic Sign 5"
[image9]: ./newimage/37 "Traffic Sign 6"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/zhongturtle/CarND-Traffic-Sign-Classifier-Project-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration


I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution

![alt text][image1]

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because the most important fator of detect what's the image content is lines and shape , not the color

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because if it could allow us to get better result

My final model consisted of the following layers:
Layer 1: Convolutional. The output shape should be 28x28x6.

Activation. Your choice of activation function.

Pooling. The output shape should be 14x14x6.

Layer 2: Convolutional. The output shape should be 10x10x16.

Activation. Your choice of activation function.

Pooling. The output shape should be 5x5x16.

Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.

Layer 3: Fully Connected. This should have 120 outputs.

Activation. Your choice of activation function.

Layer 4: Fully Connected. This should have 84 outputs.

Activation. Your choice of activation function.

Layer 5: Fully Connected (Logits). This should have 10 outputs.


To train the model, I used an AdamOptimizer as optimizer; because of my computer gaphic card memory , I chhose 128 as batch size; to avoid lack of training , I raise the number of epoch to 50 . 

My final model results were:
* validation set accuracy of 0.967
* test set accuracy of 0.941

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  A:I choose Lenet to do this project
* What were some problems with the initial architecture? 
  A:No, it's quite good
* Which parameters were tuned? How were they adjusted and why?
  A:Yes, I just take the original version and add a parameter which is used to set dropout. The result is good
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  A:I think the CNN is powerful to handle the image so the result will be good. The dropout layer help us model not overfitting
* Why did you believe it would be relevant to the traffic sign application?
  A: I just try one time  and result is amazing. I believe the power of Lenet!!
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 A: the validation accuracy is 94% high . It's high enough for me 

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] 

The fifth image might be difficult to classify because the sign in the picture is small

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution      		| General caution   									| 
| Stop     			| Stop 										|
| Bumpy road					| Bumpy road										|
| Speed limit (60km/h)      		| Speed limit (80km/h)			 				|
| Speed limit (30km/h)			|Speed limit (30km/h)     							|
| Go straight or left   | Go straight or left |

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 83.3333%

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (30km/h) (probability of 0.99), and the image does contain a Speed limit (30km/h). The top five soft max probabilities were

Traffic Sign: Speed limit (30km/h)
Top 5 probabilities:

  Prediction                                              Probabilities
-----------------------------------------------------------------------
0.Speed limit (30km/h)                                        0.999995 
1.Roundabout mandatory                                        0.000004 
2.Keep right                                                  0.000000 
3.Speed limit (50km/h)                                        0.000000 
4.Go straight or left                                         0.000000 
=======================================================================

For the second image to the sixth image result :
Traffic Sign: General caution
Top 5 probabilities:

  Prediction                                              Probabilities
-----------------------------------------------------------------------
0.General caution                                             0.999984 
1.Traffic signals                                             0.000016 
2.Right-of-way at the next intersection                       0.000000 
3.Pedestrians                                                 0.000000 
4.Road work                                                   0.000000 
=======================================================================

Traffic Sign: Stop
Top 5 probabilities:

  Prediction                                              Probabilities
-----------------------------------------------------------------------
0.Stop                                                        0.936424 
1.Speed limit (30km/h)                                        0.058781 
2.Road work                                                   0.003898 
3.Go straight or right                                        0.000499 
4.Speed limit (60km/h)                                        0.000194 
=======================================================================

Traffic Sign: Bumpy road
Top 5 probabilities:

  Prediction                                              Probabilities
-----------------------------------------------------------------------
0.Bumpy road                                                  0.812383 
1.Bicycles crossing                                           0.169315 
2.Road work                                                   0.014031 
3.Children crossing                                           0.002408 
4.Keep left                                                   0.001325 
=======================================================================

Traffic Sign: Speed limit (60km/h)
Top 5 probabilities:

  Prediction                                              Probabilities
-----------------------------------------------------------------------
0.Speed limit (80km/h)                                        0.306110 
1.Speed limit (70km/h)                                        0.285769 
2.Speed limit (50km/h)                                        0.154573 
3.Speed limit (120km/h)                                       0.046861 
4.Speed limit (30km/h)                                        0.044978 
=======================================================================

Traffic Sign: Go straight or left
Top 5 probabilities:

  Prediction                                              Probabilities
-----------------------------------------------------------------------
0.Go straight or left                                         0.999860 
1.Roundabout mandatory                                        0.000078 
2.Traffic signals                                             0.000038 
3.Speed limit (30km/h)                                        0.000017 
4.Keep left                                                   0.000004 
=======================================================================



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


### Reference
- https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb
