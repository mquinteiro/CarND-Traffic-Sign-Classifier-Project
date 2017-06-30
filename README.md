**Traffic Sign Recognition**

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

[image1]: ./histo1.png "Visualization"
[image2]: ./trainSample.png "Grayscaling"
[image3]: ./transformations.png "Random Noise"
[image4]: ./newSigns//germanSignals.png
[image5]: ./selectedGT.png
[image6]: ./prediction1.png
[image7]: ./top5.png



You're reading it! and here is a link to my [project code](https://github.com/mquinteiro/CarND-Traffic-Sign-Classifier-Project/blob/master/Trafic_1.ipynb)


I used the pandas library load and mumpy to calculate summary statistics of the traffic
signs data set:


* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43


Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution of the images throw the different classes...

![alt text][image1]

A small sample of immages of X_train are showded using subplots

![samples of X_train][image2]


As a first step, I decided to normalize the images following the commendation
of (igame-128)/128 but at the end I've used a equalize_adapthist function from
the skimage.exposure package, that seems to work better, as a disadvantage
it is thousand times slower, for that reason I save normalized images for
future users.

With this normalization I achieve an accuracy up to 0.942 in the test dataset,
thats enought to the project but I decide to go further.

In the histogram I can see a lot of differences between the number of members
of different classes, so I decide to do fake images up to the maximum class
members.

For that proposse I did 4 transfomation functions:
move: that introduce a pad by random pixels
rotate: This fucntion introuce a random angle in the image
grow: that makes a zomm and a CarND-Traffic-Sign-Classifier-Project
gausian_noise: who introduce gausiona noise in the image.

The result of the same image manipulated could be viewed in the following image

![manipulated image][image3]

With the help of that images I fill all classes to homogeneize the number of
the members for all classes.


The model that I've used is LeNet adapted to this problem.



My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5 | 5,5,3 kernels to produce 28x28x6 |
| RELU1||
|Pooling|input 28x28x7 output 14x14x6|
| Convolution 5x5x6| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max Pooling|input 10x10x16. Output  5x5x16|
|Flatten|Input = 5x5x16. Output = 400|
|Fully Connected|Input = 400. Output = 120|
|RELU||
|Fully Connected| Input = 84. Output = 43|


To train the model, I used an AdamOptimizer with this parameters:
rate = 0.001
EPOCHS = 150
BATCH_SIZE = 128

Using the full training set I achieve this results:


My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.947
* test set accuracy of 0.940

*** Test in web Images ***

To test the model outside the controled data I have download images from thew
web.

![manipulated image][image4]

Ald seleect an small group for test

![selected signs][image5]

I have several difficult the first one is the way to upload to images, I get
it in diferent formats, I've use openCV numpy but with very bad results.
Finally I achive it using matplotlib.image package,

And normaile it as the other images.

In the first attemt with a reduced dataset the accuracy that I achieve is as low as , far ago from 0.9XX that I use to achieve in the dataset.

I get full accuracy in the blue ones, bot none in the yellow and red ones.
For me it is a mistery.

![prediction][image6]



Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Ahead only     		| Stop sign   									|
| 30 km/h     			| Priority Road 										|
| 50 km/h     			| Priority Road 										|
| 60 km/h     			| Priority Road 										|
| 70 km/h     			| Speed limit 60 										|
| 80 km/h     			| Priority Road 										|
| Go straight or right| Go straight or right|
| 100 km/h     			| Priority Road 										|
| Keep left| Keep left|
| 120 km/h					| 30 km/h											|
| Children crossing	| priority road|
| Roundabout mandatory| Roundabout mandatory|


The model was able to correctly guess 4 of the 12 traffic signs, which gives an accuracy of 33%. This compares favorably to the accuracy on the test set of 94.5 is very low.


I have check the top 5 probs of each sign, and the results could be located in
the following  image:

![top5][image7]
