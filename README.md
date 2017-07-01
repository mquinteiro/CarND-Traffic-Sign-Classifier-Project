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

[image1]: ./classDistribution.png "Visualization"
[image2]: ./trainSample.png "Grayscaling"
[image3]: ./transformations.png "Random Noise"
[image4]: ./signals2.png
[image6]: ./prediction1.png
[image7]: ./ntop5.png
[image8]: ./signals2.png
[image9]: ./lrate1.png
[image10]: ./lrate01.png
[image11]: ./lrate005.png




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



My first model consisted of the following layers:

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

Not bad, but... I try harder and change several thinks, intermedian inputs and
outputs, different EPOCHS and BATCH_SIZEs and rate learning, no big differences
founded.

So I decide to test with regularization but I dind't find how to, and dropouts.

With the dropouts things changed, I use a 75% of keep probability and the behavior
is a slower training with less overfiting.
I put a dropout in each intermediate layer. Using a keep probability of one in the
prediction o evaluation processes

Running with diferent arameters got:

Rate= **0.001**
Dropout = 0.25
EPOCHS = 65
Validation accuracy = 0.962
Test accuracy = 0.931

![Rate=0.001][image9]


Rate= **0.0001**
Dropout = 0.25
EPOCHS = 65
Validation accuracy = 0.962
Test accuracy = 0.951

![Rate=0.0001][image10]

Rate= **0.0005**
Dropout = 0.30
EPOCHS = 550
Validation accuracy = 0.950
Test accuracy = 0.937
Epoch scale x10

![Rate=0.00005][image11]

So we stay in:
Rate= **0.0001**
Dropout = 0.25
EPOCHS = 65
Validation accuracy = 0.962
Test accuracy = 0.951


*** Test in web Images ***

To test the model outside the controlled data I have download images from thew
web.

![manipulated image][image4]

Ad seleect an small group for test

![selected signs][image5]

I have several difficult the first one is the way to upload to images, I get
it in diferent formats, I've use openCV numpy but with very bad results.
Finally I achive it using matplotlib.image package,

And normalize it as the other images.

In the first attempt with a reduced dataset the accuracy that I achieve is as low
 as , far ago from 0.9XX that I use to achieve in the dataset.

I get full accuracy in the blue ones, bot none in the yellow and red ones.
For me it is a mistery.

![prediction][image6]



Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|Yield|Yield|
|Roundabout mandatory|Roundabout mandatory|
|Stop|Stop|
|Speed limit (50km/h)|Speed limit (50km/h)|
|No passing|No passing|
|Speed limit (70km/h)|Speed limit (70km/h)|
|Turn right ahead|Turn right ahead|
|Go straight or right|Go straight or right|
|Priority road|Priority road|
|Double curve|***Children crossing***|
|Right-of-way at the next intersection|Right-of-way at the next intersection|
|Bicycles crossing|Bicycles crossing|


With this new model the prediction accuracy is 91.7% that good!

Also I have take a look to the different probabilities of each sing and especially to the one that was wrong prediction.

Real :No passing
No passing 0.301
Speed limit (60km/h) 0.129
Speed limit (20km/h) 0.074

Real :Double curve
Children crossing 0.911
Dangerous curve to the right 0.036
General caution 0.017

But taking a closer look we can see that prediction is very good.

I don't like the 5th probs quality, but look the photo I culd see that it have a ***watermark***

Looking the second one I can see that is double curve, but takin a closer look to examples of it class I detect that it is another sign!!!! it is first curve to the right and the dataset the curve is to the left.


## Accuracy conclusions:

Analizing the wrong result we can see which are the worse signal classes.

The following table she classes with more errors:

|C| % | mistake|
|:---------------------:|:---------------------------------------------:|
|3|0.891| [ 5 10  5  5  5  5  5  2  5  5  5  5  2  5  5  5  5  5  5  5  2  5  5  5  5   2  7  2  5  5  5  5  5  5  2  5  5  5  5  5 28 10  5  5  2  5 10  5  2] |
|5|0.863| [ 3  7  2 20  7  2  2  3  3  1  2  8  8  2  8  2  2 10 20  1  2  2  4  2  6  7  7  3  8  2  2  2  2  2  2  7  2  2  2  1  2  2  3  2  2  2  2  3  2  3  6  2  2  2 10  3  3  7  2  2  3  8  3  2  2  3  2  2  7  2  2 10  3 20  2 3  2  2  3  3  2  2  3  2  2  8] |
|18|0.877| [30 27 27 20 24 27 26 28 26 28 30 26 30 30 28 26 11 28 22 20 27 30 27 27 11 28 27 27 30 11 28 27 41 30 30 26 30 27 27 27 20 30 26 28 27 28 21 21] |
|20|0.789| [30 28 26 30 30 25 30 30 29 30 30 28 28 23 30 30 29 30 30] |
|21|0.644| [23 23 23 23 23 23 23 23 23 23 23 23 23 23 31 23 23 28 23 23 23 23 31 23 23 23 19 23 23 23 23 23] |
|27|0.817| [11 29 29 11 11 11 11 26 11 29 11] |
|30|0.753| [31 29 31 29 23 11 29 21 29 23 29  8 20 20 31 29 29 29 29 20 31 29 29 11 20 31 29 20 11 11 23 29 20 11 20 31 29] |
|42|0.800| [41  6 41 41  6 41  6  6  6 41 40  6 41 41  6 41  6 41] |

All of them are very similar and the worst is the ice.

I suppose that the reason could be the low resolution that makes the central picture of the sing a big black dot.
