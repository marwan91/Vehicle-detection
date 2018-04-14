
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_and_notcar.jpg
[image2]: ./examples/car_not_car_hog.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/all_test_images.jpg
[image5]: ./examples/video_frames.jpg
[image6]: ./examples/labeled.jpg
[image7]: ./examples/final_image.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook ( lines 22 through 108 of  `vehicle_detect.ipynb`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I had to make a compromise between computation speed and the quality of results, I chose the parameters that would save a lot of computation time but at the same time I maintained the quality of the results required for a working pipeline.

I have not taken into account the classifier test accuracy when chosing the parameters because I have found that the test accuracy is not strongly correlated to quality of the results of classifying the test images later to be shown.   

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `sklearn.svm` library and trained the classifer as shown in the first  code cell (lines 194 to 198).
I experimented at the beginning with both linear and rbf SVM classifiers , but found the Linear classifer to work faster and produce higher accuracy.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at pre defined scales within a defined portion of the image as shown in the third code cell (lines 173 to 180 ).
I chose the defined scales and also the step size for each scale , based on experimentation.
I use a window that is 64x64 pixels in size , and move it accross each test image, then perform feature extraction on each area covered by the window, then I use the classifier to predict whether the image corresponds to a vehicle or not.



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

To augment the performance of the classifer I generated heat maps which combine the predictions of the classifier for different scales, then I applied thresholding to filter out the bounding boxes that are likely to be false positives.

To improve the classifier performace even more , I filtered out the bounding boxes that are too small to represent a vehicle as shown in the third code cell (lines 80 through 81).

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the heat map in each frame of the video.  From the positive detections I created another heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest prolem I faced was the intolerable amount of false positive predictions yielded by the classifier. This is due to choosing parameters that reduced the computation time but made it harder for the classifier to distinguish between vehicle and non vehicle images. To fix that problem , different parameters need to be passed to the HOG function. 

The pipeline is likely to fail in such cases:
*objects that appear on the road but are not vehicles. Such objects might be mistaken for vehicles , or worse , might not be detected at all.
*During night time , it is harder to detect edges , which are an important feature in the image to be classified.
*During rain the road surface becomes reflective and that can confuse the classifier.

