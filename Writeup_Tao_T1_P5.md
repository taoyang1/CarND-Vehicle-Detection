
## CarND-Term1-P5 writeup
### Tao Yang

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./results/quick_visual.png
[image2]: ./results/hog_examine.png
[image3]: ./results/slide_windows_1.png
[image4]: ./results/slide_windows_2.png
[image5]: ./results/heatmap.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 351 through 354 of the file called `CarND_T1_P5_Tao.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. For HOG, I'm only using the first channel of `YCrCb` color space.

![alt text][image2]

Note that the pattern is very different for car and no-car.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters:
 - `colorspace = ['RGB','HSV','LUV','HLS','YUV','YCrCb']`
 - `orientations = [6,7,8,9,10,11,12]`
 - `pixels_per_cell = [4,8,12,16]`
 - `cells_per_block = [1,2,3]`
 - `hog_channel = [0,1,2,'ALL']`

and the following parameters give the best result by examing the accuracy score as well as the running time:

`colorspace = 'YUV', orientations = 11, pixels_per_cell = 16, cells_per_lock = 2, hog_channel = 'ALL'`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using 1) HOG features, 2) spatial features (`spatial_size=(32,32)`), and 3) color histogram features (`hist_bins=32`). See code in line 406 through 468.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search code was borrowed from the course material. See code from line 262 through 344 (function `find_cars`). The major parameters to tweak in this step is the starting and ending position of `y` and the scale. The following steps are taken to determine those:
1. Visually examine the image region where the car appears. This gives roughly 400.
2. Try different scales see if the resulting bounding boxes can bound cars in different distance.

The following parameters are used as a result:
`params = [(400,464,1.0),(416,480,1.0),(400,496,1.5),(432,528,1.5),(400,528,2.0),(432,560,2.0),(400,596,3.5),(464,660,3.5)]`

For left to right of the tuple is `y_start`,`y_stop`, `scale`. By applying these parameters we got the following image (note that there is a false positive, and it will be taken care of later).
![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales scales and eight different starting y positions using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Below is the result of applying the sliding window approach to all the test images.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from one frame of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


### Here are one frame, its corresponding heatmap, the output of `scipy.ndimage.measurements.label()` on the integrated heatmap, and the resulting bounding boxes are drawn onto this frame:

![alt text][image5]

To make the algorithm more robust, I created a class to store the past detected bounding boxes for the vehicle (a maximum history of 15 frames are maintained). See pipeline `process_frame_history` in line 660 through 701.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Some thoughts about this project:
1. Testing on which parameter, or, feature engineering is tedious and requires a lot of time to find the best combination of parameters. 
2. The running time for the final model on video is slow. It takes about 5 minutes to finish the video and it certainly won't meet the real time requirement in actual autonomous car.

Any feedback on the above two points would be appreciated.



```python

```
