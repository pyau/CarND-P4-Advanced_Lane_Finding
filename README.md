
##Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image]: ./camera_cal/calibration2.jpg
[image0]: ./output_images/corners_found11.jpg "corners"
[image1]: ./output_images/undistort13.jpg "Undistorted"
[image2]: ./test_images/test4.jpg "Road"
[image2k]: ./test_images/tracked_5.jpg "Road Transformed"
[image3]: ./output_images/threshold_5.jpg "Binary Example"
[image4]: ./output_images/perspective5.jpg "Warp Example"
[image5]: ./output_images/transformed_5.jpg "Fit Visual"
[image6]: ./output_images/warped_5.jpg "Output"
[video1]: ./output_tracked.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `camera_calibration.py`. This is how a distorted chessboard image look like. The top horizontal row, for example, is clearly curved.

![alt text][image]

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  Below image draws the chessboard corners on the distorted / uncorrected image that is found by `cv2.findChessboardCorners()`.

![alt text][image0]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result. Note that compare to the uncorrected image, you can see all the horizontal lines are straightened out.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The first image shows the original version, and the second image shows an undistorted version of the same image. The correction is not easily seen, but you can see the tree on the left side is cut off.

![alt text][image2]
![alt text][image2k]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 97 through 110 in `threshold.py`).  Here's an example of my output for this step. For color thresholding, I use a combined color thresholding of the S channel in HLS color model, and V channel in HSV color model.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in lines 22 to 48 in the file `lane_finding.py`  The `get_perspective_transform_matrix()` function takes as inputs source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
bot_width = 0.76
mid_width = 0.08
height_pct = 0.62
bottom_trim = 0.935
src = np.float32([[pimg_size[0]*(.5-mid_width/2), pimg_size[1]*height_pct],
                  [pimg_size[0]*(.5+mid_width/2), pimg_size[1]*height_pct],
                  [pimg_size[0]*(.5+bot_width/2), pimg_size[1]*bottom_trim],
                  [pimg_size[0]*(.5-bot_width/2), pimg_size[1]*bottom_trim],
                  ])
offset = pimg_size[0] * 0.25
dst = np.float32([[offset,0],
                  [pimg_size[0]-offset, 0],
                  [pimg_size[0]-offset, pimg_size[1]],
                  [offset, pimg_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 589, 446      | 320, 0        | 
| 691, 446      | 960, 0        |
| 1126, 673     | 960, 720      |
| 153, 673      | 320, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for sliding window to find lane line is found in the file `tracker.py`. In the `find_window_centroids()` function, we first found the window location where the convolution signal is the highest for left and right lane (line 25-28) for the bottom of the screen. Then for each vertical section (line 43), we find the window (within range controlled by the variable `margin`) in which the convolved signal is the strongest, and add that as centroid.

I added one improvement that should improve segmented lane lines that are farther apart from each other.  If the convolved signal is very low, then I just use the previous centroid value as the new vertical section's centroid (line 37-38 for left lane, 45-46 for right lane) in `tracker.py`.  This is equalivent to assuming ver low signal (no lane pixel found) as straight line.  This at least prevents the algorithm from incorrectly guessing left or right.

The results are then drawn in `lane_finding.py` lines 86-100.  Then the code to fit positions with a polynomial is found in `lane_finding.py` lines 102-127.  The below image shows a very smooth polynomial fit to the lane line turning right.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 142 through 158 in my code in `lane_finding.py`. This follows the curvature calculation described in http://www.intmath.com/applications-differentiation/8-radius-curvature.php. And the result is appended as text in lines 160-163 in `lane_finding.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 129 through 140 in `lane_finding.py`.  This uses the result of the polynomial for section 4, apply inverse transform (lines 136-137), and overlay the result to the base image. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_tracked.mp4)

Or this can be watched in youtube link below:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=blYe98LXVtU
" target="_blank"><img src="https://github.com/pyau/CarND-P4-Advanced_Lane_Finding/blob/master/video_screenshot.png?raw=true" 
alt="Click to watch video" width="480" height="360" border="10" /></a>


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline works reasonaly well for the project_video.mp4.  With more averaging applied, then the result should be smooth.  I had some problem with the algorithm suddenly detected the side of the road as lane line for a few frames. So I add code in `tracker.py` lines 30-38 to make sure the lane line starts at approximately the same location in the frame.

I ran my pipeline against the challenge video and the result was not satisfactory (polyfit does not yield good result, the lane will be interpreted as turning left as the frame is turning right, etc). I believe this can be fixed by experimenting different threshold functions.  The algorithm might have mistaken the shadow as part of the lane line, especially when the shadow and the lane line are close together.

In the harder challenging video, some frames do not have both lane lines captured at all. This pipeline should fail altogether.  Also, the vehicle is driving at a much slower speed. So values that are dependent on speed input should be dynamic (which the in vehicle system should have no problem obtain from CAN message, etc).  An example of this value would be meter per pixel in the y direction.