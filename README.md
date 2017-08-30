# CarND-Vehicle-Detection-and-Tracking
Udacity CarND Vehicle Detection and Tracking Project

The goal of this project is to reliably identify the location of vehicles in a video stream of a front facing camera mounted centraly in a moving vehicle.

![alt text](videos/project_video_augmented.gif "Result")

## Project Structure

- `test_images/` Directory with images to test vechicle identification
- `examples/` Directory with example images used in the writeup
- `videos/` Directory with input and output videos
- `Vehicle-Detection-and-Tracking.ipnyb` Jupyter notebook with all the project code and example images
- `README.md` Projecte writeup (you're reading it)

## Project Overview

- **Reading Labeled Data** - We start by reading in the labeled data provided

- **Feature Extraction** - We then extract the desired features

- **Scalling Features** - We continue by scalling these features

- **Training the Classifier** - We train the classifier: a Linear Support Vector Classification, or SVC

- **Detecting Vehicles** - We use the previsouly trained classifier to identify vehicles in a video stream

- **Video Augmentation** - We finaly augment the video with the vehicles detected within rectangles

## Reading Labeled Data

We start by reading and analysing the provided data, this is separated into `vechicles` and `non-vehicles` data. The data set included **8792** examples of vehicle images and **8968** examples of non vehicle images.

Here's an example of the dataset:

> Vehicles

![alt text](examples/vechicles/vehicles-sample.png "Vehicles")

> Non Vehicles

![alt text](examples/non-vehicles/non-vechicles-sample.png "Non Vehicles")


## Feature Extraction

We then proceed to extract the required features from the dataset.

To achieve this I used a class, `ExtractFeatures` that abstracts away the feature identification, extranction and concatenation.

In this case we choose to extract 3 features, **spatial information**, **Histogram of Oriented Gradients, HOG** for short, and **color channel histogram**. 

```python
def features(self, x=0, y=0, s=64):
    """Returns a vector of the concatenated features"""
    features = []
    # Add spactial features
    spatial = self._bin_spatial(self.img[y:y + s, x:x + s, :])
    features.append(spatial)
    # Add HOG features 
    hog = self._hog(x, y, s)
    features.append(hog)
    # Add Histogram features
    hist = self._color_hist(self.img[y:y + s, x:x + s, :])
    features.append(hist)
    # Finally concatente them all and return
    return np.concatenate(features)
```

### Spacial Information

We simply resize and convert the image to a single dimension vector

```python
spacial_vector = cv2.resize(img, size).ravel()
```

### HOG, or Histogram of Oriented Gradients

After a certain amount of experimentation, where we trained and evaluated the classifier with diferent combinations of parameters, found a reasonably combination which produced high levels of accuracy and finally tested running the training multiple times until we reach an average accuracy of **98.9%**. 

This resulted in a **HOG** with:
 - `10` orientations
 - `8` pixels per cell
 - `2` cells per block
 
We used the `hog` function from `skimage.feature` like so: 

 ```python
 from skimage.feature import hog
 
 for channel in range(self.depth):
    hog_feature = hog(self.img[:, :, channel], 
                      orientations=10, 
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), 
                      transform_sqrt=True, 
                      visualise=False,
                      feature_vector=False)
    self.features_hog.append(hog_feature)
```

We were then capable of getting features for individual areas of the image by calculating the HOG offsets, `x` horizontal offset, `y` vertical and `s` the side of the square area

```python
_x = max((x // 8) - 1, 0)
_y = max((y // 8) - 1, 0)
_s = (s // 8) - 1

if (_x + _s) > self.features_hog.shape[2]:
    _x = self.features_hog.shape[2] - _s
    
if (_y + _s) > self.features_hog.shape[1]:
    _y = self.features_hog.shape[1] - _s

hog_region_features = np.ravel(self.features_hog[:, _y:_y + _s, _x:_x + _s, :, :, :])
```
 
 Here's an example of the HOG applied to a random dataset
 
 > Original: Y channel on YCbCr color space
 
 ![alt text](examples/y-color-space.png "Non Vehicles")
 
 > HOG
 
  ![alt text](examples/hog-set.png "Non Vehicles")
 

### Color Channel Histogram

Finally we separate individual color channels using `numpy` `histogram` function, using **16** bins and a range of **(0, 256)**

```python
# Compute the histogram of the color channels separately
channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
# Concatenate the histograms into a single feature vector
hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
```

> For full implementation details please see the [jupyter notebook](Vehicle-Detection-and-Tracking.ipynb)

## Scalling Features

We scale the features using `sklearn.preprocessing` `StandardScaler`. This allows us standardize the features by removing the mean and scalling to single varience.

Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual feature do not more or less look like standard normally distributed data.

```python
from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))
```

## Training the Classifier

For a classifier we've chosen a **Linear Support Vector Classification**. 

This classifier gives us very good results for this dataset having achieved an accuray of around **99%**

We've also seperated the **training** and **testing** data in a **80/20** fashion 
To implement the classifier we've used `sklearn.svm` `LinearSVC` class.

```python
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=43)
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
accuracy = round(linear_svc.score(X_test, y_test), 5)
print('Classifier Accuracy: {}'.format(accuracy))
```

## Detecting Vehicles


## Video Augmentation


## Discussion
