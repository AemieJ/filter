# Filter

Python version : 3.6.7.

The following python files will be able to execute different filters . In snapchat , instagram , photobooth , etc there are many different filters which are based on facial recognition .

The code is open source and as such , the code will execute a filter on your eyes . I have used opencv , dlib and numpy to create this filter . The video will be recorded and will be saved as Filter-Eyes.avi file .

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
1. Pull all the files from this repository by cloning the project . 
``` 
git clone 'https://github.com/AemieJ/filter-python.git'
```
2. Clone the dlib models as shown in the filter-python folder itself . 
```
cd filter-python 
```

```
git clone 'https://github.com/davisking/dlib-models'
```
3. The source code for the filter is written in the filter_eyes.py 

### Importing

The libraries you have to install include : 
1. Opencv-Python
2. argparse 
3. imutils
4. time 
5. dlib 
6. numpy 

```
pip install -r requirements.txt
```

### Installing

A step by step series to get the filter running on your local machine . After the requirements have been installed on your local machine , follow the step on your command line from the filter-python folder .

```
python filter_eyes.py -predictor dlib-models-master/shape_predictor_68_face_landmarks.dat
```

## Built With

* [Open-CV](https://docs.opencv.org/4.1.0/) - Supports deep learning frameworks
* [Dlib](http://dlib.net/python/index.html) - Includes wide range of machine learning algorithm
* [Numpy](https://docs.scipy.org/doc/numpy-1.13.0/reference/) - Support for  multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.


## Authors

* **Aemie Jariwala** - *Initial work* - [AemieJ](https://github.com/AemieJ)

## Acknowledgments
* KP Kaiser
* davisking
