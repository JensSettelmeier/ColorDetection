# ColorDetection

The provided function ReflectorSegmentation finds the coordinates of an object that is surrounded by red color as long as there ar not to much other areas red colored in the picture and the light is not too bright. It is based on a thresholding in hue Channel after converting the RGB image to HSV and then applying a K-Means on the pixel values. 

# Requirments
opencv
numpy
matplotlib
sklearn

# Usage
clone the repository and execute the test_script_ReflectorSegmentation.py
