PyStrain
===========

PyStrain is a python script that calculates the strain by image correlation by tracking the position of two markers in a series of images. It uses [Openpiv](https://github.com/alexlib/openpiv-python) (Particule Image Velocimetry) to determine markers motion and [PyQtGraph](http://www.pyqtgraph.org/) for the graphical user interface.

## Requirements 

* Python 2.7 with Matplotlib, Numpy (available through [Enthought Canopy](https://store.enthought.com/downloads/) or [Anaconda](http://continuum.io/downloads) distributions for instance).
* [Openpiv](https://github.com/alexlib/openpiv-python) (with progressbar and scikit image)
* [PyQtGraph](http://www.pyqtgraph.org/)
* Download the zip file of PyStrain here and run the script: python /path-to-the-script/pyStrain.py.


# User guide

## Interface

![img1](/im1.png?raw=true)

 
## Markers displacement
You can use the set of pictures in the download image folder as an example. These images are SEM (Scanning electron microscopy) pictures showing a small Be Fiber strained in tension (straining axis along the x direction). The stress values corresponding to the images are stored in the stress.txt file.

Markers displacement vector can be determined as follow:
* A set of (at least 2) images has to be put in the images folder located in the same directory than the script file.
* Select the image of interest by changing the image number and click "update image". In the interface, the image on the right should be updated.
* Drag the ROI (in red) in the first image to the marker area. The bottom left image is a zoom in the ROI. Set the same ROI in the image on the right by pressing the button "Set area". The bottom right image should be updated. The ROI size can be changed using the handle.
* Press the "PIV" button. An arrow should be plotted in the ROI in the right image showing the marker displacement.
 
![img1](/im2.png?raw=true)

* The x and y components (u,v) of the displacement vector should appear in pixel units.

## Treat a serie of pictures
To treat a series of picture:
* Select the number of image and click "update image"
* Choose one marker, set the area, and click on "Serie". The displacement is then computed.
* Choose a second marker, set the are and click on "Serie".
* Finally click on "Strain". A stress-strain plot appears, using the stress.txt file for stress values.

![img1](/im3.png?raw=true)

* The strain values are stored in the result.txt file
* Press "Reset" to erase ROI and previous calculations

## Future improvements
* Thanks to openpiv, the strain can be easily computed directly by PIV without defining markers ROI providing the markers number is high enough. Alternatively, an average strain (kind of true strain) can be computed by defining more than 2 markers.


