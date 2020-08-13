## Demo
The demo portion of this repository can be executed under conduct.py. The modules presented provide the capability of conducting object detection in satellite imagery using user produced training data.
#
### create_training_data.py
The images found in positive_chips.zip are produced from a manaul process in QGIS of locating ships in the desired image AOI, creating a square buffer around centroid points, and clipping the satellite image to each buffer region. This in turn creates a 200m x 200m image chip of a ship. The images in negative_chips.zip are produced following similar steps. However, instead of labeling known ships, random points are generated across the AOI to create negative training samples. The CreateTrainingChips module ingests these images, crops them to the desired dimension, optionally augments the images, and finally transforms to a collection of tensors and labels to be used in training a Deep Learning Model.
#
### train_model.py
Next, the produced training data is split into train/testing datasets.A Keras sequential model is then defined, compiled, and fit to the training data. The produced model can then be optionally saved or integrated directly into the next portion where predictions are made.
#
### get_sentinel_imagery.py
An option is presented to download imagery as part of the programming process from European Space Agency open source imagery API. Required parameters include a geographic file that will be used to regional queries, a data range for searching, as well as desired maximum cloud coverage.
#
### object_detection.py
The purpose of this module is to incorporate training data and trained model to conduct object detection in an image. A built in function conducts predictions across the image using a "moving window" where "windows" that have a prediction > user-defined predction_threshold are converted into coordinates and stored 
in a list. Next, the positive prediction coordinates are filtered to produce only one prediction per potential object. This is done by analyzing overlapping bounding boxes and keeping the highest scoring prediction. The final results are then referenced back to the geographic attribution of the image and converted 
into well known text(WKT) strings that can be imported as geographic data into GIS software.

#
### conduct.py
This is the main execution module

