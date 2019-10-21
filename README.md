# Filter-Early-Match-Late
Source code for IROS 2019 paper Filter Early Match Late

Please reference: S. Hausler, A. Jacobson. and M.J. Milford, "Filter Early Match Late: Improving Network-Based Visual Place Recognition", 2019 International Conference on Intelligent Robots and Systems.

You will need to separately download a CNN model to use with this code. This work used HybridNet (https://github.com/scutzetao/DLfeature_PlaceRecog_icra2017).

To use this code, you will need to modify all the file paths to point to relevant locations on your own system. In particular, change the filepaths in Load_Paths.m and line 30 of Feature_Map_Filter_efle_Nordland.m. The code is compatible with any dataset of images, not just Nordland. 

Also included is a calibrated filter for HybridNet on Nordland, filtering Conv2 under the assumption that the feature vector will be extracted from Conv5. 

If you train your own filter, it will take ~3 minutes/calibration image for a network with 256 feature maps in Conv2. Larger networks like VGG may take longer to calibrate. 

Acknowledgements: 
MATLAB Libaries: MATLAB;
sort_nat: Douglas M. Schwarz copyright 2008; 
Hybrid Net (not included in this release): Zetao Chen 2017.
