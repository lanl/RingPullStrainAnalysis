# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 15:54:21 2021

@author: 357529
"""


## import the necessary modules from the saved folder
import sys
sys.path.insert(0, 'D:\\Projects\\RingPull\\RPSA')
from RingPullStrainAnalysis import RingPull,make_figure
from RingPullCoatingAnalysis import RingPullCoatingAnalysis

##import other important packages
import numpy as np
pi = np.pi
import matplotlib.pyplot as plt
import matplotlib as mpl


## set all the variables you will need to run this RingPullStrainAnalysis code:
    
## the file with the load frame data and their corresponding images
LF_file = 'D:\\Projects\\RingPull\\RPSA\\RPSA_example\\sample_data\\sample_data_LF.csv'
## The DIC analysis software that was used.
DIC_software = 'VIC-2D'
## geometric dimensions of the ring pull test
d_mandrel = 5.1
OD = 10.3
ID = 9.46
W  = 1

## close all other plots that we have created.
plt.close('all')

## create instance of RingPull object
## setting get_geometry_flag to True calls a user interface to determine 
## the scale and centroid of the ring
test = RingPull(LF_file=LF_file,
                software=DIC_software, 
                ID=ID, OD=OD, d_mandrel=d_mandrel, W=W,
                get_geometry_flag=True)


## Create coating analysis instance and run it on a few different frames
method = RingPullCoatingAnalysis(test,mode='compression')
n=[60,65,70]
analysis_data = method.get_side_image_strain(n,debug=False)






