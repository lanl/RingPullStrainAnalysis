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

## use the DIC data to create a digital extensometer on the test sample.
## this will call a user interface to determine the two ends of the
## extensometer.
test.digital_extensometer()

## you can also save the adjusted displacement calculation for future use
test.save_data()

## The complete load-displacement data with calculated stress and strain 
## values can be called by looking at the state variable, df
df = test.df

## plot the strain distribution from one of the DIC images
n = 245
theta = pi/2+1e-4
a = np.linspace(0,1,50)
e = test.open_Image(n).get_value(a,theta,mode='ett', extrap=False)

## RPSA also has a few handy tools to make plotting easier
## such as this make_figure function
f,ax = make_figure()
ax.plot(a,e)
ax.set_xlabel('location on ring (a)')
ax.set_ylabel('strain')


## analyze the curve as if it were a stress strain curve from a tensile test
## and output important material parameters
## can specify the x and y axis to use extensometer or load frame values
E,YS,UTS,eps_u,eps_nu,eps_total,toughness = test.process_stress_strain_curve(x_axis='eng_strain')

## plot the effective stress-strain curve
f,ax = test.plot_stress_strain()
ax.set_xlabel('Strain [mm/mm]')
ax.set_ylabel('Stress [MPa]')


## open one of the DIC_image classes from the RingPull object
img = test.open_Image(354)

## again, this data is pulled in from the csv file. 
## The object saves it as a DataFrame
df2 = img.df

## Plot the DIC results overlayed on the image
img.plot_Image(state='reference')
img.plot_Image(state='deformed')
## Coordinate transformation is automated and will be performed 
## when necessary.
img.plot_Image(state='deformed',mode='ett')



