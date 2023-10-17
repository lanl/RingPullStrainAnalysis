# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 15:54:21 2021

@author: 357529
"""


## import the module from a different folder
import sys
sys.path.insert(0, 'E:\\Projects\\RingPull\\RPSA')
from RingPullStrainAnalysis import RingPull,Ring_Image,make_figure

##import other important packages
import numpy as np
pi = np.pi
import matplotlib.pyplot as plt
import matplotlib as mpl


## set all the variables you will need to run this RingPullStrainAnalysis code:
    
## the file with the load frame data and their corresponding images
LF_file = 'E:\\Projects\\RingPull\\RPSA\\RPSA_example\\sample_data\\sample_data_LF.csv'
#The DIC analysis software that was used
DIC_software = 'VIC-2D'
## geometric dimensions of the ring pull test
d_mandrel = 5.1
OD = 10.3
ID = 9.46
W  = 1



## closes all other plots that we have created.
plt.close('all')

## create RingPull object
test = RingPull(LF_file=LF_file,
                software=DIC_software, 
                ID=ID, OD=OD, d_mandrel=d_mandrel, W=W,
                get_geometry_flag=True)


## Analyze the DIC images and pull out usefull parameters
## this is currently commented out because it has a long computation time
# test.analyze_DIC()


## running the above method will allow you to save the data
## you can also save the data without analyzing it (though you may miss out on a few variables)
# test.save_data()


## you can also access this data in script with:
df = test.df


## plot the strain distribution from one of the DIC images
test.plot_strain_distribution(n=245, theta=pi/2, mode='ett', extrap=False)


## analyze the curve as if it were a stress strain curve from a tensile test
## and output important material parameters
E,YS,UTS,eps_u,eps_nu,eps_total,toughness = test.process_stress_strain_curve()


## Some more plotting methods
f,ax = test.plot_stress_strain()
ax.set_xlabel('Strain [mm/mm]')
ax.set_ylabel('Stress [MPa]')


## open one of the DIC_image classes from the RingPull object
img = test.open_Image(354)
print(type(img)==Ring_Image)


## again, this data is pulled in from the csv file. You can see the data here:
df2 = img.df

#%%
## Plots the DIC results overlayed on the image
img.plot_DIC(state='reference',pixel_size=5)
img.plot_DIC(state='deformed',pixel_size=5)

## You can also plot strains in polar coordinates.
img.plot_DIC(state='deformed',mode='ett',pixel_size=5)





