'''
Ring Pull Strain Analysis (RPSA)

RPSA is intended as an python module to investigate data either from digital 
image correlation software such as DICengine or VIC 2-D or from physical 
simulation software such as MOOSE. Its primary capability involves taking an 
output 2-dimensional mesh of strain data specific to a gaugeless Ring Pull test 
and analyzing it for parameters of interest. RPSA also allows input load frame 
data that is synchronized with the DIC images.

v1.2

Created by:
    Peter Beck
    pmbeck@lanl.gov
Updated:
    26-Jun-2024


General notes on naming conventions and coding practices:
    - All class objects are in camel case 
    - All functions: words are separated with underscores
    - Functions starting with an underscore are mainly meant for internal use
    - variables are typically one word, but if multiple words, they are typically separated with underscore
    - The packages numpy (np) and pandas (pd) are frequently used and referred to by their abbreviations
    - most functions can either take a float, a list or a numpy array. 
    - A matplotlib axes method can be passed to any plotting method and it 
        will plot on that axes.
    - Some plotting methods allow for kwargs to be passed to the underlying
        plotting methods.
    - Some methods have a user interface.They interface was made to work on the 
        following settings, however, it is likely to work with other system 
        configurations. The settings are: 
            - Windows 10
            - Python 3.8.8 (installed via Anaconda)
            - Spyder 4, with the setting:
                - IPython console -> Graphics -> Backend: Automatic

To Do:
    - calculate mean, stdev, histogram, from user defined polygon of points. 
        This will help error calculations.

'''


# Generic imports
import os
import time
import imageio
import copy
import seaborn
seaborn.set_palette('Set1') #set color scheme


#numpy imports
import numpy as np
import numpy.ma as ma
pi = np.pi

#pandas imports
import pandas as pd
pd.options.mode.chained_assignment = None  #disable SettingWithCopyWarning in pandas
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning) #disable RuntimeWarning that kicks up with pandas

#matplotlib imports
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backend_bases import MouseButton
import matplotlib.path as mplPath
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
# import matplotlib.gridspec as gridspec
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['legend.frameon'] = False
mpl.rcParams.update({'font.size': 8})

#scipy imports
from scipy import interpolate,integrate
from scipy.interpolate import interp1d
from scipy.stats import linregress

#tkinter imports
import tkinter as tk
from tkinter import simpledialog,filedialog

from matplotlib.gridspec import GridSpec


def make_figure(ax=None):
    '''
    Description
    -----------
    A function that formats a matplotlib axes for plotting. If no axes is
    specified, will create a figure and axes object
    
    Parameters
    ----------
    ax : matplotlib.axes, optional
        The matplotlib axes object to modify. The default is None.

    Returns
    -------
    f : matplotlib.figure
        The created figure for the plot
    ax : matplotlib.axes
        The created axes for the plot
    '''
    if ax==None:
        f = plt.figure()
        ax = plt.gca()
        # f.set_size_inches(1920/f.dpi,1200/f.dpi)
        f.set_size_inches([3.25,2.9])
        f.subplots_adjust(top=0.92,bottom=0.12,left=0.17,right=0.96,hspace=0.2,wspace=0.2)        
    else:
        f = ax.get_figure()
    # axes housekeeping
    # ax.minorticks_on()
    ax.tick_params(axis='x', which='major', direction='in', top=True, bottom=True, length=4)
    ax.tick_params(axis='x', which='minor', direction='in', top=True, bottom=True, length=2)
    ax.tick_params(axis='y', which='major', direction='in', left=True, right=True, length=4)
    ax.tick_params(axis='y', which='minor', direction='in', left=True, right=True, length=2)
    ax.tick_params(direction='in')

    return f,ax

def make_img_figure(ax=None):
    '''
    Description
    -----------
    A function that formats a matplotlib axes for plotting images. If no axes is 
    specified, will create a figure and axes object
    
    Parameters
    ----------
    ax : matplotlib.axes, optional
        The matplotlib axes object to modify. The default is None.

    Returns
    -------
    f : matplotlib.figure
        The created figure for the plot
    ax : matplotlib.axes
        The created axes for the plot
    '''
    if ax==None:
        f = plt.figure()
        ax = plt.axes()
        f.set_size_inches(6.5,4)
        f.subplots_adjust(top=0.875,bottom=0.085,left=0.05,right=0.95,hspace=0.2,wspace=0.2)
    else:
        f = ax.get_figure()

    ax.axis('equal')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    return f,ax

def make_3D_figure(ax = None):
    '''
    Description
    -----------
    A function that formats a matplotlib axes for 3D plotting. If no axes is 
    specified, will create a figure and axes object
    
    Parameters
    ----------
    ax : matplotlib.axes, optional
        The matplotlib axes object to modify. The default is None.

    Returns
    -------
    f : matplotlib.figure
        The created figure for the plot
    ax : matplotlib.axes
        The created axes for the plot
    '''
    if ax==None:
        f = plt.figure()
        ax = f.add_subplot(projection='3d')
    else:
        f = ax.get_figure()
    return f,ax
    
def makeRGBA(r_set,g_set,b_set):
    '''
    Description
    -----------
    A function that creates a customcolormap from the lists of red-green-blue 
    values. All inputs must be lists or tuples of the same length. This 
    function creates evenly spaced gradients between the values listed in the 
    arrays.
    
    Parameters
    ----------
    r_set : list,tuple of floats
        list of values from 0-1 for how much red at each point.
    g_set : list,tuple of floats
        list of values from 0-1 for how much green at each point.
    b_set : list,tuple of floats
        list of values from 0-1 for how much blue at each point.

    Returns
    -------
    custom_cmap : matplotlib.colors.ListedColormap
        The custom colormap created with the input rgb values.
    '''
    output=[]
    for cset in [r_set,g_set,b_set]:
        clist = []
        clist.append(np.linspace(cset[0],cset[1],128))
        for i in range(1,len(cset)-2):
            clist.append(np.linspace(cset[i],cset[i+1],256))
        clist.append(np.linspace(cset[-2],cset[-1],128))
        output.append(np.concatenate(clist))
    output.append(np.ones(output[0].shape))
    return mpl.colors.ListedColormap(np.vstack(output).transpose())

r_set = (.2, 0, .1, .2,  1, 1, .5)
g_set = (.4, 1, .7, .2, .8, 0,  0)
b_set = (.2, 0,.75,  1, .4, 0,  0)
custom_cmap = makeRGBA(r_set,g_set,b_set)
del r_set, g_set, b_set

def make_GIF(images_list, output_filename, end_pause=True,**writer_kwargs):
    '''
    Description
    -----------
    A function that creates a .gif file of all the input images
    
    Parameters
    ----------
    images_list : list of str
        list of strings which contain the image filenames to put into the gif
    output_filename : str
        The filename to save the gif to.
    end_pause : bool, int, optional
        An integer of how many images to stack on the back of the gif to give 
        the appearance of pausing the video. If False, no images are stacked.
        If True, 80 images are stacked. The default is True
    writer_kwargs : dict, optional
        Keyword arguments to pass to the writer. default is sub_rectangles = True.

    Returns
    -------
    None
    
    '''
    default_kwargs = {}#{'subrectangles':True}
    writer_kwargs = { **default_kwargs, **writer_kwargs }
    if end_pause:
        if type(end_pause) == bool:
            end_pause = 80
        for n in range(end_pause):  # add some extra images at the end to 'pause' the GIF
            images_list.append(images_list[-1])
    with imageio.get_writer(output_filename, mode='I',**writer_kwargs) as writer:
        for f in images_list:
            # image = imageio.imread(f,pilmode='RGB')
            image = imageio.imread(f)
            writer.append_data(image)
    print('GIF has been created at the following location:')
    print(output_filename)
    return output_filename
    
def define_circle(p1, p2, p3):
    '''
    Description
    -----------
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    
    Parameters
    ----------
    p1 : list, tuple of 2 floats 
        the first point (x,y) used to define the circle.
    p2 : list, tuple of 2 floats 
        the second point (x,y) used to define the circle.
    p3 : list, tuple of 2 floats 
        the third point (x,y) used to define the circle.

    Returns
    -------
    centroid : tuple of 2 floats
        the (x,y) coordinates of the center of the circle
    radius : float
        the radius of the circle    
    '''
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def UI_get_pts(prompt,n=None,f = None,ax = None):
    '''
    Description
    -----------
    User interface to get points from the current matplotlib figure. Returns 
    a list of points that the user selected
    
    Parameters
    ----------
    prompt : str
        The prompt to the user describing the problem and which points to select.
        This gets put in the title of the plot. A description of how to select 
        the points is appended to the prompt.
    n : int, optional
        How many points the function should expect. If the user does not click 
        the correct number of points, they get asked to do it again. If this 
        value is None, then there is no set amount of points needed. The default
        is None.

    Returns
    -------
    pts : list of tuples of length 2
        List of (x,y) points that the user selected
    '''
    
    if not f:
        f = plt.gcf()
    if not ax:
        ax = plt.gca()
    
    f.subplots_adjust(top=0.80,bottom=0.08,left=0.17,right=0.95,hspace=0.2,wspace=0.2)
    prompt += '\nRight click to finish.\nMiddle mouse to remove points'
    ax.set_title(prompt)
    plt.draw() 
    while True:
        pts = plt.ginput(1000, timeout=-1,mouse_pop=MouseButton.MIDDLE,mouse_stop=MouseButton.RIGHT)
        if type(n) == type(None):
            break
        elif len(pts)==n:
            break
        elif n=='even':
            if len(pts)%2==0:
                break
        #if the above if statement doesn't break the while loop
        plt.title('Error, retrying.\n'+prompt)
        plt.draw()
    return pts

def UI_circle(ax,prompt,facecolor,num = 50):
    '''
    Description
    -----------
    User interface to get 3 points and make a circle out of them. Then plots
    this circle on the given graph. This function calls UI_get_pts to find 
    the points.
    
    Parameters
    ----------
    ax : matplotlib.axes
        The axes to select and plot the circle on
        
    prompt : str
        The prompt to the user describing the problem and which points to select.
        This gets put in the title of the plot. A description of how to select 
        the points is appended to the prompt.
    facecolor : matplotlib color input
        The color to draw the circle out of. 
    num : int, optional
        number of points to return to create the circle. The default is 50.
    
    Returns
    -------
    pts :  num x 2 array of floats
        array of (x,y) points that create a circle. the number of points is
        based off the input parameter, num +1. The first and last
        values of the array are the same in order to complete the circle.
    '''
    while True:
        pts = UI_get_pts(prompt,3)
        c,r = define_circle(*pts)
        pts = [
            (r*np.cos(theta)+c[0],
             r*np.sin(theta)+c[1])
            for theta in np.linspace(0,2*pi,num)]
        my_circle = Polygon(pts, closed = True, facecolor=facecolor,alpha = 0.2)
        ax.add_patch(my_circle)
        plt.title('Is this acceptable? Click to continue or hit enter to retry.')
        plt.draw()
        if not plt.waitforbuttonpress():#if mouse click
            break
        else: #if keyboard button
            my_circle.remove()
            plt.draw()
            #ask for new circle
    return np.array(pts)

def define_ellipse(pts):
    '''
    https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
    '''
    
    pts=np.array(pts)
    x = pts[:,0]
    y = pts[:,1]

    A = np.vstack(([x**2, x*y, y**2, x, y])).transpose()
    b = np.ones_like(x)

    ans = np.linalg.lstsq(A, b,rcond=None)[0].squeeze()
    ans = np.append(ans,-1)

    return ans

def UI_ellipse(ax,prompt):
    '''
    Description
    -----------
    User interface to get 5 points and make an ellipse out of them. Then plots
    this ellipse on the given graph. This function calls UI_get_pts to find 
    the points.

    Parameters
    ----------
    ax : matplotlib.axes
        The axes to select and plot the circle on
        
    prompt : str
        The prompt to the user describing the problem and which points to select.
        This gets put in the title of the plot. A description of how to select 
        the points is appended to the prompt.
    facecolor : matplotlib color input
        The color to draw the ellipse out of. 
    num : int, optional
        number of points to return to create the circle. The default is 50.
    
    Returns
    -------
    pts :  num x 2 array of floats
        array of (x,y) points that create a circle. the number of points is
        based off the input parameter, num +1. The first and last
        values of the array are the same in order to complete the circle.
    '''
    while True:
        pts = UI_get_pts(prompt)

        ans = define_ellipse(pts)
        
        X = np.linspace(*ax.get_xlim(),num=1000)
        Y = np.linspace(*ax.get_ylim(),num=1000)
        X,Y = np.meshgrid(X,Y)
        
        Z = ans[0]*X**2 + ans[1] *X*Y + ans[2]*Y**2 + ans[3]*X + ans[4]*Y + ans[5]
        cs = ax.contour(X,Y,Z,levels=[0],colors='b',linewidths=1)
        
        ellipse_path = cs.collections[0].get_paths()[0]
        ellipse_pts = ellipse_path.vertices
        
        plt.title('Is this acceptable? Click to continue or hit enter to retry.')
        plt.draw()
        my_bool = plt.waitforbuttonpress()
        if not my_bool:#if mouse click
            break
        else: #if keyboard button
            cs.collections[0].remove()
            plt.draw()
            #ask for new circle
    return ellipse_pts 

def UI_polygon(ax,prompt,facecolor):
    '''
    Description
    -----------
    User interface to get points and make a polygon out of them. Then plots
    this polygon on the given graph. This function calls UI_get_pts to find 
    the points.
    
    Parameters
    ----------
    ax : matplotlib.axes
        The axes to select and plot the circle on
        
    prompt : str
        The prompt to the user describing the problem and which points to select.
        This gets put in the title of the plot. A description of how to select 
        the points is appended to the prompt.
    facecolor : matplotlib color input
        The color to draw the polygon out of. 

    Returns
    -------
    pts :  n x 2 array of floats
        array of (x,y) points that create a polygon. The number of points is
        based off how many points the user selects plus 1. The first and last
        values of the array are the same in order to complete the polygon.
    '''
    while True:
        pts = UI_get_pts(prompt,None)
        pts.append(pts[0])
        pts = np.array(pts)
        my_polygon = Polygon(pts, closed = True, facecolor=facecolor,alpha = 0.2)
        ax.add_patch(my_polygon)
        
        plt.title('Is this acceptable? Click to continue or hit enter to retry.')
        plt.draw()
        if not plt.waitforbuttonpress():#if mouse click
            break
        else: #if keyboard button
            my_polygon.remove()
            plt.draw()
            #ask for new circle
    return np.array(pts)

def UI_line(ax,prompt,color='r'):
    '''
    Description
    -----------
    User interface to get points and make a line out of them. Then plots
    this line on the given graph. This function calls UI_get_pts to find 
    the points.
    
    Parameters
    ----------
    ax : matplotlib.axes
        The axes to select and plot the circle on
        
    prompt : str
        The prompt to the user describing the problem and which points to select.
        This gets put in the title of the plot. A description of how to select 
        the points is appended to the prompt.
    color : matplotlib color input
        The color to draw the line. 

    Returns
    -------
    pts :  2 x 2 array of floats
        array of (x,y) points that create the line.
    '''
    while True:
        pts = UI_get_pts(prompt,2)
        pts = np.array(pts)

        my_line = Line2D(pts[:,0],pts[:,1])
        ax.add_line(my_line)
        
        plt.title('Is this acceptable? Click to continue or hit enter to retry.')
        plt.draw()
        if not plt.waitforbuttonpress():#if mouse click
            break
        else: #if keyboard button
            my_line.remove()
            plt.draw()
            #ask for new circle
    return np.array(pts)

def find_nearest_idx(x_array,y_array,x_q,y_q):
    '''
    Description
    -----------
    Given arrays in x and y, finds the index of the array point that is 
    nearest to the test point, (x_q,y_q).
    
    Parameters
    ----------
    x_array : array_like
        numpy array of x values.The size of this array should be the same as
        the size of y_array
    y_array : array_like
        numpy array of y values.The size of this array should be the same as
        the size of x_array
    x_q : float
        x coordinate of the test point
    y_q : float
        y coordinate of the test point
        
    Returns
    -------
    idx :  int
        index in x_array and y_array that is closest to the test point
    '''
    return np.nanargmin(abs(x_array-x_q)/abs(x_q) + abs(y_array-y_q)/abs(y_q))

def log_transform(e_row,max_strain,lim=1e2):
    '''
    Description
    -----------
    A pseudo-logarithmic transform thet maps the linear region from 
    (0,max_strain) to the logarithmic domain from (0,max_strain). It compresses 
    the curve of y = ln(x) on the bounds (1,lim) to make this transform. 
    This function returns the transformed value of e_row in the new space and 
    it is symmetric about the origin, such that it can also handle negative 
    values. It is useful for transforming an axis of a plot to logarithmic 
    instead of linear.
    
    Parameters
    ----------
    e_row : float
        the value to transform.
    max strain : float
        the maximum linear value to map. Any value of e_row larger in magnitude
        than this will be replaced by this.
    lim : float, bool, optional
        parameter to judge how steep the transform will be. A higher value 
        will create a steeper tranform. The default is 100.0.This can be a 
        boolean. If false, will not perform the transform

    Returns
    -------
    img_value : float
        the transformed value.
    '''
    img_value = e_row
    if lim:#if lim is True or a number, perform transform
        if type(lim) == bool:
            lim = 1e2
        
        #make all values positive
        img_value = abs(img_value)
    
        img_value = 1+img_value/max_strain*(lim-1)
        img_value = np.log(img_value)/np.log(lim)
        #converts the absolute value back into positive/negative
        img_value=img_value*np.sign(e_row)*max_strain
    else: #if lim is False,do not perform transform
        pass
    return img_value

################################################################## 
class TensileImage:
    '''
    Description
    -----------
    Base class for handling and calculating strains on a dogbone sample during tensile testing.
    '''
    
    def __init__(self, test_img, init_img, software='VIC-2D',centroid = np.array((0,0)),scale=1):
        try:
            csv_file = test_img.split('.tif')[0]+'.csv'
            self.df = pd.read_csv(csv_file) 
        except:
            csv_file = test_img.split('.tif')[0]+'.txt'
            self.df = pd.read_csv(csv_file)             
        
        # rename column names to standard. VIC 2D puts excess spaces, which first need to be stripped.
        column_rename_dict2 = {key:key.strip() for key in self.df.columns }
        self.df.rename(column_rename_dict2, axis=1, inplace=True)

        # delete all unnecessary data
        try:
            self.df = self.df[self.df['"sigma"'] != -1]
        except:
            pass

        if software == 'MOOSE':#need to add vonmises strain, e1,e2, and stresses
            column_rename_dict = {'x':'x','y':'y','"Xp"':'x_def','"Yp"':'y_def',
                                   'disp_x':'u','disp_y':'v',
                                   'strain_xx':'exx','strain_yy':'eyy','strain_zz':'ezz',
                                   'strain_yz':'eyz','strain_xz':'exz','strain_xy':'exy',
                                   'firstinv_strain':'e_1invar'}
        elif software=='VIC-2D':
            column_rename_dict = {'"x"':'x','"y"':'y','"Xp"':'x_def','"Yp"':'y_def',
                                  '"u"':'u','"v"':'v',
                                  '"exx"':'exx','"eyy"':'eyy','"exy"':'exy',
                                  '"e1"':'e1','"e2"':'e2',
                                  '"e_vonmises"':'e_vm','"sigma"':'sigma'}
        elif software=='DICe':
            column_rename_dict = {'COORDINATE_X':'x','COORDINATE_Y':'y',
                                  'DISPLACEMENT_X':'u','DISPLACEMENT_Y':'v',
                                  'SIGMA':'sigma','VSG_STRAIN_XX':'exx',
                                  'VSG_STRAIN_YY':'eyy','VSG_STRAIN_XY':'exy'}
        else:
            assert False,"Could not recognize the software format. Please choose a software of 'MOOSE','VIC-2D', or 'DICe'"
        
        
        self.df.rename(column_rename_dict, axis=1, inplace=True)

        try:
            if not 'e_1invar' in self.df.columns:        
                self.df['e_1invar'] = self.df['e1']+self.df['e2']
        except:
            pass
            
        self.df['x_def'] = self.df['x'] + self.df['u']
        self.df['y_def'] = self.df['y'] + self.df['v']
        
        self.centroid = centroid

        self.scale = scale  # pixels/mm        
        self.df[['x', 'y', 'u', 'v', 'x_def', 'y_def']] = self.df[['x', 'y', 'u', 'v', 'x_def', 'y_def']].apply(
            lambda x: x/self.scale)
        
        self.centroid_pixel = centroid
        self.centroid = centroid/self.scale
            
        self.test_img_file = test_img
        self.init_img_file = init_img            
        
        # only save the columns that we care about
        cols = np.array(self.df.columns)
        self.df = self.df[cols[['"' not in s for s in cols]]]

    def get_value(self,x,y,mode='e_vm',extrap=False):
        x = x + self.centroid[0]
        y = y + self.centroid[1]
        
        z = interpolate.griddata((self.df['x'], self.df['y']), self.df[mode].ravel(),
                                 (x, y), method='cubic')
        
        if extrap and np.isnan(z).any():
            assert False, 'Sorry, extrapolation is not available at this time'   
        return z

    def _get_extrap_value(self,a,theta,mode='e_vm'):
        '''defnuct function right now'''
        # maybe use this for 2-D extrapolation? https://github.com/pig2015/mathpy/blob/master/polation/globalspline.py
        # https://stackoverflow.com/questions/34053174/python-scipy-for-2d-extrapolated-spline-function
        
        # general info on many python interpolation functions:
        # https://stackoverflow.com/questions/37872171/how-can-i-perform-two-dimensional-interpolation-using-scipy
        
        a_i,z = self.get_strain_distribution(theta,mode,extrap=False,N=50)
        z = np.reshape(z, a_i.shape)
        a_i = a_i[np.invert(np.isnan(z))]
        z=z[np.invert(np.isnan(z))]

        
        if (np.size(a_i) <= 1):#if the entire array is NaN or if there is not enough values to extrapolate off of
            z = np.nan
        else:
            f = interpolate.interp1d(a_i,z,fill_value='extrapolate')
            z = f(a)
        return z

    def get_value_deformed(self,x_def,y_def,mode='e_vm',extrap=False):
        x_def = x_def + self.centroid[0]
        y_def = y_def + self.centroid[1]
        
        z = interpolate.griddata((self.df['x_def'], self.df['y_def']), self.df[mode].ravel(),
                                 (x_def, y_def), method='cubic')
        if extrap:
            assert False, 'Sorry, extrapolation is not available at this time'
        return z

    def digital_extensometer(self, x1,y1,x2,y2,ext_mode='norm'):
        x_def1 = self.get_value(x1 ,y1, mode='x_def')
        y_def1 = self.get_value(x1, y1, mode='y_def')
        x_def2 = self.get_value(x2, y2, mode='x_def')
        y_def2 = self.get_value(x2, y2, mode='y_def')
        
        if ext_mode =='norm':
            pass
        elif ext_mode=='x':
            y_def1 = 0
            y_def2 = 0
        elif ext_mode=='y':
            x_def1 = 0
            x_def2 = 0
        
        vect=np.array((x_def1-x_def2,y_def1-y_def2))
        disp = np.linalg.norm(vect,axis=0)
        
        return disp
    
    def plot_Image(self, state='deformed', mode='e_vm', log_transform_flag=1e2,
                   max_strain=None, ax=None, **plot_kwargs):
        '''
        Description
        -----------
        A heat map of the strain around the ring. Specific to DIC data.       

        Parameters
        ----------
        state : str, optional
            Whether you want the ring plotted in the 'reference' or 'deformed' 
            state. The default is 'deformed'.
        mode : str, optional
            The type of strain to plot. The default is 'e_vm'.
        log_transform_flag : bool, float, optional
            Flag for if the data should be plotted on a nonlinear scale. This 
            nonlinear transform also uses this variable as a parameter for how 
            nonlinear to make it. The default is 1e2.
        max_strain : float, optional
            Sets the plot limits for strain to plot. If False or None, the
            function autoscales. The default is None.
        ax : matplotlib.axes, optional
            The matplotlib axes object to plot on. The default is None.
        plot_kwargs : dict, optional
            Dictionary of key word arguments to pass to the matplotlib scatter 
            function. The default is {'alpha': 1,'cmap':'custom','s':0.25}.

        Returns
        -------
        f : matplotlib.figure
            The created figure for the plot
        ax : matplotlib.axes
            The created axes for the plot

        '''

        default_kwargs = {'alpha':1,'cmap':'custom','s':0.5}
        plot_kwargs = { **default_kwargs, **plot_kwargs }
        

        if not max_strain: ##if max_strain==None
            max_strain = np.nanmax(abs(self.df[mode]))
            
        if state =='reference':
            x = self.df['x']
            y = self.df['y']
            
        elif state =='deformed':
            x = self.df['x_def']
            y = self.df['y_def']
            
        else:
            assert False, "The state must be either 'reference' or 'deformed'"
        
        x = x*self.scale
        y = y*self.scale
        
        z = self.df[mode]
        z = log_transform(z, max_strain, log_transform_flag)
       
        
        # if user does not specify color map or if they say cmap = 'custom', 
        # then set custom colormap
        # if they set cmap to 'none' or None,
        # then take away DIC data, by making alpha = 0
        try: 
            if plot_kwargs['cmap']=='custom':
                assert False
            elif plot_kwargs['cmap']=='none':
                plot_kwargs['alpha'] = 0
                del plot_kwargs['cmap']
            elif plot_kwargs['cmap']==None:
                plot_kwargs['alpha'] = 0
                del plot_kwargs['cmap']
        except (KeyError,AssertionError):#if the cmap kwarg either doesn't exist or is 'custom'
            if type(plot_kwargs)==type(None):
                plot_kwargs={}
            plot_kwargs['cmap'] = custom_cmap  
            
            
        f,ax = make_img_figure(ax)
        
        try:#try to plot a real image behind the DIC data
            if state =='reference':
                plot_img = mpimg.imread(self.init_img_file)
            elif state =='deformed':
                plot_img = mpimg.imread(self.test_img_file)
            ax.imshow(plot_img, cmap='gray') 
        except:#no image associated with it
            pass

        cmapable = ax.scatter(x,y,c=z,vmin=-max_strain,vmax=max_strain,**plot_kwargs)

        #plot cmap unless there is no color (alpha = 0)
        if plot_kwargs['alpha']==0:
            pass
        else:
            cbar=plt.colorbar(cmapable,ax=ax,
                              ticks=[-max_strain,
                                     log_transform(-max_strain*3/4, max_strain, log_transform_flag),
                                     log_transform(-max_strain/2, max_strain, log_transform_flag),
                                     log_transform(-max_strain/4, max_strain, log_transform_flag),
                                     0,
                                     log_transform(max_strain/4, max_strain, log_transform_flag),
                                     log_transform(max_strain/2, max_strain, log_transform_flag),
                                     log_transform(max_strain*3/4, max_strain, log_transform_flag),
                                     max_strain])
            cbar.ax.set_yticklabels([f'{-max_strain:.3f}','', f'{-max_strain/2:.3f}','',f'{0:.4f}','',f'{max_strain/2:.3f}','',f'{max_strain:.3f}'],
                                     rotation=45)
            cbar.set_label('strain ('+mode+')', rotation=270, labelpad=10)
            for label in cbar.ax.get_yticklabels():
                label.set_verticalalignment('baseline')
        return f,ax  

class RingImage(TensileImage):
    '''
    Description
    -----------
    A class to analyze the output image and data file from the DIC software. 
    This class inherets from TensileImage and modifies methods pertaining to 
    experimental (DIC) data.
    '''

    def __init__(self, test_img, init_img,software='VIC-2D 7',ID=9.46, OD=10.3, d_mandrel=3, OD_path=None, ID_path=None, scale = None, centroid=None):
        '''
        Description
        -----------
        Initializes DIC_Image class.
        
        Parameters
        ----------
        test_img : str
            Filepath for the image that we are analyzing.
        init_img : str
            Filepath for the initial image.
        software : str, optional
            The software used to run the image correlation. The default is
            'VIC-2D 7'.
        ID : float, optional
            The inner diameter of the sample in mm. The default is 9.46.
        OD : float, optional
            The outer dimater of the sample in mm. The default is 10.3.
        d_mandrel : float, optional
            the diameter of the mandrel used when loading the sample in mm. 
            The default is 3.
        OD_path : mplPath.Path object, optional
            Path of the outer diameter which will be used for indexing points 
            inside the area of interest(AOI). This is primarily going to be passed 
            from the RingPull class. If None, will not exclude points outside 
            the AOI. The default is None.
        ID_path : mplPath.Path object, optional
            Path of the inner diameter which will be used for indexing points 
            inside the area of interest(AOI). This is primarily going to be passed 
            from the RingPull class. If the variable OD_path is None, will not 
            exclude points outside the AOI. The default is None.
        scale : float, optional
            The scale of the image in pixels/mm. If None, sets the scale to 1.
            The default is None.
        centroid : np.array, optional
            Length 2 array of the (x,y) coordinates of the centroid of the 
            ring. This is used to convert to polar coordinates. The default 
            is None.

        Returns
        -------
        None.
        '''
        
        super().__init__(test_img, init_img, software, centroid, scale)

        # if specified, delete all the entries out of the bounds of the ring
        if type(OD_path) != type(None):
            idx_pts = [(self.df['x'][n]*self.scale,self.df['y'][n]*self.scale) for n in self.df.index] 
            
            if type(ID_path) != type(None):
                inside = OD_path.contains_points(idx_pts)*np.invert(ID_path.contains_points(idx_pts))
            else:
                inside = OD_path.contains_points(idx_pts)
            self.df = self.df[inside] 
            

        if type(centroid) == type(None):
            # have not determined the center, then use ring geometry to find it
            # find the centroid as the center between the 
            # largest and smallest x positions where there is data. Same with y.
            centroid = np.array([min(self.df.x)+max(self.df.x), min(self.df.y)+max(self.df.y)])/2
            self.centroid = centroid

        # only save the columns that we care about
        cols = np.array(self.df.columns)
        self.df = self.df[cols[['"' not in s for s in cols]]]
        
        # set geometric dimensions as class attributes
        self.ID = ID
        self.OD = OD
        self.d_avg = (ID+OD)/2
        self.d_mandrel = d_mandrel
        self.thickness = (OD-ID)/2

        
    def find_displacement(self):
        '''
        Description
        -----------
        Calculates the displacement of the mandrels by using the inner 
        surface of the ring where it contacts the mandrels. This is used to 
        bypass complicance of the load frame.

        Returns
        -------
        disp : float
            The calculated displacement.
        '''
        # look at the inner surface of the ring on either side
        a = 0.00
        theta = [0, pi]
        # find the deformation vector, u
        z = self.get_value(a, theta, mode='u',extrap = True)
        # subtract the two deformation vectors to get displacement
        disp = float(z[0]-z[1])    
        return disp  

    def _get_extrap_value(self,a,theta,mode='e_vm',extrap='extrap',smoothing=False):
        def extrapolation(a_i,theta_i,mode,extrap):
            def lin_regress(a_i,z,a,cutoff=0.5):
                z = z[a_i > cutoff]
                a_i = a_i[a_i > cutoff]
                m,b,_,_,_ = linregress(a_i,z)
                reg_point = a * m + b
                return reg_point       
            
            N = 50
            a = np.linspace(0, 1, N)
            
            z = self.get_value(a,theta_i,mode,extrap=False)
            # z = z.flatten()

            a = a[np.invert(np.isnan(z))]
            z = z[np.invert(np.isnan(z))]
            # assert False
            if (np.size(a) <= 1):#if the entire array is NaN or if there is not enough values to extrapolate off of
                z = np.nan
            else:
                if extrap == 'extrap' or extrap==True:
                    f = interpolate.interp1d(a,z,fill_value='extrapolate')
                    z = f(a_i)   
                elif extrap=='lin_regress':
                    z = lin_regress(a,z,a_i)
                elif extrap == 'nearest':
                    f = interpolate.interp1d(a,z,bounds_error=False,fill_value=z[-1])
                    z = f(a_i)
            return z
        
        z = np.zeros(a.shape)
        for i in range(len(a)):
            a_i = a[i]
            theta_i = theta[i]
            if smoothing:
                M = 11
                frac = 1/100
                theta_array = np.linspace(theta_i-frac*pi,theta_i+frac*pi,M)
                z_array = np.array([extrapolation(a_i,angle,mode,extrap) for angle in theta_array])
                
                def gaussian(x,sigma):
                    return np.exp(-x**2/(2*sigma**2))
                weights = gaussian(np.linspace(-5,5,M),1.75)
                weights = weights / sum(weights)
                weighted_mean = sum(z_array*weights)
                z[i] = (weighted_mean)
            else:
                z[i] = extrapolation(a_i,theta_i,mode,extrap)

        return z
    
    def get_value(self,a,theta,mode='e_vm',extrap=False,smoothing=False):
        '''
        Description
        -----------
        Returns the requested values of strain or displacement at a specific 
        point defined by a and theta. This value will be interpolated between
        the surrounding grid points. Optional extrapolation as described in the 
        _get_extrap_value function.

        Parameters
        ----------
        a : float, np.array
            The location, a within the ring of the requested values. 
            a = 0 corresponds with the inner surface of the ring and
            a = 1 corresponds with the outer surface of the ring.
        theta : float, np.array
            The angular location of the requested values.
        mode : str, optional
            The reuqested strain/diplacement value to pull from the point 
            cloud. The default is 'e_vm'.
        extrap : bool, optional
            Boolean flag for if extrapolation is requested. The default is False.

        Returns
        -------
        z : float, np.array
            The return value(s) of strain/displacement.

        '''
        # create a meshgrid of a and theta values that encompasses the input a and theta values
        if mode not in self.df.columns:#mode is in df
            self.analyze_radial_strains()
        

        a,theta = np.meshgrid(a,theta)
        
        r = self.get_r(a)
        # choose strain value to use
        z = self.df[mode]
        # convert to xy coordinates
        x = r*np.cos(theta)+self.centroid[0]
        y = -r*np.sin(theta)+self.centroid[1]
        # mesh interpolation of values
        
        z = interpolate.griddata((self.df['x'], self.df['y']), z.ravel(),
                                 (x, y), method='cubic')


        if extrap and np.isnan(z).any():
            idx = np.isnan(z)

            # print(a)
            # print(a[idx])

            z[idx] = self._get_extrap_value(a[idx],theta[idx],mode,extrap,smoothing)
            
        if 1 in z.shape:
            z = z.flatten()
            
        return z



    def polar_conversion(self,x,y):
        # make x-y coords start at the centroid of the figure
        x = x - self.centroid[0]
        y = y - self.centroid[1] 
        
        r = np.sqrt(x**2+y**2)
        theta = np.arctan2(y, x)
        
        return r,theta


    def analyze_radial_strains(self,save= False):
        '''
        Description
        -----------
        For all the points in the point cloud DataFrame, transform cartesian 
        strains to polar strains.

        Returns
        -------
        None.

        '''

        # # make x-y coords start at the centroid of the figure
        # x = self.df['x'] - self.centroid[0]
        # y = self.df['y'] - self.centroid[1]
        # # convert x-y to r-theta coords
        # theta = np.arctan2(y, x)
        # r = np.sqrt(x**2+y**2)
        
        r,theta = self.polar_conversion(self.df['x'],self.df['y'])
        

        #add r and theta to the DataFrame
        self.df['r'] = r
        self.df['theta'] = theta
        self.df['a'] = self.get_a(r)
        n=0
        n_max=len(self.df.index)
        for i in self.df.index:
            tensor = np.matrix([[self.df.loc[i,'exx'],self.df.loc[i,'exy']],
                                [self.df.loc[i,'exy'],self.df.loc[i,'eyy']]])
            R = np.matrix([[np.cos(theta[i]), -np.sin(theta[i])], [np.sin(theta[i]), np.cos(theta[i])]])
            tensor = np.matmul(np.matmul(R,tensor),R.transpose())
            self.df.loc[i,'err']=tensor[0, 0]
            self.df.loc[i,'ert']=tensor[1, 0]
            self.df.loc[i,'ett']=tensor[1, 1]
            
            n+=1
            if n %1000 ==999 or n==n_max:
                print(f'{n+1}/{n_max+1} radial points analyzed')
                
        if save:
            print('saving still a work in progress')
            
    def get_a(self, r):
        '''
        Description
        -----------
        Converts radius to parameter, a which is the fraction that we are 
        looking into the thickness of the ring. a = 0 corresponds with the 
        inner surface and a = 1 corresponds with the outer surface.
        
        Parameters
        ----------
        r : float
            radius to evaluate.

        Returns
        -------
        a : float
            output fraction into the thickness of the ring.

        '''
        # from a radial position, return the value of a
        r = np.array(r)
        a = (2*r-self.ID)/(self.OD-self.ID)
        return a

    def get_r(self, a): 
        '''
        Description
        -----------
        Converts the parameter, a, to radius. a is the fraction that we are 
        looking into the thickness of the ring. a = 0 corresponds with the 
        inner surface and a = 1 corresponds with the outer surface.        

        Parameters
        ----------
        a : float
            fraction into ring thickness.

        Returns
        -------
        r : float
            radius to evaluate.

        '''
        # from an a value, return radial position
        a = np.array(a)
        r = (a*self.OD+(1-a)*self.ID)/2
        return r

    def _get_cart_meshgrid(self,a,theta):
        '''
        Description
        -----------
        creates a meshgrid in cartesian coordinates of points based off the 
        input arrays.        
        
        Parameters
        ----------
        a : np.array
            Array of values of parameter, a, to make meshgrid.
        theta : np.array
            Array of theta values to make meshgrid from.

        Returns
        -------
        x : np.array
            meshgrid of horizontal values.
        y : np.array
            meshgrid of vertical values.
        '''

        a, theta = np.meshgrid(a, theta)
        r = self.get_r(a)
        x = r*np.cos(theta)+self.centroid[0]
        y = -r*np.sin(theta)+self.centroid[1]
        return x,y

    def digital_extensometer(self, x1,y1,x2,y2,ext_mode='norm'):
        r1,theta1 = self.polar_conversion(x1,y1)
        r2,theta2 = self.polar_conversion(x2,y2)
        a1 = self.get_a(r1)
        a2 = self.get_a(r2)
        
       
        x_def1 = self.get_value(a1 ,theta1, mode='x_def')
        y_def1 = self.get_value(a1 ,theta1, mode='y_def')
        x_def2 = self.get_value(a2 ,theta2, mode='x_def')
        y_def2 = self.get_value(a2 ,theta2, mode='y_def')
        
        if ext_mode =='norm':
            pass
        elif ext_mode=='x':
            y_def1 = 0
            y_def2 = 0
        elif ext_mode=='y':
            x_def1 = 0
            x_def2 = 0
        
        vect=np.array((x_def1-x_def2,y_def1-y_def2))
        disp = np.linalg.norm(vect,axis=0)
        
        return disp        

    def plot_Image(self, state='deformed', mode='e_vm', log_transform_flag=1e2,
                   max_strain=None, ax=None, **plot_kwargs):

        if mode not in self.df.columns:#if mode is ett,ert,err, then analyze strains
            self.analyze_radial_strains()
        
        f,ax = super().plot_Image(state=state, mode=mode, log_transform_flag=log_transform_flag,
                                  max_strain=max_strain, ax=ax, **plot_kwargs)
        return f,ax

##################################################################
class TensileTest():
    
    def __init__(self, LF_file=None,software='VIC-2D',L0=1,A_x=1,get_geometry_flag=False):
        self.df = pd.read_csv(LF_file)
        
        self.filepath = '/'.join(LF_file.replace('\\','/').split('/')[0:-1])
        self.LF_file =  LF_file.replace('\\','/').split('/')[-1]
        
        for i, row in self.df.iterrows():
            try:
                if len(row['top_img_file'].split('/')) == 1:
                    self.df.loc[i,'top_img_file'] =  self.filepath +'/'+ row['top_img_file']
            except: #No images associated with test
                pass
            try:
                if len(row['side_img_file'].split('/')) == 1:
                    self.df.loc[i,'side_img_file'] =  self.filepath +'/'+ row['side_img_file']
            except:#No side view images in LF file
                pass  
        # if there is an unlabeled column, delete. This comes from saving a 
        # dataframe to a csv without turning off the index
        try:  
            self.df.drop('Unnamed: 0', axis=1, inplace=True)
        except:
            pass


        # figures out how many datapoints long the test is
        self.num_datapoints = self.df.shape[0]

        # set the software we are running
        self.software = software

        self.gauge_length = L0
        self.A_x = A_x

        self.df['stress (MPa)'] = self.df['load (N)']/self.A_x
        self.df['eng_strain'] = self.df['displacement (mm)']/self.gauge_length
        self.df['true_strain'] = np.log(1+self.df['eng_strain'])
        self.df['true_stress'] = self.df['stress (MPa)'] * \
            (1+self.df['eng_strain'])
            
        self.centroid = np.array((0,0))
        self.scale = 1
        if get_geometry_flag:
            self.get_geometry()         
            

    def get_geometry(self):
        img = self.open_Image(0)
        plot_img = mpimg.imread(img.test_img_file)

        f,ax = make_img_figure()
        f.subplots_adjust(top=0.80,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)         

        ax.imshow(plot_img, cmap='gray')
        
        pts = UI_line(ax,'Left click 2 points to make a scale bar.',color='r')
        time.sleep(0.25)
       

        root = tk.Tk()
        d = simpledialog.askfloat(title='Scale',
                                  prompt='Please input the length in mm of this line',
                                  initialvalue = 1)
        root.destroy()
        plt.close()
        disp = np.linalg.norm(np.diff(pts,axis=0))
        
        self.scale = disp/d #pixels/mm
        self.centroid = np.array((0,0))
        

    def digital_extensometer(self, ext_mode='norm', pts = None):
        if pts == None:
            img = self.open_Image(0)
            f,ax = img.plot_Image()
            prompt= 'Choose 2 points to create a digital extensometer on the sample'
            pts = UI_line(ax,prompt)
            plt.close(f)
        pts = pts/self.scale
        x1= pts[0][0]
        y1= pts[0][1]
        x2= pts[1][0]
        y2= pts[1][1]
        

        img = self.open_Image(0)
        x = np.array(img.df['x'])
        y = np.array(img.df['y'])
        idx1 = find_nearest_idx(x,y,x1,y1)
        idx2 = find_nearest_idx(x,y,x2,y2)
        x1 = x[idx1]
        y1 = y[idx1]
        x2 = x[idx2]
        y2 = y[idx2]
            
        adj_displ = []
        for n in range(self.num_datapoints):
        # for n in [0,1,2,3]:
            if n %50 == 49 or n==self.num_datapoints-1:
                print(f'{n+1}/{self.num_datapoints} extensometer points analyzed')
            
            adj_displ.append(self.open_Image(n).digital_extensometer(x1,y1,x2,y2,ext_mode))
        adj_displ = np.array(adj_displ)
        
        self.df['adj_displ'] = adj_displ - adj_displ[0]
        self.df['adj_eng_strain'] = self.df['adj_displ'] / self.gauge_length
        self.df['adj_true_strain'] = np.log(1+self.df['adj_eng_strain'])

    def process_stress_strain_curve(self,x_axis= 'adj_eng_strian',y_axis='stress (MPa)',plot_flag = False):
        """
        Description
        -----------
        Analyzes the ring pull curve as if it were a tensile stress strain 
        curve.  Returns similar outputs as you would get from a tensile 
        analysis.    

        Parameters
        ----------
        plot_flag : bool, optional
            Flag for if the algorithm should plot the final graph. The 
            default is False
        
        Returns
        -------
        true_modulus : float
            The linear slope (quasi-elastic modulus) of the ring pull curve. 
            If the user inputs a different value to correct the curve, this 
            new value will be returned.
        YS : float
            The 0.2% offset strength that is calculated.
        UTS : float
            The maximum strength the material saw.
        eps_u : float
            uniform elongation.
        eps_nu : float
            non-uniform elongation.
        eps_total : float
            total elongation.
        toughness : float
            toughness.

        """
        


        y = self.df[y_axis]
        
        try:  # try to use DIC adjusted strain. if not, use regular strain
            x = self.df[x_axis]
        except:
            x = self.df['eng_strain']
            
        idx = np.invert(np.isnan(x))
        x = np.array(x[idx])
        y = np.array(y[idx]) 
            
        
        f,ax= make_figure()
        ax.plot(x,y,color='C1')
        prompt = 'Left click 2 points to define the elastic region.'
        pts = UI_line(ax,prompt,color='b')
        pts = [(pts[0,0],pts[0,1]),(pts[1,0],pts[1,1])]
        idx = [find_nearest_idx(x,y,pt[0],pt[1]) for pt in pts]
        idx = np.sort(idx)
        plt.close(f)
        c = np.polynomial.polynomial.polyfit(x[idx[0]:idx[1]], y[idx[0]:idx[1]], 1)
        calc_modulus = c[1]
        
        # ask user to specify new modulus which we will correct the curve to meet
        root = tk.Tk()
        true_modulus = simpledialog.askfloat(title='True Modulus',
                                             prompt='The slope is {:3.2} GPa. Please input the modulus you would like to correct to in GPa'.format(calc_modulus/1000),
                                             initialvalue=calc_modulus/1000)
        root.destroy()
        true_modulus = true_modulus*1000
        
        # correct equivalent strain to fix the curve
        x = (x - y*(1/c[1]-1/true_modulus) + c[0]/c[1])
        self.df['mod_adj_eng_strain'] = copy.copy(x)
        
       


        # find UTS
        UTS_idx = np.nanargmax(y)
        UTS = np.nanmax(y)
        eps_u = x[UTS_idx]


        #plot curve
        f,ax = make_figure()
        ax.plot(x,y,color='k')
        ax.plot(0.002+np.array([0,UTS])/true_modulus,[0,UTS],'--',color='C0',lw= 0.5)
        ax.set_xlabel('Strain [mm/mm]')
        ax.set_ylabel('Stress [MPa]')     
        
        #find YS
        prompt = 'Left click 1 point to define the yield strength.'
        pt = UI_get_pts(prompt,n=1,ax=ax)[0]
        YS_idx = find_nearest_idx(x,y,pt[0],pt[1])
        YS = y[YS_idx]
  
        #plot UTS
        ax.plot(x[YS_idx],y[YS_idx],'^',color='C0',label='0.2% offset')
        ax.plot(x[UTS_idx],y[UTS_idx],'dr',color='C1',label='ultimate tensile')
        
        
        # find elongation at fracture
        prompt = 'Left click 1 point to define the fracture point.'
        pt = UI_get_pts(prompt,n=1,ax=ax)[0]
        eps_idx = find_nearest_idx(x,y,pt[0],pt[1])
        eps_total = x[eps_idx]        
        eps_nu = eps_total-eps_u

        #plot elongation
        ax.plot(eps_u*np.ones(2,),[0,UTS],'--',color='C2')
        ax.plot(eps_total*np.ones(2,),[0,y[eps_idx]],'--',color='C3')      
        ax.plot(eps_u,0,'o',color='C2',label='uniform elongation')
        ax.plot(eps_total,0,'s',color='C3',label='total elongation')

        # Find Area under curve (toughness)0:        

        
        toughness = integrate.trapezoid(y, x)
        
        
        if plot_flag:
            ax.legend()
            ax.set_title('')
            f.subplots_adjust(top=0.99,bottom=0.12,left=0.17,right=0.99,hspace=0.2,wspace=0.2)            
        else:
            time.sleep(0.25)
            plt.close(f)

       
        return true_modulus, YS, UTS, eps_u, eps_nu, eps_total, toughness

    def open_Image(self, n):
        """
        Description
        -----------
        opens the DIC_Image class for the index defined by n.

        Parameters
        ----------
        n : int
            index for the image to open
            
        Returns
        -------
        img : DIC_Image
            The DIC_Image that was requested.

        """
        
        # find filename for the image
        img_file = self.df['top_img_file'].iloc[n]
        
        # open Image
        img = TensileImage(test_img = img_file, init_img = self.df['top_img_file'][0], 
                        software = self.software,centroid = self.centroid,scale=self.scale)
        return img

    def save_data(self):
        """
        Description
        -----------
        Asks the user to save the updated file as a csv.     

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        
        root = tk.Tk()
        f = filedialog.asksaveasfile(title='Save Load Frame File As',
                                     filetypes=[('CSV file','.csv')],
                                     defaultextension='.csv',
                                     initialdir=self.filepath,
                                     initialfile=self.LF_file)
        root.destroy()
        
        if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
            print('Did not save data')
        else:
            # save data to output file. 
            self.df.to_csv(f.name, index=False)
            print('Analyzed data saved to the following file:')
            print(f.name)

    def plot_stress_strain(self, ax=None, shift=False,
                           x_axis='eng_strain', y_axis='stress (MPa)',n=None,
                           **plot_kwargs):
        """
        Description
        -----------
        Plots the ring pull stress strain curves. 
        

        Parameters
        ----------
        ax : matplotlib.axes, optional
            The matplotlib axes object to modify. The default is None.
        shift : bool, int, optional
            Integer number for amount of datapoints to shift the graph. If 
            True, shifts the plot to the maximum value. The default is False.
        x_axis : str, optional
            The column of the datafarame to plot on the x-axis. The default is 
            'eng_strain'.
        y_axis : str, optional
            The column of the dataframe to plot on the y-axis. The default is 
            'stress (MPa)'.
        n : int, optional
            the index of the row in the test dataframe. If not None, will plot 
            this point as a large black circle
        plot_kwargs : dict, optional
            Dictionary of key word arguments to pass to the matplotlib plot 
            function. The default is 
            {'linestyle': '-','linewidth':1,'marker':'o','markersize': 1.5}.

        Returns
        -------
        f : matplotlib.figure
            The created figure for the plot
        ax : matplotlib.axes
            The created axes for the plot

        """
        
        default_kwargs = {'linestyle': '-','linewidth':1,'marker':'o','markersize': 1.5}
        plot_kwargs = { **default_kwargs, **plot_kwargs }
        f,ax = make_figure(ax)

        
        x = self.df[x_axis]
        y = self.df[y_axis]

        # shifts the data by as many frames as you want.
        # Or can shift so it is lined up with UTS
        if type(shift) == int or type(shift) == float:
            shift = int(shift)
            x = x-x[shift]
        elif shift:
            shift = self.df['stress (MPa)'].argmax()
            x = x-x[shift]
        else:
            shift = 0

         
        

        ax.plot(x, y, **plot_kwargs)
        if n:
            ax.plot(x[n],y[n],'ok')
        return f, ax
         
    def plot_Image(self,n,**kwargs):
        f,ax = self.open_Image(n).plot_Image(**kwargs)
        return f,ax

class RingPull(TensileTest):
    '''
    Description
    -----------
    A class that contains analysis procedures for both load-displacement and 
    point cloud data. This is a base class for classes that involve MOOSE 
    simulations and experimental data (Load Frame + DIC).
    '''
    def __init__(self, LF_file=None,software=None, ID=None, OD=None, d_mandrel=None, W=None,get_geometry_flag=False):
        """
        Description
        -----------
        Initializes the RingPull class with important geometry features and 
        data.

        Parameters
        ----------
        LF_file : str, optional
            Filepath for the csv file where all the load frame data and image 
            filenames are kept. The default is None.
        software : str, optional
            The software used to generate the data. Options are 'VIC-2D 6', 
            'VIC 2D 7', 'DICe', and 'MOOSE' . The default is None.
        ID : float, optional
            Inner diameter of the ring in mm. The default is None.
        OD : float, optional
            Outer dimater of the ring in mm. The default is None.
        d_mandrel : float, optional
            Mandrel dimater in mm. The default is None.
        W : float, optional
            Width (z-directional dimension) of the ring in mm. The default is None.
        get_geometry_flag : bool, optional
            flag for if the get_geometry function should be run upon 
            initialization. The default is True.
        
        Returns
        -------
        None.

        """
        
        # set geometric dimensions as class attributes
        self.ID = ID
        self.OD = OD
        self.d_avg = (ID+OD)/2
        self.W = W
        self.d_mandrel = d_mandrel
        self.thickness = (OD-ID)/2
        self.OD_path = None
        self.ID_path = None

        

        # find gauge length
        # L. Yegorova, et al., Description of Test Procedures and Analytical Methods, Database on the Behavior of High Burnup Fuel Rods with Zr1%Nb Cladding and UO2 Fuel (VVER Type) under Reactivity Accident Conditions, 2, U.S. Nuclear Regulatory Commission, 1999, pp. 6.16e6.19. NUREG/IA-0156, n.d.
        k = 1
        # gauge_length = pi/2*(self.ID-k*d_mandrel)
        gauge_length = pi/2*((self.ID+self.OD)/2-k*d_mandrel)

        # calculate cross sectional area
        A_x = ((OD-ID)*W)
        
        super().__init__(LF_file,software,gauge_length,A_x,get_geometry_flag)

    def get_a(self, r):
        return self.open_Image(0).get_a(r)        

    def get_r(self, a):
        return self.open_Image(0).get_r(a)   
    
    def open_Image(self, n):
        # find filename for the image
        img_file = self.df['top_img_file'].iloc[n]
        
        # open Image
        img = RingImage(test_img = img_file, init_img = self.df['top_img_file'][0], 
                        software = self.software, 
                        ID=self.ID, OD=self.OD, d_mandrel=self.d_mandrel,
                        OD_path=self.OD_path,ID_path=self.ID_path,
                        scale = self.scale,
                        centroid = self.centroid)
        return img
    


    def get_geometry(self):
        img = self.open_Image(0)
        plot_img = mpimg.imread(img.test_img_file)

        f,ax = make_img_figure()   
        ax.imshow(plot_img, cmap='gray')
        
        
        # prompt = 'Left click at least 5 points on the outer diameter of \nthe ring to define an ellipse.'
        # pts = UI_ellipse(ax,prompt)    
        
        # prompt = 'Left click at least 5 points on the inner diameter of \nthe ring to define an ellipse.'
        # pts2 = UI_ellipse(ax,prompt)    
        
        # self.OD_path = mplPath.Path(pts[0:-1,:])
        # self.ID_path = mplPath.Path(pts2[0:-1,:])        
        # self.centroid = np.mean(pts,axis=0)
        # self.scale = np.mean(np.mean(np.amax(pts[0:-1,:],axis=0) - np.amin(pts[0:-1,:],axis=0))/self.OD) #pixels per mm
        
        
        pts = UI_circle(ax,'Left click 3 points to make the outer circle.',facecolor='r')
        pts2 = UI_circle(ax,'Left click 3 points to make the inner circle.',facecolor='b')

        self.OD_path = mplPath.Path(pts[0:-1,:])
        self.ID_path = mplPath.Path(pts2[0:-1,:])
        self.centroid = np.mean(pts[0:-1,:],axis=0)
        self.scale = np.mean(np.amax(pts[0:-1,:],axis=0) - np.amin(pts[0:-1,:],axis=0))/self.OD #pixels per mm
        
        plt.close(f)

    def plot_side_img(self,n, ax=None,side_view_col='side_img_file',side_img_zoom = ((None,None),(None,None))):
        f,ax = make_img_figure(ax) # if no axes provided, create one
        plot_img = mpimg.imread(self.df['side_img_file'][n]) 
        ax.imshow(plot_img, cmap='gray')
        ax.set_xlim(side_img_zoom[0])
        ax.set_ylim(side_img_zoom[1])  
        return f,ax



class BrazilDisk(RingPull):

    def __init__(self, LF_file=None,software=None, OD=None, W=None,get_geometry_flag=False):

        
        super().__init__(LF_file=LF_file,   software=software, 
                         ID=0,              OD=OD, 
                         d_mandrel=0,       W=W,
                         get_geometry_flag=get_geometry_flag)

        #reproduce some thingd from Tensile Test class
        
        self.gauge_length = OD
        self.A_x = OD*W

        self.df['stress (MPa)'] = self.df['load (N)']/self.A_x
        self.df['eng_strain'] = self.df['displacement (mm)']/self.gauge_length
        self.df['true_strain'] = np.log(1+self.df['eng_strain'])
        self.df['true_stress'] = self.df['stress (MPa)'] * \
            (1+self.df['eng_strain'])
                

    def get_geometry(self):
        img = self.open_Image(0)
        plot_img = mpimg.imread(img.test_img_file)

        f,ax = make_img_figure()   
        ax.imshow(plot_img, cmap='gray')
        
        pts = UI_circle(ax,'Left click 3 points to make the outer circle.',facecolor='r')
        # pts2 = UI_circle(ax,'Left click 3 points to make the inner circle.',facecolor='b')

        self.OD_path = mplPath.Path(pts[0:-1,:])
        self.centroid = np.mean(pts[0:-1,:],axis=0)
        self.scale = np.mean(np.amax(pts[0:-1,:],axis=0) - np.amin(pts[0:-1,:],axis=0))/self.OD #pixels per mm
        plt.close(f)
        



##################################################################

if __name__ == '__main__':  # only runs if this script is the main script
    print('Please go to example script, RPSA_example.py')





