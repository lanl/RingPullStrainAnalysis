'''
Ring Pull Strain Analysis (RPSA)

RPSA is intended as an python module to investigate data from digital image 
correlation software such as DICengine or VIC 2-D. Its primary capability
involves taking an output 2-dimensional mesh of strain data specific to a 
gaugeless Ring Pull test and analyzing it for parameters of interest. RPSA 
also allows input load frame data that is synchronized with the DIC images.

v1.1

Created by:
    Peter Beck
    pmbeck@lanl.gov
Updated:
    21-Nov-2022

'''


# Generic imports
import pickle
import os
import time
import imageio
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
# import matplotlib.gridspec as gridspec

#scipy imports
from scipy import interpolate,integrate

#tkinter imports
import tkinter as tk
from tkinter import simpledialog,filedialog

mpl.rcParams['figure.dpi'] = 300

# import seaborn
# seaborn.set_palette('Set1')



def make_figure(ax=None):
    if ax==None:
        f = plt.figure()
        ax = plt.gca()
    else:
        f = ax.get_figure()
    # axes housekeeping
    ax.minorticks_on()
    ax.tick_params(axis='x', which='major', direction='in', top=True, bottom=True, length=4)
    ax.tick_params(axis='x', which='minor', direction='in', top=True, bottom=True, length=2)
    ax.tick_params(axis='y', which='major', direction='in', left=True, right=True, length=4)
    ax.tick_params(axis='y', which='minor', direction='in', left=True, right=True, length=2)
    ax.tick_params(direction='in')
    return f,ax

def make_img_figure(ax=None):
    if ax==None:
        f = plt.figure(dpi=200)
        ax = plt.axes()
        f.set_size_inches([6.4,4])
    else:
        f = ax.get_figure()

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    return f,ax
    
def make_pixel_box(img, m, n, value, box_size=3):
    # in an image, sets the pixel specified by [m,n]
    # and everthing in a (box_size x box_size) box around it to the input value
    # box_size is rounded up to the nearest odd integer
    box_size = np.floor(box_size/2).astype(int)
    for m_i in range(m-box_size, m+1+box_size):
        for n_i in range(n-box_size, n+1+box_size):
            img[m_i, n_i] = value
    return img

def makeRGBA(r_set,g_set,b_set):
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

def make_GIF(images_list, output_filename, end_pause=True):
    if end_pause:
        if type(end_pause) == bool:
            end_pause = 80
        for n in range(end_pause):  # add some extra images at the end to 'pause' the GIF
            images_list.append(images_list[-1])
    with imageio.get_writer(output_filename, mode='I') as writer:
        for f in images_list:
            image = imageio.imread(f)
            writer.append_data(image)
    print('GIF has been created at the following location:')
    print(output_filename)
    
    
def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
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

def UI_get_pts(prompt,n=None):
    prompt += '\nRight click to finish.\nMiddle mouse to remove points'
    plt.title(prompt)
    plt.draw() 
    while True:
            pts = plt.ginput(1000, timeout=-1,mouse_pop=MouseButton.MIDDLE,mouse_stop=MouseButton.RIGHT)
            if type(n) == type(None):
                break
            elif len(pts)==n:
                break
            else:
                plt.title('Error, retrying.\n'+prompt)
                plt.draw()
    return pts


def UI_circle(ax,prompt,facecolor):
    while True:
        pts = UI_get_pts(prompt,3)
        c,r = define_circle(*pts)
        pts = [
            (r*np.cos(theta)+c[0],
             r*np.sin(theta)+c[1])
            for theta in np.linspace(0,2*pi)]
        my_circle = Polygon(pts, True, facecolor=facecolor,alpha = 0.2)
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

def UI_polygon(ax,prompt,facecolor):
    while True:
        plt.title(prompt)
        plt.draw()
        pts = UI_get_pts(prompt,1000)
        pts.append(pts[0])
        pts = np.array(pts)
        my_polygon = Polygon(pts, True, facecolor=facecolor,alpha = 0.2)
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


def find_nearest_idx(x_array,y_array,x_i,y_i):
    return ((x_array-x_i)**2 + (y_array-y_i)**2).argmin()


# define function to request user input.
def ask_ginput(n,x,y,prompt = 'Pick 2 points to define linear region'):
    f,ax = make_figure()
    plt.title('Click to begin', fontsize=16)
    plt.show()
    plt.waitforbuttonpress()
    while True:
        f.subplots_adjust(top=0.80,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)
        ax.plot(x,y,'-o',color='C1')
        pts = UI_get_pts(prompt,n)
        idx = [find_nearest_idx(x,y,pt[0],pt[1]) for pt in pts]
        plt.plot(x[idx],y[idx],'*-',color='r')
        plt.title('Happy? Mouse click to continue,\nHit Enter to choose new points', fontsize=16)
        plt.draw()
        if plt.waitforbuttonpress():
            plt.close(f)
            del f
            f=plt.figure()
        else:
            plt.close(f)
            del f
            break
    return idx


##################################################################    
class DIC_image:
    '''
    A class to analyze the output image and data file from the DIC software
    Inputs:
        test_img - image that you are analyzing the DIC from
        init_img - initial image of the sample
        DIC_software - the software used to run the image correlation
        ID - the inner diameter of the sample
        OD - the outer diameter of the sample
        d_mandrel - the diameter of the pin used when loading the sample
        data_keep_array - the index of the datapoints in the csv file to keep
        scale - the scale of the image in pixels/mm   
        centroid - the ring centroid pixel in the image      
    '''

    def __init__(self, test_img, init_img,DIC_software='VIC-2D',
                 ID=9.46, OD=10.3, d_mandrel=3,
                 data_keep_array=None,scale = None,centroid = None):

        # read output csv from the DIC program
        df = pd.read_csv(test_img.split('.')[0]+'.csv')
        
        if DIC_software=='VIC-2D 7':
            column_rename_dict = {'  "x"':'x','  "y"':'y',
                                   '  "u"':'u','  "v"':'v',
                                   '  "exx"':'exx','  "eyy"':'eyy','  "exy"':'exy',
                                   '  "e1"':'e1','  "e2"':'e2',
                                   '  "e_vonmises"':'e_vm','  "sigma"':'sigma'}
        elif DIC_software=='VIC-2D 6':
            column_rename_dict = {' "x"':'x','  "y"':'y',
                                   '  "u"':'u','  "v"':'v',
                                   '  "exx"':'exx','  "eyy"':'eyy','  "exy"':'exy',
                                   '  "e1"':'e1','  "e2"':'e2',
                                   '  "e_vonmises"':'e_vm','  "sigma"':'sigma'}
        elif DIC_software=='DICe':
            column_rename_dict = {'COORDINATE_X':'x','COORDINATE_Y':'y',
                                  'DISPLACEMENT_X':'u','DISPLACEMENT_Y':'v',
                                  'SIGMA':'sigma','VSG_STRAIN_XX':'exx',
                                  'VSG_STRAIN_YY':'eyy','VSG_STRAIN_XY':'exy'}
        else:
            print("Could not recognize the DIC data. Please choose a DIC_software of 'VIC-2D 7', 'VIC-2D 6', or 'DICe'")
            assert False
        # rename column names to standard
        df.rename(column_rename_dict, axis=1, inplace=True)

        # if specified, delete all the entries out of the bounds of the ring
        if type(data_keep_array) != type(None):
            df = df[data_keep_array]
            #delete all unnecessary data
            df = df[df['sigma'] != -1]
        
        if type(scale) != type(None):
            # convert from pixels to mm
            self.scale = scale  # pixels/mm
            
            df[['x', 'y', 'u', 'v']] = df[['x', 'y', 'u', 'v']].apply(
                lambda x: x/self.scale)
        else:
            self.scale = 1#default value if there is no scale
        if type(centroid) != type(None):
            self.centroid_pixel = centroid.astype(int)
            self.centroid_scaled = centroid/self.scale
        else:
            # find the centroid as the center between the 
            # largest and smallest x positions where there is data. Same with y.
            centroid = np.array([min(df.x)+max(df.x), min(df.y)+max(df.y)])/2
            self.centroid_pixel = centroid.astype(int)
            self.centroid_scaled = centroid/self.scale
        

        self.test_img_file = test_img
        self.init_img_file = init_img
        
        # set some geometric ring dimensions as class attributes
        self.ID = ID
        self.OD = OD
        self.d_avg = (ID+OD)/2
        self.d_mandrel = d_mandrel


        # find gauge length
        # L. Yegorova, et al., Description of Test Procedures and Analytical Methods, Database on the Behavior of High Burnup Fuel Rods with Zr1%Nb Cladding and UO2 Fuel (VVER Type) under Reactivity Accident Conditions, 2, U.S. Nuclear Regulatory Commission, 1999, pp. 6.16e6.19. NUREG/IA-0156, n.d.
        k = 1
        self.gauge_length = pi/2*(self.d_avg-k*d_mandrel)
        self.pin_angle = pi/2*(1-(k*d_mandrel)/self.d_avg)+pi/2

        if not 'e_1invar' in df.columns:        
            df['e_1invar'] = df['e1']+df['e2']
        if not 'e_hydro' in df.columns:
            df['e_hydro'] = df['e_1invar']/3
            
        # only save the columns that we care about
        df = df[['x', 'y', 'u', 'v', 'exx',
                      'exy', 'eyy', 'e1', 'e2',
                      'e_vm','e_1invar','e_hydro']]        
        self.df = df

    def analyze_radial_strains(self):
        # make x-y coords start at the centroid of the figure
        x = self.df['x'] - self.centroid_scaled[0]
        y = self.df['y'] - self.centroid_scaled[1]
        # convert x-y to r-theta coords
        theta = np.arctan2(y, x)
        # r = np.sqrt(x**2+y**2)
        # a = self.get_a(r)
        # a = pd.Series(a,theta.index)
        # find the strain tensor for each datapoint entry
        n=0
        n_max=len(self.df.index)
        for i in self.df.index:
            # tensor=self.get_strain_tensor(a[i], theta[i], coords='rtheta')[-1]
            
            tensor = np.matrix([[self.df.loc[i,'exx'],self.df.loc[i,'exy']],
                                [self.df.loc[i,'exy'],self.df.loc[i,'eyy']]])
            R = np.matrix([[np.cos(theta[i]), -np.sin(theta[i])], [np.sin(theta[i]), np.cos(theta[i])]])
            tensor = np.matmul(np.matmul(R,tensor),R.transpose())
            
            self.df.loc[i,'err']=tensor[0, 0]
            self.df.loc[i,'ert']=tensor[1, 0]
            self.df.loc[i,'ett']=tensor[1, 1]
            
            n+=1
            if n %1000 ==999:
                print(str(n+1)+'/'+str(n_max+1)+' radial points analyzed')
            elif n==n_max:
                print(str(n+1)+'/'+str(n_max+1)+' radial points analyzed')
       
    def digital_extensometer(self, ext_theta=None,ext_mode='norm'):
        if not ext_theta:
            ext_theta=(self.pin_angle-pi/2,self.pin_angle)
        
        theta_min = ext_theta[0]
        theta_max = ext_theta[1]
        theta = np.linspace(theta_min,theta_max)
        fun = lambda mode:np.array(self.get_value(0.5,theta,mode)[2])
        if ext_mode =='norm':
            position = np.array([fun('x')+fun('u'),fun('y')+fun('v')]).reshape((2,len(theta)))
        elif ext_mode=='x':
            position = np.array([fun('x')+fun('u'),0*fun('y')]).reshape((2,len(theta)))
        elif ext_mode=='y':
            position = np.array([0*fun('x'),fun('y')+fun('v')]).reshape((2,len(theta)))
        vect=np.diff(position)
        return sum(np.linalg.norm(vect,axis=0))

    def find_displacement(self):
        # look at the inner surface of the ring on either side
        a = 0.00
        theta = [0, pi]
        # find the deformation vector, u
        _, _, z = self.get_value(a, theta, mode='u')
        # subtract the two deformation vectors to get displacement
        return float(z[0]-z[1])

    def get_a(self, r):
        # from a radial position, return the value of a
        r = np.array(r)
        return (2*r-self.ID)/(self.OD-self.ID)

    def get_r(self, a):
        # from an a value, return radial position
        a = np.array(a)
        return (a*self.OD+(1-a)*self.ID)/2

    def get_extrap_value(self,a,theta,mode='e_vm'):
        a_i = np.linspace(0,1,100)
        z = self.get_value(a_i,theta,mode,extrap=False)[2]
        z = np.reshape(z, a_i.shape)
        a_i=a_i[np.invert(np.isnan(z))]
        z=z[np.invert(np.isnan(z))]
        
        if (np.size(a_i) <= 1):#if the entire array is NaN or if there is not enough values to extrapolate off of
            return self.get_r(a), theta, np.nan
        else:
            f = interpolate.interp1d(a_i,z,fill_value='extrapolate')
            return self.get_r(a), theta, f(a)

    def get_value(self,a,theta,mode='e_vm',extrap=False):
        # create a meshgrid of a and theta values that encompasses the input a and theta values
        
        if  mode in self.df.columns:#mode is in df
            a, theta = np.meshgrid(a, theta)
            r = self.get_r(a)
            # choose strain value to use
            z = self.df[mode]
            # convert to xy coordinates
            x = r*np.cos(theta)+self.centroid_scaled[0]
            y = -r*np.sin(theta)+self.centroid_scaled[1]
            # mesh interpolation of values
            z = interpolate.griddata((self.df['x'], self.df['y']), z.ravel(),
                                     (x, y), method='cubic')
        else: # mode is ['err','ert','ett'] and have not run self.analyze_radial_strains()
            # gets the strain tensor for each
            _, _, z = self.get_strain_tensor(a, theta, coords='rtheta')
            a, theta = np.meshgrid(a, theta)
            r = self.get_r(a)
            if type(z) != list:
                z = [z]
            z = np.array([z_i[0, 0] if mode == 'err'
                          else z_i[0, 1] if mode == 'ert'
                          else z_i[1, 1] for z_i in z])
            z = np.reshape(z, r.shape)
        if extrap and np.isnan(z).any():
            get_extrap_values = np.vectorize(self.get_extrap_value)
            z[np.isnan(z)] = get_extrap_values(a[np.isnan(z)],theta[np.isnan(z)],mode)[2]
        return r, theta, z


    def get_strain_tensor(self, a=0.5, theta=pi/2, coords='xy'):
        # get strain tensor values in xy coordinates
        _, _, exx = self.get_value(a, theta, mode='exx')
        _, _, exy = self.get_value(a, theta, mode='exy')
        r, theta, eyy = self.get_value(a, theta, mode='eyy')

        # flatten these values so we can iterate over them
        exx = exx.flatten()
        eyy = eyy.flatten()
        exy = exy.flatten()
        r = r.flatten()
        theta = theta.flatten()

        # create list of rotation matrices based on the list of theta
        # if the coords input is 'xy', then it return the identity matrix
        R_list = [np.matrix([[np.cos(theta[i]), -np.sin(theta[i])], [np.sin(theta[i]), np.cos(theta[i])]]) if coords == 'rtheta'
                  else np.matrix([[1, 0], [0, 1]]) for i in range(len(theta))]
        # create list of xy strain tensors
        tensor_list = [np.matrix([[exx[i], exy[i]], [exy[i], eyy[i]]])
                       for i in range(len(theta))]

        # tensor rotation
        # R * T * R^-1; R^-1 = R.transpose()
        tensor_list = [np.matmul(np.matmul(
            R_list[i], tensor_list[i]), R_list[i].transpose()) for i in range(len(tensor_list))]

        # if there was only 1 input, return the tensor, not a list of tensors.
        if len(tensor_list) == 1:
            r = r[0]
            theta = theta[0]
            tensor_list = tensor_list[0]
        return r, theta, tensor_list

    def get_strain_distribution(self, theta=pi/2, mode='ett', extrap = False):
        # get linspace of values spanning the thickness
        a = np.linspace(0, 1, 200)

        # get the strain at each of these values
        _, _, e = self.get_value(a,theta,mode,extrap)
        # transpose this array
        e = e.transpose()

        return a, e
    
    def plot_3d(self,mode='e_vm',ax=None):
        if ax==None:
            f=plt.figure()
            f.subplots_adjust(top=1,bottom=0,left=0.1,right=0.9)
            ax=plt.subplot(1,1,1,projection='3d')
        else:
            f = ax.get_figure()
        a=np.linspace(0,1)
        theta=np.linspace(0,2*pi,300)
        r,theta,e = self.get_value(a, theta, mode)
        
        x=r*np.cos(theta)
        y=r*np.sin(theta)
        ax.plot_surface(x,y,e)
        
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('strain [mm/mm]')
        return f, ax

    def plot_DIC(self, state='deformed', mode='e_vm',log_transform=True, max_strain=None, ax=None, plot_kwargs={'alpha': 1,'cmap':'custom'},pixel_size=3):
        f,ax = make_img_figure(ax) # if no axes provided, create one

        df = self.df
        
        #if looking for rtheta strains and it is not available, then create them
        if mode in ['err','ert','ett'] and mode not in df.columns:
            print('Could not find polar strain values. Analyzing for these values')
            self.analyze_radial_strains()
        if state=='reference':
            plot_img = mpimg.imread(self.init_img_file)
        elif state=='deformed':
            plot_img = mpimg.imread(self.test_img_file)
        
        # create images to fill in DIC strain values
        DIC_img = np.zeros(plot_img.shape, dtype=plot_img.dtype)
        
        if not max_strain: ##if max_strain==None
            max_strain = np.nanmax(abs(df[mode]))
        
        # fill in a blank image with DIC strain values
        for row in df.index:
            if state=='reference':
                n = int(df.loc[row, 'x']*self.scale)
                m = int(df.loc[row, 'y']*self.scale)
            elif state=='deformed':
                n = int((df.loc[row, 'x']+df.loc[row, 'u'])*self.scale)
                m = int((df.loc[row, 'y']+df.loc[row, 'v'])*self.scale)
            e_row = df.loc[row,mode]
            img_value = int(abs(e_row)/max_strain*255)
            
            if log_transform and img_value!=0:#transform into log space
                img_value = np.log(float(img_value))/np.log(255)*255
                img_value=int(img_value)
                
            #converts the absolute value back into positive/negative
            img_value=(img_value*np.sign(e_row)+255)/2
                
            if img_value==0:
                #make it a minimum 1 so that it doesn't get masked later
                img_value=1
            DIC_img = make_pixel_box(DIC_img, m, n, img_value,pixel_size)
        
        
        # add pixels to the centroid of the image
        DIC_img = make_pixel_box(DIC_img, self.centroid_pixel[0], self.centroid_pixel[1], 255,pixel_size)
        
        # Mask the image to only show the pixels !=0
        # this helps with plotting
        DIC_img = ma.masked_where(DIC_img == 0, DIC_img)
        
        #set RGB colors from 0 - 1 for points on the colorbar
        r_set = (.2, 0,.65, .2,  1, 1, .5)
        g_set = (.4, 1, .4, .2, .8, 0,  0)
        b_set = (.2, 0,  1,  1, .4, 0,  0)
        
        try:#if user does not specify color map or if they say colormap = 'custom', then set custom colormap
            if plot_kwargs['cmap']=='custom':
                assert False
        except (KeyError,AssertionError):
            if type(plot_kwargs)==type(None):
                plot_kwargs={}
            plot_kwargs['cmap']=makeRGBA(r_set,g_set,b_set)   
        
        # plot the DIC data on the image
        ax.imshow(plot_img, cmap='gray')
        cmapable=ax.imshow(DIC_img, interpolation='none',vmin=0,vmax=255, **plot_kwargs)
        
        
        cbar=plt.colorbar(cmapable,ax=ax,ticks=[0,128,255])
        cbar.ax.set_yticklabels([f'{-max_strain:.4f}',f'{0:.4f}',f'{max_strain:.4f}'],rotation=45)
        cbar.set_label('strain ('+mode+')', rotation=270, labelpad=10)
        for label in cbar.ax.get_yticklabels():
            label.set_verticalalignment('baseline')
        
        # hide the x-y axes
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        return f, ax
    

    def plot_polar(self, a=[0.85, 0.5, 0.15], mode='e_vm', extrap=False, ax=None,max_strain=None, colors=['#c33124', '#f98365', '#e8a628']):
        if ax == None:
            f, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
        else:
            f = ax.get_figure()
        if type(a) != list:  # if a is a float, turn it into list
            a = [a]
        # create linspace of theta spanning a circle
        n_points = 1000
        theta = np.linspace(0, 2*pi, n_points)
        # find strain at each of these points
        r, _, z = self.get_value(a, theta, mode,extrap)

        # plot unit circle
        ax.plot(theta, np.ones(theta.shape), 'k')

        # plot the strain coming off the unit circle at 10x magnitude
        magnify = 10
        if len(z.shape) > 1:
            for n in range(z.shape[1]):
                ax.plot(theta, 1+magnify*z[:, n],
                        label=a[n], color=colors[n])
            f.legend(frameon=False,loc='upper right')
        else:
            ax.plot(theta, 1+magnify*z, label=a)
            f.legend(frameon=False,loc='upper right')
        # Hide the grid and radial labels. Set the limits in radial direction
        ax.grid(False)
        ax.set_rticks([])
        if max_strain == None:
            ax.set_rlim(0, 1+magnify*z[~np.isnan(z)].max()*1.1)
        else:#max_strain is a float
            ax.set_rlim(0,1+magnify*max_strain)
        
        # creates the lines to plot
        r = np.linspace(0, magnify)
        theta = [n*pi/2 for n in range(4)]
        theta.append(self.pin_angle)
        theta.append(pi-self.pin_angle)
        theta.append(pi+self.pin_angle)
        theta.append(-self.pin_angle)

        # plots the lines
        for angle in theta:
            ax.plot(angle*np.ones(r.shape), r, color='k', linewidth=0.75)
        return f, ax


    def plot_neutral_axis(self, ax=None, plot_min_flag=True):
        if ax == None:  # if no axes provided, create one
            f = plt.figure()
            ax = plt.axes(projection='polar')
        else:
            f = ax.get_figure()
            
        # plots inner and outer diameters with thick black lines
        a = [0, 1]
        theta = np.linspace(0, 2*pi, 200)
        r, theta, _ = self.get_value(a, theta)
        theta = theta[:, 0]
        for n in range(2):
            ax.plot(theta, r[:, n], color='k', linewidth=3)

        # gets strain distribution across the thickness for each theta value
        a, e = self.get_strain_distribution(theta, mode='ett')

        # data manipulation
        e = e.transpose()
        e = ma.array(e, mask=np.isnan(e))

        # find where strain crosses zero
        e_idx = [np.argwhere(np.diff(np.sign(e[i, :]))).flatten()
                 for i in range(e.shape[0])]
        # plots each point that is found where strain crosses zero
        for i in range(e.shape[0]):
            ax.plot(theta[i]*np.ones(e_idx[i].shape),
                    self.get_r(a[e_idx[i]]), 'ro', markersize=3)

        # find and plot minimum abs(strain)
        if plot_min_flag:
            e_idx = np.nanargmin(np.absolute(e), axis=1)
            ax.plot(theta, self.get_r(a[e_idx]), 'b-')

        # axes manipulation to get pretty plots
        ax.set_rlim(self.get_r(0)/1.125, self.get_r(1) * 1.005)
        ax.grid(False)
        ax.set_rticks([])
        ax.spines['polar'].set_visible(False)
        return f, ax



##################################################################
class RingPull():
    '''
    A class that analyzes both the DIC and the tensile data from the ring pull test. Inherits the TensileTest class.
    Inputs:
        LF_file - the csv file where all the load frame data and potentially image filenames are kept
        DIC_software - the software used to run the image correlation
        ID - the ID of the ring in mm
        OD - the OD of the ring in mm
        d_mandrel - the pin diameter in mm
        W - the width of the ring in mm
        get_geometry - a flag to tell the code if you want to run the get_geometry method on initiation
    '''
    
    def __init__(self, LF_file=None,DIC_software=None, ID=None, OD=None, d_mandrel=None, W=None,get_geometry=True):

        #open tkinter module for asking user inputs
        root = tk.Tk()
        #if inputs are not specified, ask user for the inputs
        if type(LF_file) == type(None):
            LF_file = filedialog.askopenfilename(title='Select Load Frame File',
                                       initialdir=os.getcwd())
        if type(DIC_software) == type(None):
            DIC_software = simpledialog.askstring(title='DIC Software',
                             prompt='Which DIC software did you use to analyze the images? VIC-2D or DICe?')
        if type(ID) == type(None):
            ID = simpledialog.askfloat(title='Ring ID',
                                       prompt='What is the inner diameter of the ring in mm?')
        if type(OD) == type(None):
            OD = simpledialog.askfloat(title='Ring OD',
                                       prompt='What is the outer diameter of the ring in mm?')  
        if type(d_mandrel) == type(None):
            d_mandrel = simpledialog.askfloat(title='Mandrel Diameter',
                                               prompt='What is the mandrel diameter in mm?')
        if type(W) == type(None):
            W = simpledialog.askfloat(title='Width',
                                               prompt='What is the ring width in mm?')
        
        #kill the tkinter main loop
        root.destroy()
        
        self.df = pd.read_csv(LF_file)
        
        self.filepath = '/'.join(LF_file.replace('\\','/').split('/')[0:-1])

        for i, row in self.df.iterrows():
            if len(row['top_img_file'].split('/')) == 1:
                self.df.loc[i,'top_img_file'] =  self.filepath +'/'+ row['top_img_file']
            try:
                if len(row['side_img_file'].split('/')) == 1:
                    self.df.loc[i,'side_img_file'] =  self.filepath +'/'+ row['side_img_file']
            except KeyError:#No side view images in LF file
                pass

        

        try:  # if there is an unlabeled column, delete
            self.df.drop('Unnamed: 0', axis=1, inplace=True)
        except:
            pass

        # figures out how many datapoints long the test is
        self.num_datapoints = self.df.shape[0]

        # set the DIC software we are running
        self.DIC_software = DIC_software

        # set geometric dimensions as class attributes
        self.ID = ID
        self.OD = OD
        self.d_avg = (ID+OD)/2
        self.W = W
        self.d_mandrel = d_mandrel
        self.thickness = (OD-ID)/2


        # find gauge length
        # L. Yegorova, et al., Description of Test Procedures and Analytical Methods, Database on the Behavior of High Burnup Fuel Rods with Zr1%Nb Cladding and UO2 Fuel (VVER Type) under Reactivity Accident Conditions, 2, U.S. Nuclear Regulatory Commission, 1999, pp. 6.16e6.19. NUREG/IA-0156, n.d.
        k = 1
        self.gauge_length = pi/2*(self.ID-k*d_mandrel)
        self.pin_angle = pi/2*(1-(k*d_mandrel)/self.ID)+pi/2

        # calculate stress and strain values from load-displacement and save them in the DataFrame
        A_x = ((OD-ID)*W)
        self.df['stress (MPa)'] = self.df['load (N)']/A_x
        self.df['eng_strain'] = self.df['displacement (mm)']/self.gauge_length
        self.df['true_strain'] = np.log(1+self.df['eng_strain'])
        self.df['true_stress'] = self.df['stress (MPa)'] * \
            (1+self.df['eng_strain'])
        
        
        self.data_keep_array = None
        self.centroid = None
        self.scale = None
        if get_geometry:
            self.get_geometry()        
        
    def get_geometry(self):
        try:
            img = self.open_DIC(0)
            plot_img = mpimg.imread(img.test_img_file)
            idx_pts = [(img.df['x'][n],img.df['y'][n]) for n in img.df.index] 
        except FileNotFoundError:
            plot_img = mpimg.imread(self.df['top_img_file'].iloc[0])
            x,y = np.meshgrid(np.arange(plot_img.shape[0]),np.arange(plot_img.shape[1]))
            x=x.flatten()
            y=y.flatten()
            idx_pts = [(x[n],y[n]) for n in range(len(x))]
        
        f,ax = make_img_figure()
        f.subplots_adjust(top=0.80,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)        
        
       # plot the DIC data on the image
        ax.imshow(plot_img, cmap='gray')
        
        pts = UI_circle(ax,'Left click 3 points to make the outer circle.',facecolor='r')
        pts2 = UI_circle(ax,'Left click 3 points to make the inner circle.',facecolor='b')
        time.sleep(0.25)
        plt.close() 
        
        path1 = mplPath.Path(pts[0:-1,:])
        path2 = mplPath.Path(pts2[0:-1,:])
                
        inside = path1.contains_points(idx_pts)*np.invert(path2.contains_points(idx_pts))
        self.data_keep_array = inside
        self.centroid = np.mean(pts[0:-1,:],axis=0)
        self.scale = np.mean(np.amax(pts[0:-1,:],axis=0) - np.amin(pts[0:-1,:],axis=0))/self.OD #pixels per mm

    def open_DIC(self, n, overwrite=False, save_DIC=False, rtheta=False):
        # find filename for the image
        img_file = self.df['top_img_file'].iloc[n]
        output_filename = img_file.split('.')[0]+'.pkl'
        try:  # try opening saved .pkl file. If not, analyze DIC data
            if not overwrite:
                with open(output_filename, 'rb') as handle:
                    img = pickle.load(handle)
            else:
                assert False
        except (FileNotFoundError, AssertionError,PermissionError):
            # open DIC_Image
            img = DIC_image(test_img = img_file, init_img = self.df['top_img_file'][0], 
                            DIC_software = self.DIC_software, 
                            ID=self.ID, OD=self.OD, d_mandrel=self.d_mandrel,
                            data_keep_array=self.data_keep_array,
                            scale = self.scale,
                            centroid = self.centroid)
        #calculate the strains in the radial and hoop direction
        if rtheta:
            try: 
                img.df['ett']
            except:
                img.analyze_radial_strains()
        # Save .pkl file if we want to save it.
        if save_DIC:
            if os.path.exists(output_filename):
                os.remove(output_filename)
            with open(output_filename, 'w+b') as handle:
                pickle.dump(img, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return img
    
    def analyze_single(self, n, ext_theta=None, ext_mode='x',open_kwargs={'overwrite':False,'save_DIC':False,'rtheta':False}):
        if not ext_theta:
            ext_theta=((self.pin_angle-pi/2,self.pin_angle),
                       (-self.pin_angle,-self.pin_angle+pi/2))
        # open DIC_Image for image index n
        img = self.open_DIC(n, **open_kwargs)
        # get values from DIC_Image methods and append to DataFrame
        self.df.loc[n, 'adj_displ'] = img.find_displacement()
        # self.df.loc[n, 'dist_between_gauge'] = img.get_dist_between_gauge()
        self.df.loc[n, 'adj_eng_strain'] = self.df.loc[n,'adj_displ']/self.gauge_length
        self.df.loc[n, 'adj_true_strain'] = np.log(1+self.df.loc[n, 'adj_eng_strain'])

        try:
            self.df.loc[n, 'extensometer 1'] = img.digital_extensometer(ext_theta,ext_mode)
        except:
            for j,theta_i in enumerate(ext_theta):
                self.df.loc[n, 'extensometer '+str(j)] = img.digital_extensometer(theta_i,ext_mode)
                
    def analyze_DIC(self, ext_theta=None, ext_mode='x',open_kwargs={'overwrite':False,'save_DIC':False,'rtheta':False}):
        #create extensomters as an attrbute for later reference of which extensometer is what
        self.extensometers = ext_theta
        
        # run through all the frames in the data and analyze them
        for n in range(len(self.df['top_img_file'])):
            self.analyze_single(n, ext_theta, ext_mode,open_kwargs)
            print(str(n) + '/' + str(len(self.df['top_img_file'])) + ' analyzed')
        self.save_data()

    def get_a(self, r):
        # from a radial position, return the value of a
        r = np.array(r)
        return (2*r-self.ID)/(self.OD-self.ID)

    def get_r(self, a):
        # from an a value, get radial position
        a = np.array(a)
        return (a*self.OD+(1-a)*self.ID)/2

    def get_value(self, n, a, theta, mode='e_vm',extrap=False):
        # get the strain values for each image in the test
        z = [self.open_DIC(n).get_value(a, theta, mode, extrap)[2]
             for n in range(len(self.images_list))]
        # create a meshgrid of r and theta values
        r, theta, _ = self.open_DIC(0).get_value(a, theta, mode,extrap)
        return r, theta, z
    
    def get_side_image_angle(self,n,side_img_file=None):
        root = tk.Tk()
        if side_img_file == None: # else we assume side_img_file is a string with the full filepath
            side_img_file = filedialog.askopenfile(title='Select Side View Image',
                                                   initialdir = '/'.join(self.df['side_img_file'][n].split('/')[0:-1]),
                                                   initialfile=self.df['side_img_file'][n].split('/')[-1])
            side_img_file = side_img_file.name
        root.destroy()
        
        f,ax = make_img_figure()
        f.subplots_adjust(top=0.80,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)
        plot_img = mpimg.imread(side_img_file)
    
        # plot the image and hide the axes
        ax.imshow(plot_img, cmap='gray')
            
        
        #get scale from the side image
        #don't do if scale is already set
        if not hasattr(self,'side_view_scale'):
            prompt = 'Left click 2 points to define the ring thickness.'
            pts = UI_get_pts(prompt=prompt,n=2)
            pts = np.array(pts)
            self.side_view_scale = np.diff(pts[:,1])/self.W #pixels per mm
        
        #get centroid of the ring
        #don't do if center is already set
        if not hasattr(self,'side_view_center'):
            prompt = 'Left click 2 points to define the left and right edges of either the ring or the mandrels.'
            pts = UI_get_pts(prompt=prompt,n=2)
            pts = np.array(pts)        
            self.side_view_center=np.mean(pts[:,0])

        #get point where you are trying to locate the theta
        prompt = 'Left click points on the ring to find theta.'
        pts = UI_get_pts(prompt=prompt)
        pts = np.array(pts)
        side_view_test_point=pts[:,0]     
    
        #close side image plot
        plt.close(f)
        
        #if get_geometry has not been run yet
        if self.scale is None:
            self.get_geometry()
        
        #calc the x distance in mm from the side view centroid to the test point
        side_view_distance =  (side_view_test_point - self.side_view_center)/self.side_view_scale
        
        #calc the x pixel location from the top view centroid to the test point
        x_loc = side_view_distance * self.scale+self.centroid[0] # mm * pixels/mm = pixel
                
        #get point where you are trying to locate the theta
        f,ax = make_img_figure()
        f.subplots_adjust(top=0.80,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)
        plot_img = mpimg.imread(self.df['top_img_file'][n])
        ax.imshow(plot_img, cmap='gray')
        [plt.axvline(x) for x in x_loc]
        prompt = 'Left click points on the ring where the line intersects with the outer diameter.'
        pts = UI_get_pts(prompt=prompt)
        plt.close(f)

        
        img= self.open_DIC(n)
        a = 1
        theta = np.linspace(0,2*pi,361)
            
        x = img.get_value(a,theta,mode='x',extrap=True)[2]
        y = img.get_value(a,theta,mode='y',extrap=True)[2]

        x_pixel = x*self.scale
        y_pixel = y*self.scale      


        idx_list = [find_nearest_idx(x_pixel,y_pixel,pts[i][0],pts[i][1]) for i in range(len(pts))]

        # idx1 = find_nearest_idx(x_pixel,y_pixel,pts[0][0],pts[0][1])
        # idx2 = find_nearest_idx(x_pixel,y_pixel,pts[1][0],pts[1][1])

        return [theta[idx] for idx in idx_list]

    def process_stress_strain_curve(self):
        y = self.df['stress (MPa)']
        try:  # try to use DIC adjusted strain. if not, use regular strain
            x = self.df['adj_eng_strain']
        except:
            x = self.df['eng_strain']
        
        prompt = 'Left click 2 points to define the elastic region.'
        idx = ask_ginput(2,x,y,prompt )
        
        
        c = np.polynomial.polynomial.polyfit(x[idx[0]:idx[1]], y[idx[0]:idx[1]], 1)
        calc_modulus = c[1]

        # ask user to specify new modulus which we will correct the curve to meet
        root = tk.Tk()
        true_modulus = simpledialog.askfloat(title='True Modulus',
                                             prompt='The slope is {:3.2} GPa. Please input the modulus you would like to correct to in GPa'.format(calc_modulus/1000),
                                             initialvalue=calc_modulus/1000)
        root.destroy()
        true_modulus = true_modulus*1000
        
        # correct engineering strain to fix the curve
        x = (x + c[0]/c[1] - y*(1/c[1]-1/true_modulus))
        
        # moving average of the curves for proper YS location
        N = 25
        x_smooth = np.convolve(x, np.ones(N)/N, mode='valid')
        y_smooth = np.convolve(y, np.ones(N)/N, mode='valid')

        # find YS from 0.2% offset
        YS_idx = np.nanargmin(np.absolute(y_smooth-true_modulus*(x_smooth-0.002)))
        YS = y_smooth[YS_idx]
        # find UTS
        UTS_idx = np.nanargmax(y)
        UTS = np.nanmax(y)
        eps_u = x[UTS_idx]-UTS/true_modulus
        # find elongation at fracture
        eps_total = x.iloc[-1]-y.iloc[-1]/true_modulus
        eps_nu = eps_total-eps_u

        # Find Area under curve (toughness)
        toughness = integrate.trapezoid(y, x)

        return true_modulus, YS, UTS, eps_u, eps_nu, eps_total, toughness
    
    
    def plot_side_img(self,n, ax=None):
        f,ax = make_figure(ax) # if no axes provided, create one
        plot_img = mpimg.imread(self.df['side_img_file'][n]) 
        ax.imshow(plot_img, cmap='gray')
        # hide the x-y axes
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        return f,ax
    
        
    def plot_strain_distribution(self, n, theta=pi/2, mode='e_vm',extrap=False, ax=None,
                                 fill=False,  fill_colors = ['#F47558', '#89D279'], 
                                 plot_kwargs={'color': 'k', 'label': 'ett'}):
        f, ax = make_figure(ax)

        # open DIC_Image
        img = self.open_DIC(n)
        # plot the strain distribution
        a, e = img.get_strain_distribution(theta, mode, extrap)
        
        

        # plot the strain distribution
        ax.plot(a, e, **plot_kwargs)
        if fill:
            # only fills if fill=True and if it knows which to fill
            if e.shape[1] > 1:
                print('Cannot fill with multiple lines. Skipping this argument.')
            else:
                # fills the plot with appropriate tension/compression fill colors
                ax.fill_between(a[[e >= 0][0][:, 0]], e[e >= 0],
                                color=fill_colors[0], label='tension')
                ax.fill_between(a[[e <= 0][0][:, 0]], e[e <= 0],
                                color=fill_colors[1], label='compression')
        # modify the axis parameters to create pretty plots
        ax.set_xlim(0, 1)
        ax.set_ylabel('strain [mm/mm]')
        ax.plot([0, 1], [0, 0], '--k')
        ax.tick_params(axis='x', which='minor', direction='in', top=False, bottom=False, length=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['ID', 'OD'])
        ax.legend(frameon=False)
        return f, ax

    def plot_stress_strain(self, ax=None, shift=False,
                           x_axis='eng_strain', y_axis='stress (MPa)',
                           plot_kwargs={'linestyle': '-','marker':'*','markersize': 1.5}):
 
        f,ax = make_figure(ax)# if no axes provided, create one

        # shifts the data by as many frames as you want.
        # Or can shift so it is lined up with UTS
        if shift == True:
            shift = self.df['stress (MPa)'].argmax()
        elif type(shift) == int or type(shift) == float:
            shift = int(shift)
        else:
            shift = 0
        x = self.df[x_axis]
        y = self.df[y_axis]
         
        x = x-x[shift]

        ax.plot(x, y, **plot_kwargs)
        ax.legend(frameon=False)
        return f, ax
    
    def save_data(self):
        root = tk.Tk()
        f = filedialog.asksaveasfile(title='Save Load Frame File As',
                                     filetypes=[('CSV file','.csv')],
                                     defaultextension='.csv',
                                     initialdir=self.filepath,
                                     initialfile='output.csv')
        root.destroy()
        
        if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
            print('ADid not save data')
        else:
            # save data to output file. 
            self.df.to_csv(f.name, index=False)
            print('Analyzed data saved to the following file:')
            print(f.name)    
    
##################################################################

if __name__ == '__main__':  # only run if this script is the main script
    print('Please go to example script, RPSA_example.py')


