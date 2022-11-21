# RingPullStrainAnalysis

This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.

2.Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.

3.Neither the name of the copyright holder nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

NAME
    RingPullStrainAnalysis - Ring Pull Strain Analysis (RPSA)

DESCRIPTION
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
        4-Oct-2022

NAME
    RingPullStrainAnalysis - Ring Pull Strain Analysis (RPSA)

DESCRIPTION
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

CLASSES
    builtins.object
        DIC_image
        RingPull
    
    class DIC_image(builtins.object)
     |  DIC_image(test_img, init_img, DIC_software='VIC-2D', ID=9.46, OD=10.3, d_mandrel=3, data_keep_array=None, scale=None, centroid=None)
     |  
     |  A class to analyze the output image and data file from the DIC software
     |  Inputs:
     |      test_img - image that you are analyzing the DIC from
     |      init_img - initial image of the sample
     |      DIC_software - the software used to run the image correlation
     |      ID - the inner diameter of the sample
     |      OD - the outer diameter of the sample
     |      d_mandrel - the diameter of the pin used when loading the sample
     |      data_keep_array - the index of the datapoints in the csv file to keep
     |      scale - the scale of the image in pixels/mm   
     |      centroid - the ring centroid pixel in the image
     |  
     |  Methods defined here:
     |  
     |  __init__(self, test_img, init_img, DIC_software='VIC-2D', ID=9.46, OD=10.3, d_mandrel=3, data_keep_array=None, scale=None, centroid=None)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  analyze_radial_strains(self)
     |  
     |  digital_extensometer(self, ext_theta=None, ext_mode='norm')
     |  
     |  find_displacement(self)
     |  
     |  get_a(self, r)
     |  
     |  get_extrap_value(self, a, theta, mode='e_vm')
     |  
     |  get_r(self, a)
     |  
     |  get_strain_distribution(self, theta=1.5707963267948966, mode='ett', extrap=False)
     |  
     |  get_strain_tensor(self, a=0.5, theta=1.5707963267948966, coords='xy')
     |  
     |  get_value(self, a, theta, mode='e_vm', extrap=False)
     |  
     |  plot_3d(self, mode='e_vm', ax=None)
     |  
     |  plot_DIC(self, state='deformed', mode='e_vm', log_transform=True, max_strain=None, ax=None, plot_kwargs={'alpha': 1, 'cmap': 'custom'}, pixel_size=3)
     |  
     |  plot_neutral_axis(self, ax=None, plot_min_flag=True)
     |  
     |  plot_polar(self, a=[0.85, 0.5, 0.15], mode='e_vm', extrap=False, ax=None, max_strain=None, colors=['#c33124', '#f98365', '#e8a628'])
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class RingPull(builtins.object)
     |  RingPull(LF_file=None, DIC_software=None, ID=None, OD=None, d_mandrel=None, W=None, get_geometry=True)
     |  
     |  A class that analyzes both the DIC and the tensile data from the ring pull test. Inherits the TensileTest class.
     |  Inputs:
     |      LF_file - the csv file where all the load frame data and potentially image filenames are kept
     |      DIC_software - the software used to run the image correlation
     |      ID - the ID of the ring in mm
     |      OD - the OD of the ring in mm
     |      d_mandrel - the pin diameter in mm
     |      W - the width of the ring in mm
     |      get_geometry - a flag to tell the code if you want to run the get_geometry method on initiation
     |  
     |  Methods defined here:
     |  
     |  __init__(self, LF_file=None, DIC_software=None, ID=None, OD=None, d_mandrel=None, W=None, get_geometry=True)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  analyze_DIC(self, ext_theta=None, ext_mode='x', open_kwargs={'overwrite': False, 'save_DIC': False, 'rtheta': False})
     |  
     |  analyze_single(self, n, ext_theta=None, ext_mode='x', open_kwargs={'overwrite': False, 'save_DIC': False, 'rtheta': False})
     |  
     |  get_a(self, r)
     |  
     |  get_geometry(self)
     |  
     |  get_r(self, a)
     |  
     |  get_side_image_angle(self, n, side_img_file=None)
     |  
     |  get_value(self, n, a, theta, mode='e_vm', extrap=False)
     |  
     |  open_DIC(self, n, overwrite=False, save_DIC=False, rtheta=False)
     |  
     |  plot_side_img(self, n, ax=None)
     |  
     |  plot_strain_distribution(self, n, theta=1.5707963267948966, mode='e_vm', extrap=False, ax=None, fill=False, fill_colors=['#F47558', '#89D279'], plot_kwargs={'color': 'k', 'label': 'ett'})
     |  
     |  plot_stress_strain(self, ax=None, shift=False, x_axis='eng_strain', y_axis='stress (MPa)', plot_kwargs={'linestyle': '-', 'marker': '*', 'markersize': 1.5})
     |  
     |  process_stress_strain_curve(self)
     |  
     |  save_data(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FUNCTIONS
    UI_circle(ax, prompt, facecolor)
    
    UI_get_pts(prompt, n=None)
    
    UI_polygon(ax, prompt, facecolor)
    
    ask_ginput(n, x, y, prompt='Pick 2 points to define linear region')
        # define function to request user input.
    
    define_circle(p1, p2, p3)
        Returns the center and radius of the circle passing the given 3 points.
        In case the 3 points form a line, returns (None, infinity).
    
    find_nearest_idx(x_array, y_array, x_i, y_i)
    
    makeRGBA(r_set, g_set, b_set)
    
    make_GIF(images_list, output_filename, end_pause=True)
    
    make_figure(ax=None)
    
    make_img_figure(ax=None)
    
    make_pixel_box(img, m, n, value, box_size=3)

DATA
    pi = 3.141592653589793


