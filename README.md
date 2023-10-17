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
    RPSA is intended as an python module to investigate data either from digital 
    image correlation software such as DICengine or VIC 2-D or from physical 
    simulation software such as MOOSE. Its primary capability involves taking an 
    output 2-dimensional mesh of strain data specific to a gaugeless Ring Pull test 
    and analyzing it for parameters of interest. RPSA also allows input load frame 
    data that is synchronized with the DIC images.
    
    v1.3
    
    Created by:
        Peter Beck
        pmbeck@lanl.gov
    Updated:
        03-Oct-2023
    
    
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

CLASSES
    builtins.object
        Image_Base
            Ring_Image
        TensileTest
            RingPull
    
    class Image_Base(builtins.object)
     |  Image_Base(test_img, init_img, software='VIC-2D', centroid=(0, 0), scale=1)
     |  
     |  Description
     |  -----------
     |  Base class for handling and calculating strains on a dogbone sample during tensile testing.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, test_img, init_img, software='VIC-2D', centroid=(0, 0), scale=1)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  digital_extensometer(self, x1, y1, x2, y2, ext_mode='norm')
     |  
     |  get_value(self, x, y, mode='e_vm', extrap=False)
     |  
     |  get_value_deformed(self, x_def, y_def, mode='e_vm', extrap=False)
     |  
     |  plot_Image(self, state='deformed', mode='e_vm', log_transform_flag=100.0, max_strain=None, ax=None, **plot_kwargs)
     |      Description
     |      -----------
     |      A heat map of the strain around the ring. Specific to DIC data.       
     |      
     |      Parameters
     |      ----------
     |      state : str, optional
     |          Whether you want the ring plotted in the 'reference' or 'deformed' 
     |          state. The default is 'deformed'.
     |      mode : str, optional
     |          The type of strain to plot. The default is 'e_vm'.
     |      log_transform_flag : bool, float, optional
     |          Flag for if the data should be plotted on a nonlinear scale. This 
     |          nonlinear transform also uses this variable as a parameter for how 
     |          nonlinear to make it. The default is 1e2.
     |      max_strain : float, optional
     |          Sets the plot limits for strain to plot. If False or None, the
     |          function autoscales. The default is None.
     |      ax : matplotlib.axes, optional
     |          The matplotlib axes object to plot on. The default is None.
     |      plot_kwargs : dict, optional
     |          Dictionary of key word arguments to pass to the matplotlib scatter 
     |          function. The default is {'alpha': 1,'cmap':'custom','s':0.25}.
     |      
     |      Returns
     |      -------
     |      f : matplotlib.figure
     |          The created figure for the plot
     |      ax : matplotlib.axes
     |          The created axes for the plot
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class RingPull(TensileTest)
     |  RingPull(LF_file=None, software=None, ID=None, OD=None, d_mandrel=None, W=None, get_geometry_flag=False)
     |  
     |  Description
     |  -----------
     |  A class that contains analysis procedures for both load-displacement and 
     |  point cloud data. This is a base class for classes that involve MOOSE 
     |  simulations and experimental data (Load Frame + DIC).
     |  
     |  Method resolution order:
     |      RingPull
     |      TensileTest
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, LF_file=None, software=None, ID=None, OD=None, d_mandrel=None, W=None, get_geometry_flag=False)
     |      Description
     |      -----------
     |      Initializes the RingPull class with important geometry features and 
     |      data.
     |      
     |      Parameters
     |      ----------
     |      LF_file : str, optional
     |          Filepath for the csv file where all the load frame data and image 
     |          filenames are kept. The default is None.
     |      software : str, optional
     |          The software used to generate the data. Options are 'VIC-2D 6', 
     |          'VIC 2D 7', 'DICe', and 'MOOSE' . The default is None.
     |      ID : float, optional
     |          Inner diameter of the ring in mm. The default is None.
     |      OD : float, optional
     |          Outer dimater of the ring in mm. The default is None.
     |      d_mandrel : float, optional
     |          Mandrel dimater in mm. The default is None.
     |      W : float, optional
     |          Width (z-directional dimension) of the ring in mm. The default is None.
     |      get_geometry_flag : bool, optional
     |          flag for if the get_geometry function should be run upon 
     |          initialization. The default is True.
     |      
     |      Returns
     |      -------
     |      None.
     |  
     |  get_a(self, r)
     |  
     |  get_geometry(self)
     |      Description
     |      -----------
     |      Function that opens the initial image and has the user click points to 
     |      make the ring geometry. Uses this to define the scale and centroid of 
     |      the ring.
     |      
     |      Parameters
     |      ----------
     |      None.
     |      
     |      Returns
     |      -------
     |      None.
     |  
     |  get_r(self, a)
     |  
     |  open_Image(self, n)
     |      Description
     |      -----------
     |      opens the DIC_Image class for the index defined by n.
     |      
     |      Parameters
     |      ----------
     |      n : int
     |          index for the image to open
     |          
     |      Returns
     |      -------
     |      img : DIC_Image
     |          The DIC_Image that was requested.
     |  
     |  plot_side_img(self, n, ax=None, side_img_zoom=((None, None), (None, None)), side_view_col='side_img_file')
     |  
     |  plot_strain_distribution(self, n, theta=1.5707963267948966, mode='ett', extrap=False, ax=None, **plot_kwargs)
     |      Description
     |      -----------
     |      Plots the strain distribution across the thickness of the ring. 
     |      Parameters
     |      ----------
     |      n : TYPE
     |          DESCRIPTION.
     |      theta : float, np.array, optional
     |          The value of theta to get strain value from. The default is pi/2.
     |      mode : str, optional
     |          The type of strain to return. The default is 'ett'.
     |      extrap : TYPE, optional
     |          Whether or not to use extrapolation to get values. If False, the 
     |          function returns NaN values where it is not defined. The default 
     |          is False.
     |      ax : matplotlib.axes, optional
     |          The matplotlib axes object to modify. The default is None.
     |      fill : bool, optional
     |          Flag for if the area unde the curve should be filled. The default is False.
     |      fill_colors : list of 2 colors, optional
     |          List of col. The default is ['#F47558', '#89D279'].
     |      plot_kwargs : dict, optional
     |          Dictionary of colors. The default is color = k.
     |      Returns
     |      -------
     |      f : matplotlib.figure
     |          The created figure for the plot
     |      ax : matplotlib.axes
     |          The created axes for the plot
     |  
     |  plot_top_img(self, n, ax=None, top_img_zoom=((None, None), (None, None)), **plot_kwargs)
     |  
     |  plot_top_side_img(self, n, top_img_zoom=((None, None), (None, None)), side_img_zoom=((None, None), (None, None)), side_view_col='side_img_file')
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from TensileTest:
     |  
     |  adjust_displacements(self, x1, y1, x2, y2, ext_mode='norm')
     |  
     |  plot_Image(self, n, **kwargs)
     |  
     |  plot_stress_strain(self, ax=None, shift=False, x_axis='eng_strain', y_axis='stress (MPa)', n=None, **plot_kwargs)
     |      Description
     |      -----------
     |      Plots the ring pull stress strain curves. 
     |      
     |      
     |      Parameters
     |      ----------
     |      ax : matplotlib.axes, optional
     |          The matplotlib axes object to modify. The default is None.
     |      shift : bool, int, optional
     |          Integer number for amount of datapoints to shift the graph. If 
     |          True, shifts the plot to the maximum value. The default is False.
     |      x_axis : str, optional
     |          The column of the datafarame to plot on the x-axis. The default is 
     |          'eng_strain'.
     |      y_axis : str, optional
     |          The column of the dataframe to plot on the y-axis. The default is 
     |          'stress (MPa)'.
     |      n : int, optional
     |          the index of the row in the test dataframe. If not None, will plot 
     |          this point as a large black circle
     |      plot_kwargs : dict, optional
     |          Dictionary of key word arguments to pass to the matplotlib plot 
     |          function. The default is 
     |          {'linestyle': '-','linewidth':1,'marker':'o','markersize': 1.5}.
     |      
     |      Returns
     |      -------
     |      f : matplotlib.figure
     |          The created figure for the plot
     |      ax : matplotlib.axes
     |          The created axes for the plot
     |  
     |  process_stress_strain_curve(self, plot_flag=False)
     |      Description
     |      -----------
     |      Analyzes the ring pull curve as if it were a tensile stress strain 
     |      curve.  Returns similar outputs as you would get from a tensile 
     |      analysis.    
     |      
     |      Parameters
     |      ----------
     |      plot_flag : bool, optional
     |          Flag for if the algorithm should plot the final graph. The 
     |          default is False
     |      
     |      Returns
     |      -------
     |      true_modulus : float
     |          The linear slope (quasi-elastic modulus) of the ring pull curve. 
     |          If the user inputs a different value to correct the curve, this 
     |          new value will be returned.
     |      YS : float
     |          The 0.2% offset strength that is calculated.
     |      UTS : float
     |          The maximum strength the material saw.
     |      eps_u : float
     |          uniform elongation.
     |      eps_nu : float
     |          non-uniform elongation.
     |      eps_total : float
     |          total elongation.
     |      toughness : float
     |          toughness.
     |  
     |  save_data(self)
     |      Description
     |      -----------
     |      Asks the user to save the updated file as a csv.     
     |      
     |      Parameters
     |      ----------
     |      None.
     |      
     |      Returns
     |      -------
     |      None.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from TensileTest:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Ring_Image(Image_Base)
     |  Ring_Image(test_img, init_img, software='VIC-2D 7', ID=9.46, OD=10.3, d_mandrel=3, OD_path=None, ID_path=None, scale=None, centroid=None)
     |  
     |  Description
     |  -----------
     |  A class to analyze the output image and data file from the DIC software. 
     |  This class inherets from Image_Base and modifies methods pertaining to 
     |  experimental (DIC) data.
     |  
     |  Method resolution order:
     |      Ring_Image
     |      Image_Base
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, test_img, init_img, software='VIC-2D 7', ID=9.46, OD=10.3, d_mandrel=3, OD_path=None, ID_path=None, scale=None, centroid=None)
     |      Description
     |      -----------
     |      Initializes DIC_Image class.
     |      
     |      Parameters
     |      ----------
     |      test_img : str
     |          Filepath for the image that we are analyzing.
     |      init_img : str
     |          Filepath for the initial image.
     |      software : str, optional
     |          The software used to run the image correlation. The default is
     |          'VIC-2D 7'.
     |      ID : float, optional
     |          The inner diameter of the sample in mm. The default is 9.46.
     |      OD : float, optional
     |          The outer dimater of the sample in mm. The default is 10.3.
     |      d_mandrel : float, optional
     |          the diameter of the mandrel used when loading the sample in mm. 
     |          The default is 3.
     |      OD_path : mplPath.Path object, optional
     |          Path of the outer diameter which will be used for indexing points 
     |          inside the area of interest(AOI). This is primarily going to be passed 
     |          from the RingPull class. If None, will not exclude points outside 
     |          the AOI. The default is None.
     |      ID_path : mplPath.Path object, optional
     |          Path of the inner diameter which will be used for indexing points 
     |          inside the area of interest(AOI). This is primarily going to be passed 
     |          from the RingPull class. If the variable OD_path is None, will not 
     |          exclude points outside the AOI. The default is None.
     |      scale : float, optional
     |          The scale of the image in pixels/mm. If None, sets the scale to 1.
     |          The default is None.
     |      centroid : np.array, optional
     |          Length 2 array of the (x,y) coordinates of the centroid of the 
     |          ring. This is used to convert to polar coordinates. The default 
     |          is None.
     |      
     |      Returns
     |      -------
     |      None.
     |  
     |  analyze_radial_strains(self)
     |      Description
     |      -----------
     |      For all the points in the point cloud DataFrame, transform cartesian 
     |      strains to polar strains.
     |      
     |      Returns
     |      -------
     |      None.
     |  
     |  digital_extensometer(self, ext_theta, a=0.5, ext_mode='norm')
     |      Description
     |      -----------
     |      Tracks two points during the test and measures their relative movement
     |      to each other. This is essentially a virtual extensometer that can be
     |      placed anywhere. Can track either horizontal (x) displacement, 
     |      vertical (y), or total displacement (norm).
     |      
     |      Parameters
     |      ----------
     |      ext_theta : list,np.array
     |          A length 2 list of the 2 angles to track. If None or False is
     |          passed, will use the pin angle as theta values.
     |      a : float, optional
     |          The parameter a at which to evaluate the points. The default is 0.5.
     |      ext_mode : str, optional
     |          Options: 'x', 'y', 'norm'. The default is 'norm'.
     |      
     |      Returns
     |      -------
     |      disp : float
     |          The displacement value from the start of the test.
     |  
     |  find_displacement(self)
     |      Description
     |      -----------
     |      Calculates the displacement of the mandrels by using the inner 
     |      surface of the ring where it contacts the mandrels. This is used to 
     |      bypass complicance of the load frame.
     |      
     |      Returns
     |      -------
     |      disp : float
     |          The calculated displacement.
     |  
     |  get_a(self, r)
     |      Description
     |      -----------
     |      Converts radius to parameter, a which is the fraction that we are 
     |      looking into the thickness of the ring. a = 0 corresponds with the 
     |      inner surface and a = 1 corresponds with the outer surface.
     |      
     |      Parameters
     |      ----------
     |      r : float
     |          radius to evaluate.
     |      
     |      Returns
     |      -------
     |      a : float
     |          output fraction into the thickness of the ring.
     |  
     |  get_r(self, a)
     |      Description
     |      -----------
     |      Converts the parameter, a, to radius. a is the fraction that we are 
     |      looking into the thickness of the ring. a = 0 corresponds with the 
     |      inner surface and a = 1 corresponds with the outer surface.        
     |      
     |      Parameters
     |      ----------
     |      a : float
     |          fraction into ring thickness.
     |      
     |      Returns
     |      -------
     |      r : float
     |          radius to evaluate.
     |  
     |  get_strain_distribution(self, theta=1.5707963267948966, mode='ett', extrap=False, N=200)
     |      Description
     |      -----------
     |      Returns the distribution of strain across the thickness of the ring.
     |      
     |      
     |      Parameters
     |      ----------
     |      theta : float, np.array, optional
     |          The value of theta to get strain value from. The default is pi/2.
     |      mode : str, optional
     |          The type of strain to return. The default is 'ett'.
     |      extrap : TYPE, optional
     |          Whether or not to use extrapolation to get values. If False, the 
     |          function returns NaN values where it is not defined. The default 
     |          is False.
     |      
     |      Returns
     |      -------
     |      a : np.array
     |          Array of a values from 0 to 1.
     |      e : np.array
     |          Matrix of strain values that correspond with the array of a values
     |          and the input theta.
     |  
     |  get_value(self, a, theta, mode='e_vm', extrap=False)
     |      Description
     |      -----------
     |      Returns the requested values of strain or displacement at a specific 
     |      point defined by a and theta. This value will be interpolated between
     |      the surrounding grid points. Optional extrapolation as described in the 
     |      _get_extrap_values function.
     |      
     |      Parameters
     |      ----------
     |      a : float, np.array
     |          The location, a within the ring of the requested values. 
     |          a = 0 corresponds with the inner surface of the ring and
     |          a = 1 corresponds with the outer surface of the ring.
     |      theta : float, np.array
     |          The angular location of the requested values.
     |      mode : str, optional
     |          The reuqested strain/diplacement value to pull from the point 
     |          cloud. The default is 'e_vm'.
     |      extrap : bool, optional
     |          Boolean flag for if extrapolation is requested. The default is False.
     |      
     |      Returns
     |      -------
     |      z : float, np.array
     |          The return value(s) of strain/displacement.
     |  
     |  plot_Image(self, state='deformed', mode='e_vm', log_transform_flag=100.0, max_strain=None, ax=None, **plot_kwargs)
     |      Description
     |      -----------
     |      A heat map of the strain around the ring. Specific to DIC data.       
     |      
     |      Parameters
     |      ----------
     |      state : str, optional
     |          Whether you want the ring plotted in the 'reference' or 'deformed' 
     |          state. The default is 'deformed'.
     |      mode : str, optional
     |          The type of strain to plot. The default is 'e_vm'.
     |      log_transform_flag : bool, float, optional
     |          Flag for if the data should be plotted on a nonlinear scale. This 
     |          nonlinear transform also uses this variable as a parameter for how 
     |          nonlinear to make it. The default is 1e2.
     |      max_strain : float, optional
     |          Sets the plot limits for strain to plot. If False or None, the
     |          function autoscales. The default is None.
     |      ax : matplotlib.axes, optional
     |          The matplotlib axes object to plot on. The default is None.
     |      plot_kwargs : dict, optional
     |          Dictionary of key word arguments to pass to the matplotlib scatter 
     |          function. The default is {'alpha': 1,'cmap':'custom','s':0.25}.
     |      
     |      Returns
     |      -------
     |      f : matplotlib.figure
     |          The created figure for the plot
     |      ax : matplotlib.axes
     |          The created axes for the plot
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Image_Base:
     |  
     |  get_value_deformed(self, x_def, y_def, mode='e_vm', extrap=False)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from Image_Base:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class TensileTest(builtins.object)
     |  TensileTest(LF_file=None, software=None, L0=1, A_x=1)
     |  
     |  ##################################################################
     |  
     |  Methods defined here:
     |  
     |  __init__(self, LF_file=None, software=None, L0=1, A_x=1)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  adjust_displacements(self, x1, y1, x2, y2, ext_mode='norm')
     |  
     |  open_Image(self, n)
     |      Description
     |      -----------
     |      opens the DIC_Image class for the index defined by n.
     |      
     |      Parameters
     |      ----------
     |      n : int
     |          index for the image to open
     |          
     |      Returns
     |      -------
     |      img : DIC_Image
     |          The DIC_Image that was requested.
     |  
     |  plot_Image(self, n, **kwargs)
     |  
     |  plot_stress_strain(self, ax=None, shift=False, x_axis='eng_strain', y_axis='stress (MPa)', n=None, **plot_kwargs)
     |      Description
     |      -----------
     |      Plots the ring pull stress strain curves. 
     |      
     |      
     |      Parameters
     |      ----------
     |      ax : matplotlib.axes, optional
     |          The matplotlib axes object to modify. The default is None.
     |      shift : bool, int, optional
     |          Integer number for amount of datapoints to shift the graph. If 
     |          True, shifts the plot to the maximum value. The default is False.
     |      x_axis : str, optional
     |          The column of the datafarame to plot on the x-axis. The default is 
     |          'eng_strain'.
     |      y_axis : str, optional
     |          The column of the dataframe to plot on the y-axis. The default is 
     |          'stress (MPa)'.
     |      n : int, optional
     |          the index of the row in the test dataframe. If not None, will plot 
     |          this point as a large black circle
     |      plot_kwargs : dict, optional
     |          Dictionary of key word arguments to pass to the matplotlib plot 
     |          function. The default is 
     |          {'linestyle': '-','linewidth':1,'marker':'o','markersize': 1.5}.
     |      
     |      Returns
     |      -------
     |      f : matplotlib.figure
     |          The created figure for the plot
     |      ax : matplotlib.axes
     |          The created axes for the plot
     |  
     |  process_stress_strain_curve(self, plot_flag=False)
     |      Description
     |      -----------
     |      Analyzes the ring pull curve as if it were a tensile stress strain 
     |      curve.  Returns similar outputs as you would get from a tensile 
     |      analysis.    
     |      
     |      Parameters
     |      ----------
     |      plot_flag : bool, optional
     |          Flag for if the algorithm should plot the final graph. The 
     |          default is False
     |      
     |      Returns
     |      -------
     |      true_modulus : float
     |          The linear slope (quasi-elastic modulus) of the ring pull curve. 
     |          If the user inputs a different value to correct the curve, this 
     |          new value will be returned.
     |      YS : float
     |          The 0.2% offset strength that is calculated.
     |      UTS : float
     |          The maximum strength the material saw.
     |      eps_u : float
     |          uniform elongation.
     |      eps_nu : float
     |          non-uniform elongation.
     |      eps_total : float
     |          total elongation.
     |      toughness : float
     |          toughness.
     |  
     |  save_data(self)
     |      Description
     |      -----------
     |      Asks the user to save the updated file as a csv.     
     |      
     |      Parameters
     |      ----------
     |      None.
     |      
     |      Returns
     |      -------
     |      None.
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
    UI_circle(ax, prompt, facecolor, num=50)
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
    
    UI_get_pts(prompt, n=None)
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
    
    UI_line(ax, prompt, color='r')
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
    
    UI_polygon(ax, prompt, facecolor)
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
    
    define_circle(p1, p2, p3)
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
    
    find_nearest_idx(x_array, y_array, x_q, y_q)
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
    
    log_transform(e_row, max_strain, lim=100.0)
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
    
    makeRGBA(r_set, g_set, b_set)
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
    
    make_3D_figure(ax=None)
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
    
    make_GIF(images_list, output_filename, end_pause=True, **writer_kwargs)
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
    
    make_figure(ax=None)
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
    
    make_img_figure(ax=None)
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

DATA
    custom_cmap = <matplotlib.colors.ListedColormap object>
    pi = 3.141592653589793



