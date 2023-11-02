'''
Ring Pull Coating Analysis (RPCA)

RPCA is intended as an add-on to the Ring Pull Strain Analysis (RPSA) module.
It allows analysis of the failure strain of coating layers on rings that were 
tested via the gaugeless ring pull technique. RPCA consists of a module which 
takes an input the RingPull class defined in RPSA. This module includes 
functions for correlating points located on side views of the ring with DIC 
strain maps taken from the top-down view.

v1.0

Created by:
    Peter Beck
    pmbeck@lanl.gov
Updated:
    03-Oct-2023

General notes:
    - both the RPSA and the RPCA scripts should be placed in the same folder.
        This module will not work otherwise.
    - there are two classes here: RingPullCoating1 and RingPullCoating2. The
        primary difference between these is where the side-view cameras are 
        located.
    - The classes save the results in the form of a multi-indexed pandas 
        DataFrame.
    - Debugging methods are available based on the saved dataframe for manual 
        validation of accuracy.
    - this module was created with the following settings:
        - Windows 10
        - Python 3.8.8 (installed via Anaconda)
        - Spyder 4, with the setting:
            - IPython console -> Graphics -> Backend: Automatic

'''


from RingPullStrainAnalysis import *

class RingPullCoatingAnalysis1():
    def __init__(self, RingPull, side_view_col='side_img_file'): 
        self.test = RingPull
        self.side_view_col = side_view_col
    
    def get_side_image_strain(self,n,N=50,M=10, side_img_zoom=[(None,None),(None,None)],debug=False):
        # if n is not a list or array,make it so
        if type(n) != list and type(n) != type(np.array([])):
            n = [n]
        
        pts_array,x_loc_array_top, x_loc_array_side = self._side_image_angle_UI(n,side_img_zoom)
        
        print('starting angle analysis')
        angles_array = self._side_image_angle_analysis(n,pts_array)
        
        print('starting strain analysis')
        extrap_points_array, reg_points_array, nearest_values_array = self._side_image_strain_analysis(n,angles_array,N,M)
        
        if debug:
            print('Debugging on. Calculating and generating graphs.')
            for i, n_i in enumerate(n):
                self._plot_debug_angle_analysis(n_i,angles_array[i],x_loc_array_top[i],x_loc_array_side[i],side_img_zoom)
                self._plot_debug_strain_analysis(n_i, angles_array[i])

        print('Finished analysis. Saving data.')
        data = self._save_side_image_strain(n,angles_array,extrap_points_array,reg_points_array,nearest_values_array,x_loc_array_top,x_loc_array_side)

        return data

    def _side_image_angle_UI(self,n,side_img_zoom=[(None,None),(None,None)]):
        #if get_geometry has not been run yet
        if self.test.scale is None:
            self.test.get_geometry()
        
        x_loc_array_side = []
        x_loc_array_top = []
        
        self.get_side_img_scale(n[0],side_img_zoom)
        
        for n_i in n:
            print('Image ' + str(n_i))
            # plot the side view image
            f,ax = self.test.plot_side_img(n_i,side_img_zoom=side_img_zoom,side_view_col= self.side_view_col)       

            #get point where you are trying to locate the theta
            prompt = 'Left click points on the ring to find theta.'
            pts = UI_get_pts(prompt=prompt)
            #close side image plot   
            plt.close(f)
            
            pts = np.array(pts)
            x_loc_side=pts[:,0]

            side_view_distance =  (x_loc_side - self.side_view_center) / self.side_view_scale  
            x_loc_top = -side_view_distance * self.test.scale+self.test.centroid[0] # mm * pixels/mm = pixel   
            x_loc_array_side.append(x_loc_side)
            x_loc_array_top.append(x_loc_top)
            
        img_array = []
        pts_array = []
        for i,n_i in enumerate(n):
            print('Image ' + str(n_i))
            #get point where you are trying to locate the theta  
            img = self.test.open_Image(n_i)
            f,ax = img.plot_Image(state='deformed', mode='e_vm',log_transform_flag=True, max_strain=0.5, ax=None)
            f.subplots_adjust(top=0.80,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)
            [ax.axvline(x,color='C'+str(ii),lw=1) for ii,x in enumerate(x_loc_array_top[i])]
            prompt = 'Left click points on the ring where the line intersects with the outer diameter.'
            pts = UI_get_pts(prompt=prompt)
            plt.close(f)
            
            img_array.append(img)
            pts_array.append(pts)

        return pts_array, x_loc_array_top, x_loc_array_side

    def get_side_img_scale(self,n_i=0,side_img_zoom=[(None,None),(None,None)]):
        f,ax = self.test.plot_side_img(n_i,side_img_zoom=side_img_zoom,side_view_col= self.side_view_col)
        #get scale from the side image
        #don't do if scale is already set
        if not hasattr(self,'side_view_scale'):
            prompt = 'Left click 2 points to define the image scale.\nDefault is the ring width'
            pts = UI_get_pts(prompt=prompt,n=2)
            pts = np.array(pts)
            
            root = tk.Tk()
            length = simpledialog.askfloat(title='Length',
                                       prompt='What is the length between the selected points in mm?',
                                       initialvalue=self.test.W)
            root.destroy()
            self.side_view_scale = np.sqrt(np.sum(np.diff(pts,axis=0)**2)) /  length  #mm per pixel 
        #get centroid of the ring
        #don't do if center is already set
        if not hasattr(self,'side_view_center'):
            prompt = 'Left click 2 points to define the left and right edges of either the ring or the mandrels.\nThis is to define the centroid of the ring'
            pts = np.array(UI_get_pts(prompt=prompt,n=2))        
            self.side_view_center=np.mean(pts[:,0])
        plt.close(f)

    def _side_image_angle_analysis(self,n,pts_array):
        angles_array = []
        for i, n_i in enumerate(n):
            print(str(i+1) +'/' +str(len(n)))
            pts = pts_array[i]
            a = 1
            img = self.test.open_Image(n_i)
            angles = []
            for pt in pts:
                angle = pi#guess
                for frac in [1,1/10,1/50]:
                    theta = np.linspace(angle-frac*pi, angle+frac*pi,12)
                    x = img.get_value(a,theta,mode='x',extrap=True) + img.get_value(a,theta,mode='u',extrap=True)
                    y = img.get_value(a,theta,mode='y',extrap=True) + img.get_value(a,theta,mode='v',extrap=True)
                    x_pixel = x*self.test.scale
                    y_pixel = y*self.test.scale      
                    angle = theta[find_nearest_idx(x_pixel,y_pixel,pt[0],pt[1])]
                    if angle >= 2*pi:
                        angle = angle-2*pi
                angles.append(angle)
            angles_array.append(angles)
        return angles_array


    def _side_image_strain_analysis(self,n,angles_array,N=50,M=10):
        if type(n) != list and type(n) != type(np.array([])):
            n = [n]
        
        extrap_points_array = []
        reg_points_array   = []
        nearest_values_array = []
        for i, n_i in enumerate(n):
            img = self.test.open_Image(n_i)
            extrap_points = []
            reg_points = []
            min_values = []
            max_values = []
            nearest_values = []
            
            for angle in angles_array[i]:
                #extrapolation 
                frac = 1/10
                theta = np.linspace(angle-frac*pi,angle+frac*pi,N)
                a = 1
                ett = img.get_value(a,theta,mode='ett',extrap=True)
                ett = np.reshape(ett, theta.shape)

                strain_fun = interp1d(np.convolve(theta, np.ones(M)/M, mode='valid'),
                                      np.convolve(ett,  np.ones(M)/M, mode='valid'),
                                      bounds_error = False,fill_value=np.nan)
                extrap_points.append(strain_fun(angle)) 
                
                a = np.linspace(0.5,1,50)
                ett = img.get_value(a,angle,mode='ett',extrap=False)
                ett = np.reshape(ett,a.shape)

                a = a[~np.isnan(ett)]
                ett = ett[~np.isnan(ett)]
                m,b,_,_,_ = linregress(a,ett)
                
                reg_point = m*1 + b
                reg_points.append(reg_point)
                
                min_values.append(min(ett))
                max_values.append(max(ett))
                nearest_values.append(ett[-1])


            extrap_points_array.append(extrap_points)
            reg_points_array.append(reg_points)
            nearest_values_array.append(nearest_values)
            print(str(i+1) +'/' +str(len(n)))        
        
        return extrap_points_array, reg_points_array, nearest_values_array

    def _save_side_image_strain(self,n,angles_array,extrap_points_array,reg_points_array,nearest_values_array,x_loc_array_top,x_loc_array_side):
        #create multiindex for dataframe
        multiindex = []
        
        max_num = [len(angles) for angles in angles_array]
        max_num = max(max_num)

        for i in range(max_num):
            multiindex.append(('angles',i))
            multiindex.append(('extrap_points',i))
            multiindex.append(('nearest_values',i))
            multiindex.append(('reg_points',i))   
            multiindex.append(('x_loc_side',i))
            multiindex.append(('x_loc_top',i))
            
        multiindex.sort()
        multiindex = pd.MultiIndex.from_tuples(multiindex) 

        try:
            #convert all lists to np arrays
            angles_array = np.array(angles_array)
            extrap_points_array = np.array(extrap_points_array)
            reg_points_array = np.array(reg_points_array)
            nearest_values_array = np.array(nearest_values_array)
            x_loc_array_side = np.array(x_loc_array_side)           
            x_loc_array_top = np.array(x_loc_array_top)
    
            #concatenate and store in dataframe
            data = np.concatenate(
                (angles_array,
                 extrap_points_array,
                 nearest_values_array,
                 reg_points_array,
                 x_loc_array_side,
                 x_loc_array_top),
                axis=1)
            
            data = pd.DataFrame(index=n,columns=multiindex,data = data)
        except:
            data = {'angles':angles_array,
                    'extrap_points':extrap_points_array,
                    'nearest_values':nearest_values_array,
                    'reg_points':reg_points_array,
                    'x_loc_side':x_loc_array_side,
                    'x_loc_top':x_loc_array_top}
        return data

    def _plot_debug_angle_analysis(self,n_i,angles,x_loc_top,x_loc_side,side_img_zoom=[(None,None),(None,None)]):
        if self.test.scale==None:
            self.test.get_geometry()
        
        self.get_side_img_scale(n_i,side_img_zoom)
        
        f,ax = self.test.plot_top_side_img(n_i,side_img_zoom=side_img_zoom,side_view_col= self.side_view_col)
        
        if True:#x_loc_array_top:
            [ax[0].axvline(x,color='C'+str(ii),lw=1) for ii,x in enumerate(x_loc_top)]
            
        if True:#x_loc_array_side:
            [ax[1].axvline(x,color='C'+str(ii),lw=0.5) for ii,x in enumerate(x_loc_side)]
            
        img = self.test.open_Image(n_i)
        for angle in angles:
            x = img.get_value(1,angle,mode='x',extrap=True) + img.get_value(1,angle,mode='u',extrap=True)
            y = img.get_value(1,angle,mode='y',extrap=True) + img.get_value(1,angle,mode='v',extrap=True)
            x_pixel = x*self.test.scale
            y_pixel = y*self.test.scale  
            ax[0].plot(x_pixel,y_pixel,'o')
        ax[0].set_title('Frame '+str(n_i))
        f.subplots_adjust(top=0.91,bottom=0.0,left=0.0,right=0.96,hspace=0.2,wspace=0.2) 
        return f,ax

    def _plot_debug_strain_analysis(self,n_i,angles,M=10,N=50,extrap_value=np.nan,reg_value=np.nan):
        img = self.test.open_Image(n_i)
        f,ax = make_figure()
        for i,angle in enumerate(angles):
            #debug for strain extrapolation
            frac = 1/10
            theta = np.linspace(angle-frac*pi,angle+frac*pi,N)
            theta_plot = np.linspace(-frac*pi,frac*pi,N)
            a = 1
            ett = img.get_value(a,theta,mode='ett',extrap=True)
            ett = np.reshape(ett, theta.shape)
            strain_fun = interp1d(np.convolve(theta, np.ones(M)/M, mode='valid'),
                                  np.convolve(ett,  np.ones(M)/M, mode='valid'),
                                  bounds_error = False,fill_value=np.nan)

            ax.plot(theta_plot,ett,color = 'C'+str(i),lw=1)
            ax.plot(theta_plot,strain_fun(theta),color = 'C'+str(i),lw=1,ls='--')
        ax.set_title('Extrapolated strain, n = '+str(n_i)+'; i= '+str(i))
        ax.set_xlabel('Angle [rad]')
        ax.set_ylabel('hoop strain [mm/mm]')
        ax.axvline(0,color='k',lw=1)
        ax.axhline(0,color='k',lw=0.5,ls='--')
        if not np.isnan(extrap_value).all():
            for value in extrap_value:
                ax.plot(0,value,'.k')
        f,ax = make_figure()
        for i,angle in enumerate(angles):
            #debug for other metrics
            a = np.linspace(0.5,1,50)
            ett = img.get_value(a,angle,mode='ett',extrap=False)
            ett = np.reshape(ett,a.shape)

            a = a[~np.isnan(ett)]
            ett = ett[~np.isnan(ett)]
                            
            m,b,_,_,_ = linregress(a,ett)
            
            f,ax = self.test.plot_strain_distribution(n_i, theta=angle, mode='ett',extrap=False, ax=ax,
                             fill=False,  fill_colors = ['#F47558', '#89D279'], 
                             color = 'C'+str(i))
            ax.set_title('Cross thickness strain, n = '+str(n_i)+'; i= '+str(i))
            ax.set_xlabel('normalized thickness [mm/mm]')
            ax.set_ylabel('hoop strain [mm/mm]')
            ax.plot(np.linspace(0,1),np.linspace(0,1)*m+b,color='k')  


class RingPullCoatingAnalysis2():
    def __init__(self, RingPull, side_view_col='side_img_file'): 
        self.test = RingPull
        self.side_view_col = side_view_col


    def get_side_image_strain(self,n,N=50,M=10, side_img_zoom=[(None,None),(None,None)],debug=False):
        # if n is not a list or array,make it so
        if type(n) != list and type(n) != type(np.array([])):
            n = [n]
        
        pts_array,y_loc_array_top, y_loc_array_side = self._side_image_angle_UI(n,side_img_zoom)

        print('starting angle analysis')
        angles_array = self._side_image_angle_analysis(n,pts_array)
        
        print('starting strain analysis')
        extrap_points_array, reg_points_array, nearest_values_array = self._side_image_strain_analysis(n,angles_array,N,M)
               
        if debug:
            print('Debugging on. Calculating and generating graphs.')
            for i, n_i in enumerate(n):
                self._plot_debug_angle_analysis(n_i,angles_array[i],y_loc_array_top[i],y_loc_array_side[i],side_img_zoom)
                self._plot_debug_strain_analysis(n_i, angles_array[i])

        print('Finished analysis. Saving data.')
        data = self._save_side_image_strain(n,angles_array,extrap_points_array,reg_points_array,nearest_values_array,y_loc_array_top,y_loc_array_side)

        return data        
        


    def _side_image_angle_UI(self,n,side_img_zoom=[(None,None),(None,None)]):
        #if get_geometry has not been run yet
        if self.test.scale == 1:
            self.test.get_geometry()


        y_loc_array_side = []
        y_loc_array_top = []
        
        self.get_side_img_scale(side_img_zoom)

        for n_i in n:
            print('Image ' + str(n_i))

            # plot the side view image
            f,ax = self.test.plot_side_img(n_i,side_img_zoom=side_img_zoom,side_view_col= self.side_view_col)       
            #get point where you are trying to locate the theta
            prompt = 'Left click points on the ring to find theta.'
            pts = UI_get_pts(prompt=prompt)
            #close side image plot   
            plt.close(f)
            pts = np.array(pts)
            y_loc_side=pts[:,1]

            side_view_distance =  (self.side_view_centroid[1] - y_loc_side) / self.side_view_scale[1] 
            y_loc_top = -side_view_distance * self.test.scale + self.test.centroid[1] # mm * pixels/mm = pixel 

            y_loc_array_side.append(y_loc_side)
            y_loc_array_top.append(y_loc_top)


        img_array = []
        pts_array = []
        for i,n_i in enumerate(n):
            print('Image ' + str(n_i))
            #get point where you are trying to locate the theta  
            img = self.test.open_Image(n_i)
            f,ax = img.plot_Image(state='deformed', mode='e_vm',log_transform_flag=True, max_strain=0.5, ax=None)
            f.subplots_adjust(top=0.80,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.2)
            [ax.axhline(y,color='C'+str(ii),lw=1) for ii,y in enumerate(y_loc_array_top[i])]
            prompt = 'Left click points on the ring where the line intersects with the outer diameter.'
            pts = UI_get_pts(prompt=prompt)
            plt.close(f)
            
            img_array.append(img)
            pts_array.append(pts)

        return pts_array, y_loc_array_top, y_loc_array_side


    def get_side_img_scale(self,side_img_zoom=[(None,None),(None,None)]):
        f,ax = self.test.plot_side_img(0,side_img_zoom=side_img_zoom,side_view_col= self.side_view_col)
        #get scale from the side image
        #don't do if scale is already set
        if (not hasattr(self,'side_view_scale')):
            prompt = 'Left click at least 5 points on the outer diameter of \nthe ring to define an ellipse.'
            pts = UI_ellipse(ax,prompt)
            #assumes the ellipse major and minor axis are aligned with the x and y axis
            self.side_view_centroid = np.mean(pts,axis=0)
            self.side_view_radii = np.max(pts-self.side_view_centroid,axis=0)
            self.side_view_scale = self.side_view_radii * 2 / self.test.OD  #pix/mm 
            

        plt.close(f)


    def _side_image_angle_analysis(self,n,pts_array):
        angles_array = []
        for i, n_i in enumerate(n):
            print(str(i+1) +'/' +str(len(n)))
            pts = pts_array[i]
            a = 1
            img = self.test.open_Image(n_i)
            angles = []
            for pt in pts:
                angle = pi#guess
                for frac in [1,1/10,1/50]:
                    theta = np.linspace(angle-frac*pi, angle+frac*pi,12)
                    x = img.get_value(a,theta,mode='x',extrap=True) + img.get_value(a,theta,mode='u',extrap=True)
                    y = img.get_value(a,theta,mode='y',extrap=True) + img.get_value(a,theta,mode='v',extrap=True)
                    x_pixel = x*self.test.scale
                    y_pixel = y*self.test.scale      
                    angle = theta[find_nearest_idx(x_pixel,y_pixel,pt[0],pt[1])]
                    if angle >= 2*pi:
                        angle = angle-2*pi
                angles.append(angle)
            angles_array.append(angles)
        return angles_array


    def _side_image_strain_analysis(self,n,angles_array,N=50,M=10):
        if type(n) != list and type(n) != type(np.array([])):
            n = [n]
        
        extrap_points_array = []
        reg_points_array   = []
        nearest_values_array = []
        for i, n_i in enumerate(n):
            img = self.test.open_Image(n_i)
            extrap_points = []
            reg_points = []
            min_values = []
            max_values = []
            nearest_values = []
            
            for angle in angles_array[i]:
                #extrapolation 
                frac = 1/10
                theta = np.linspace(angle-frac*pi,angle+frac*pi,N)
                a = 1
                ett = img.get_value(a,theta,mode='ett',extrap=True)
                ett = np.reshape(ett, theta.shape)

                strain_fun = interp1d(np.convolve(theta, np.ones(M)/M, mode='valid'),
                                      np.convolve(ett,  np.ones(M)/M, mode='valid'),
                                      bounds_error = False,fill_value=np.nan)
                extrap_points.append(strain_fun(angle)) 
                
                a = np.linspace(0.5,1,50)
                ett = img.get_value(a,angle,mode='ett',extrap=False)
                ett = np.reshape(ett,a.shape)

                a = a[~np.isnan(ett)]
                ett = ett[~np.isnan(ett)]
                m,b,_,_,_ = linregress(a,ett)
                
                reg_point = m*1 + b
                reg_points.append(reg_point)
                
                min_values.append(min(ett))
                max_values.append(max(ett))
                nearest_values.append(ett[-1])


            extrap_points_array.append(extrap_points)
            reg_points_array.append(reg_points)
            nearest_values_array.append(nearest_values)
            print(str(i+1) +'/' +str(len(n)))        
        
        return extrap_points_array, reg_points_array, nearest_values_array

    def _save_side_image_strain(self,n,angles_array,extrap_points_array,reg_points_array,nearest_values_array,y_loc_array_top,y_loc_array_side):
        #create multiindex for dataframe
        multiindex = []
        
        max_num = [len(angles) for angles in angles_array]
        max_num = max(max_num)

        for i in range(max_num):
            multiindex.append(('angles',i))
            multiindex.append(('extrap_points',i))
            multiindex.append(('nearest_values',i))
            multiindex.append(('reg_points',i))   
            multiindex.append(('y_loc_side',i))
            multiindex.append(('y_loc_top',i))
            
        multiindex.sort()
        multiindex = pd.MultiIndex.from_tuples(multiindex) 

        try:
            #convert all lists to np arrays
            angles_array = np.array(angles_array)
            extrap_points_array = np.array(extrap_points_array)
            reg_points_array = np.array(reg_points_array)
            nearest_values_array = np.array(nearest_values_array)
            y_loc_array_side = np.array(y_loc_array_side)           
            y_loc_array_top = np.array(y_loc_array_top)
    
            #concatenate and store in dataframe
            data = np.concatenate(
                (angles_array,
                 extrap_points_array,
                 nearest_values_array,
                 reg_points_array,
                 y_loc_array_side,
                 y_loc_array_top),
                axis=1)
            
            data = pd.DataFrame(index=n,columns=multiindex,data = data)
        except:
            data = {'angles':angles_array,
                    'extrap_points':extrap_points_array,
                    'nearest_values':nearest_values_array,
                    'reg_points':reg_points_array,
                    'y_loc_side':y_loc_array_side,
                    'y_loc_top':y_loc_array_top}
        return data


    def _plot_debug_angle_analysis(self,n_i,angles,y_loc_top,y_loc_side,side_img_zoom=[(None,None),(None,None)]):
        if self.test.scale==None:
            self.test.get_geometry()
        
        self.get_side_img_scale(n_i,side_img_zoom)
        
        f,ax = self.test.plot_top_side_img(n_i,side_img_zoom=side_img_zoom,side_view_col= self.side_view_col)
        
        if True:#y_loc_array_top:
            [ax[0].axhline(y,color='C'+str(ii),lw=1) for ii,y in enumerate(y_loc_top)]
            
        if True:#y_loc_array_side:
            [ax[1].axhline(y,color='C'+str(ii),lw=0.5) for ii,y in enumerate(y_loc_side)]
            
        img = self.test.open_Image(n_i)
        for angle in angles:
            x = img.get_value(1,angle,mode='x',extrap=True) + img.get_value(1,angle,mode='u',extrap=True)
            y = img.get_value(1,angle,mode='y',extrap=True) + img.get_value(1,angle,mode='v',extrap=True)
            x_pixel = x*self.test.scale
            y_pixel = y*self.test.scale  
            ax[0].plot(x_pixel,y_pixel,'o')
        ax[0].set_title('Frame '+str(n_i))
        f.subplots_adjust(top=0.91,bottom=0.0,left=0.0,right=0.96,hspace=0.2,wspace=0.2) 
        return f,ax

    def _plot_debug_strain_analysis(self,n_i,angles,M=10,N=50,extrap_value=np.nan,reg_value=np.nan):
        img = self.test.open_Image(n_i)
        f,ax = make_figure()
        for i,angle in enumerate(angles):
            #debug for strain extrapolation
            frac = 1/10
            theta = np.linspace(angle-frac*pi,angle+frac*pi,N)
            theta_plot = np.linspace(-frac*pi,frac*pi,N)
            a = 1
            ett = img.get_value(a,theta,mode='ett',extrap=True)
            ett = np.reshape(ett, theta.shape)
            strain_fun = interp1d(np.convolve(theta, np.ones(M)/M, mode='valid'),
                                  np.convolve(ett,  np.ones(M)/M, mode='valid'),
                                  bounds_error = False,fill_value=np.nan)

            ax.plot(theta_plot,ett,color = 'C'+str(i),lw=1)
            ax.plot(theta_plot,strain_fun(theta),color = 'C'+str(i),lw=1,ls='--')
        ax.set_title('Extrapolated strain, n = '+str(n_i)+'; i= '+str(i))
        ax.set_xlabel('Angle [rad]')
        ax.set_ylabel('hoop strain [mm/mm]')
        ax.axvline(0,color='k',lw=1)
        ax.axhline(0,color='k',lw=0.5,ls='--')
        if not np.isnan(extrap_value).all():
            for value in extrap_value:
                ax.plot(0,value,'.k')
        f,ax = make_figure()
        for i,angle in enumerate(angles):
            #debug for other metrics
            a = np.linspace(0.5,1,50)
            ett = img.get_value(a,angle,mode='ett',extrap=False)
            ett = np.reshape(ett,a.shape)

            a = a[~np.isnan(ett)]
            ett = ett[~np.isnan(ett)]
                            
            m,b,_,_,_ = linregress(a,ett)
            
            f,ax = self.test.plot_strain_distribution(n_i, theta=angle, mode='ett',extrap=False, ax=ax,
                             fill=False,  fill_colors = ['#F47558', '#89D279'], 
                             color = 'C'+str(i))
            ax.set_title('Cross thickness strain, n = '+str(n_i)+'; i= '+str(i))
            ax.set_xlabel('normalized thickness [mm/mm]')
            ax.set_ylabel('hoop strain [mm/mm]')
            ax.plot(np.linspace(0,1),np.linspace(0,1)*m+b,color='k')  








##################################################################

if __name__ == '__main__':  # only runs if this script is the main script
    print('Please go to example script, RPSA_example.py')
    
    










    
    
    
    
    
    
    
    
    
    
    
    
