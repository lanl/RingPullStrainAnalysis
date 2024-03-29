'''
Ring Pull Coating Analysis (RPCA)

RPCA is intended as an add-on to the Ring Pull Strain Analysis (RPSA) module.
It allows analysis of the failure strain of coating layers on rings that were 
tested via the gaugeless ring pull technique. RPCA consists of a module which 
takes an input the RingPull class defined in RPSA. This module includes 
functions for correlating points located on side views of the ring with DIC 
strain maps taken from the top-down view.

v1.1

Created by:
    Peter Beck
    pmbeck@lanl.gov
Updated:
    22-Nov-2023

General notes:
    - both the RPSA and the RPCA scripts should be placed in the same folder.
        This module will not work otherwise.
    - The classes save the results in the form of a pandas DataFrame that can 
        be converted into a csv file
    - Debugging methods are available based on the saved dataframe for manual 
        validation of accuracy.
    - this module was created with the following settings:
        - Windows 10
        - Python 3.8.8 (installed via Anaconda)
        - Spyder 4, with the setting:
            - IPython console -> Graphics -> Backend: Automatic

'''


from RingPullStrainAnalysis import *

class RingPullCoatingAnalysis():
    def __init__(self, RingPull_test, side_view_col='side_img_file', mode='tension'): 
        self.test = RingPull_test
        self.side_view_col = side_view_col
        if mode !='tension' and mode !='compression':
            print(mode !='tension' and mode !='compression')
            assert False, "The mode needs to be either 'tension' or 'compression'"
        self.mode = mode

    def get_side_image_strain(self,n,N=50,M=10, side_img_zoom=[(None,None),(None,None)],debug=False):
        # if n is not a list or array,make it so
        if type(n) != list and type(n) != type(np.array([])):
            n = [n]
        
        pts_array,loc_array_top, loc_array_side = self._side_image_angle_UI(n,side_img_zoom)
        
        print('starting angle analysis')
        angles_array = self._side_image_angle_analysis(n,pts_array)
        self.angles_array = angles_array
        print('starting strain analysis')
        extrap_points_array, reg_points_array, nearest_values_array = self._side_image_strain_analysis(n,angles_array,N,M)
        
        if debug:
            print('Debugging on. Calculating and generating graphs.')
            for i, n_i in enumerate(n):
                self._plot_debug_angle_analysis(n_i,angles_array[i],loc_array_top[i],loc_array_side[i],side_img_zoom)
                self._plot_debug_strain_analysis(n_i, angles_array[i])
                print(f'{i+1}/{len(n)}')

        print('Finished analysis. Outputting data.')
        data = self._save_side_image_strain(n,angles_array,extrap_points_array,reg_points_array,nearest_values_array,loc_array_top,loc_array_side)

        return data

    def _side_image_angle_UI(self,n,side_img_zoom=[(None,None),(None,None)]):
        #if get_geometry has not been run yet
        if self.test.scale == 1:
            self.test.get_geometry()
        
        loc_array_side = []
        loc_array_top = []
        
        self.get_side_view_scale(side_img_zoom)
        
        for i,n_i in enumerate(n):
            print(f'Image {n_i}')
            # plot the side view image
            if i>0:
                if i ==1:
                    if self.mode == 'tension':
                        f = plt.figure(figsize=(6.5,4))
                        gs = GridSpec(1,2,wspace=.01,hspace=.01)
                        ax1 = f.add_subplot(gs[0,0])
                        ax2 = f.add_subplot(gs[0,1])
                        prompt = 'Left click points on the ring (right) to find theta.' 
                    elif self.mode=='compression':
                        f = plt.figure(figsize=(6,4.5))
                        gs = GridSpec(2,1,wspace=.01,hspace=.01)
                        ax1 = f.add_subplot(gs[0,0])
                        ax2 = f.add_subplot(gs[1,0])
                        prompt = 'Left click points on the ring (bottom) to find theta.' 
                else:
                    ax1.images[0].remove()
                    ax2.images[0].remove()
  
                        
                f,ax1 = self.test.plot_side_img(n[i-1],ax = ax1, side_img_zoom=side_img_zoom,side_view_col= self.side_view_col)     
                f,ax2 = self.test.plot_side_img(n_i,ax = ax2, side_img_zoom=side_img_zoom,side_view_col= self.side_view_col)
                f.subplots_adjust(top=0.99,bottom=0.01,left=0.01,right=0.99,hspace=0.2,wspace=0.2)  
            else:
                f,ax = self.test.plot_side_img(n_i,side_img_zoom=side_img_zoom,side_view_col= self.side_view_col)  
                prompt = 'Left click points on the ring to find theta.'         

            #get point where you are trying to locate the theta
            pts = UI_get_pts(prompt=prompt)
            #close side image plot   
            # plt.close(f)
            pts = np.array(pts)
            
            if self.mode =='tension':
                loc_side=pts[:,1]
                side_view_distance =  (loc_side - self.side_view_centroid[1]) / self.side_view_scale[1]
                loc_top = side_view_distance * self.test.scale + self.test.centroid[1] # mm * pixels/mm = pixel   
            elif self.mode == 'compression':
                loc_side=pts[:,0]
                side_view_distance =  (loc_side - self.side_view_centroid[0] ) / self.side_view_scale[0]
                loc_top = -side_view_distance * self.test.scale+self.test.centroid[0] # mm * pixels/mm = pixel  

            loc_array_side.append(loc_side)
            loc_array_top.append(loc_top)
            
        img_array = []
        pts_array = []
        for i,n_i in enumerate(n):
            print(f'Image {n_i}')
            
            if i==0:
                f,ax = make_img_figure()
            #get point where you are trying to locate the theta  
            img = self.test.open_Image(n_i)
            f,ax = img.plot_Image(state='deformed', mode='e_vm',log_transform_flag=True, max_strain=0.5, ax=ax)
            if self.mode =='tension':
                [ax.axhline(y,color=f'C{ii}',lw=1) for ii,y in enumerate(loc_array_top[i])]
            elif self.mode == 'compression':
                [ax.axvline(x,color=f'C{ii}',lw=1) for ii,x in enumerate(loc_array_top[i])]         
            prompt = f'Left click {len(loc_array_top[i])} points on the ring where the line intersects with the outer diameter.'
            pts = UI_get_pts(n=len(loc_array_top[i]), prompt=prompt)

            ax.images[0].remove()
            [line.remove() for line in ax.lines[::-1]]
            ax.collections[0].colorbar.remove()
            ax.collections[0].remove()
            
            img_array.append(img)
            pts_array.append(pts)

        return pts_array, loc_array_top, loc_array_side

    def get_side_view_scale(self,n,side_img_zoom=[(None,None),(None,None)]):
        #get scale from the side image
        #don't do if scale is already set
        if (not hasattr(self,'side_view_scale')):
            if self.mode == 'tension':
                f,ax = self.test.plot_side_img(0,side_img_zoom=side_img_zoom,side_view_col= self.side_view_col) 
                prompt = 'Left click at least 5 points on the outer diameter of \nthe ring to define an ellipse.'
                pts = UI_ellipse(ax,prompt)
                #assumes the ellipse major and minor axis are aligned with the x and y axis
                self.side_view_centroid = np.mean(pts,axis=0)
                side_view_radii = np.max(pts-self.side_view_centroid,axis=0)
                self.side_view_scale = side_view_radii * 2 / self.test.OD  #pix/mm 
                plt.close(f)
            elif self.mode == 'compression':
                #get scale of the ring
                f,ax = self.test.plot_side_img(0,side_img_zoom=side_img_zoom,side_view_col= self.side_view_col) 
                prompt = 'Left click 2 points to define the image scale.\nDefault is the ring width'
                pts = UI_get_pts(prompt=prompt,n=2)
                pts = np.array(pts)
                
                root = tk.Tk()
                length = simpledialog.askfloat(title='Length',
                                               prompt='What is the length between the selected points in mm?',
                                               initialvalue=self.test.W)
                root.destroy()
                self.side_view_scale = np.array([np.sqrt(np.sum(np.diff(pts,axis=0)**2)) / length, 1])  #mm per pixel 
                
                #get centroid of the ring
                prompt = 'Left click 2 points to define the left and right edges of either the ring or the mandrels.\nThis is to define the centroid of the ring'
                pts = UI_get_pts(prompt=prompt,n=2)
                pts = np.array(pts)        
                self.side_view_centroid=np.array([np.mean(pts[:,0]),1])             
                plt.close(f)
                
                
    def _side_image_angle_analysis(self,n,pts_array):
        if self.test.centroid[0] == 0 and self.test.centroid[1] == 0:
            self.test.get_geometry()
        
        angles_array = []
        for i, n_i in enumerate(n):
            print(f'{i+1}/{len(n)}')
            pts = pts_array[i]
            a = 1
            img = self.test.open_Image(n_i)
            angles = []
            for pt in pts:
                angle = pi#guess
                for frac in [1,1/4,1/19,1/94,1/469]:
                    theta = np.linspace(angle-frac*pi, angle+frac*pi,10)
                    x = img.get_value(a,theta,mode='x_def',extrap=True)
                    y = img.get_value(a,theta,mode='y_def',extrap=True)
                    x_pixel = x*self.test.scale
                    y_pixel = y*self.test.scale   
                    idx = find_nearest_idx(x_pixel-self.test.centroid[0],
                                           y_pixel-self.test.centroid[1],
                                           pt[0]-self.test.centroid[0],
                                           pt[1]-self.test.centroid[1])
                    angle = theta[idx]
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
                try:
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
                    
                    reg_point = 1 * m + b
                    reg_points.append(reg_point)
                    
                    min_values.append(min(ett))
                    max_values.append(max(ett))
                    nearest_values.append(ett[-1])
                except:#if bad DIC data
                    reg_points.append(np.nan)
                    
                    min_values.append(np.nan)
                    max_values.append(np.nan)
                    nearest_values.append(np.nan)
                    

            extrap_points_array.append(extrap_points)
            reg_points_array.append(reg_points)
            nearest_values_array.append(nearest_values)
            print(f'{i+1}/{len(n)}')        
        
        return extrap_points_array, reg_points_array, nearest_values_array


    def _side_image_strain_analysis2(self,n,angles_array):
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
                try:
                    a = 1
                    strain_value = img.get_value(a,angle,mode='ett',extrap='extrap',smoothing = True)
                    extrap_points.append(strain_value)
                    
                    reg_point = img.get_value(a,angle,mode='ett',extrap='lin_regress',smoothing = False)
                    extrap_points.append(reg_point)                    
                    
                    nearest_value = img.get_value(a,angle,mode='ett',extrap='nearest',smoothing = False)
                    extrap_points.append(nearest_value)                    
                    
                    ett = img.get_value(a,theta,mode='ett',extrap=False)
                    min_values.append(np.nanmin(ett))
                    max_values.append(np.nanmax(ett))  
                    
                except:#if bad DIC data
                    reg_points.append(np.nan)
                    min_values.append(np.nan)
                    max_values.append(np.nan)
                    nearest_values.append(np.nan)

            extrap_points_array.append(extrap_points)
            reg_points_array.append(reg_points)
            nearest_values_array.append(nearest_values)
            print(f'{i+1}/{len(n)}')        
        
        return extrap_points_array, reg_points_array, nearest_values_array

    def _plot_debug_angle_analysis(self,n_i,angles,loc_top,loc_side,side_img_zoom=[(None,None),(None,None)]):
        if self.test.scale==1:
            self.test.get_geometry()
        
        self.get_side_view_scale(n_i,side_img_zoom)
        
        if self.mode == 'tension':
            f = plt.figure(figsize=(6.5,3))
            gs = GridSpec(1,2,wspace=.01,hspace=1)
            ax1 = f.add_subplot(gs[0,0])
            ax2 = f.add_subplot(gs[0,1])
        elif self.mode=='compression':
            f = plt.figure(figsize=(4.5,4.5))
            f.subplots_adjust(top=0.93,bottom=0.00,left=0.00,right=0.925,hspace=0.2,wspace=0.2)
            gs = GridSpec(8,5,wspace=0.0010,hspace=0.100)
            ax1 = f.add_subplot(gs[0:6,0:5])
            ax2 = f.add_subplot(gs[6:8,0:5])
            
        self.test.open_Image(n_i).plot_Image(ax=ax1,max_strain = 0.5) 
        ax1.collections[0].colorbar.remove()
        
        self.test.plot_side_img(n_i,ax2,self.side_view_col,side_img_zoom)

        if self.mode=='tension':
            [ax1.axhline(y,color='C'+str(ii),lw=1) for ii,y in enumerate(loc_top)]
            [ax2.axhline(y,color='C'+str(ii),lw=0.5) for ii,y in enumerate(loc_side)]            
        elif self.mode=='compression':
            [ax1.axvline(x,color='C'+str(ii),lw=1) for ii,x in enumerate(loc_top)]
            [ax2.axvline(x,color='C'+str(ii),lw=0.5) for ii,x in enumerate(loc_side)]
            
            
        img = self.test.open_Image(n_i)
        for angle in angles:
            x = img.get_value(1,angle,mode='x',extrap=True) + img.get_value(1,angle,mode='u',extrap=True)
            y = img.get_value(1,angle,mode='y',extrap=True) + img.get_value(1,angle,mode='v',extrap=True)
            x_pixel = x*self.test.scale
            y_pixel = y*self.test.scale
            
            ax1.plot(x_pixel,y_pixel,'o',ms=5)
        ax1.set_title('Frame '+str(n_i))
        f.subplots_adjust(top=0.91,bottom=0.0,left=0.0,right=0.96,hspace=0.2,wspace=0.2) 
        return f,[ax1,ax2]

    def _plot_debug_strain_analysis(self,n_i,angles,M=10,N=50,extrap_value=np.nan,reg_value=np.nan):
        img = self.test.open_Image(n_i)
        f,ax = make_figure()
        for i,angle in enumerate(angles):
            try:
                #debug for strain extrapolation
                frac = 1/10
                theta = np.linspace(angle-frac*pi,angle+frac*pi,N)
                theta_plot = np.linspace(-frac*pi,frac*pi,N)
                a = np.ones(theta.shape)
                ett = img.get_value(a,theta,mode='ett',extrap=True)
                ett = np.reshape(ett, theta.shape)
                strain_fun = interp1d(np.convolve(theta, np.ones(M)/M, mode='valid'),
                                      np.convolve(ett,  np.ones(M)/M, mode='valid'),
                                      bounds_error = False,fill_value=np.nan)
    
                ax.plot(theta_plot,ett,color = 'C'+str(i),lw=1)
                ax.plot(theta_plot,strain_fun(theta),color = 'C'+str(i),lw=1,ls='--')
                ax.plot(0,strain_fun(angle),'o',color = 'C'+str(i),ms=5)
                ax.set_ylim(-.07,0.07)
            except:
                pass
            
        ax.set_title(f'Extrapolated strain, n = {n_i}')
        ax.set_xlabel('Deviation from Angle [rad]')
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
            ett = img.get_value(a,angle*np.ones(a.shape),mode='ett',extrap=False)
            ett = np.reshape(ett,a.shape)
            
            a = a[~np.isnan(ett)]
            ett = ett[~np.isnan(ett)]
            try:
                m,b,_,_,_ = linregress(a,ett)
                
                a, e = img.get_strain_distribution(theta=angle, mode='ett', extrap=False)
                ax.plot(a,e,color = 'C'+str(i))
                ax.plot(np.linspace(0,1),np.linspace(0,1)*m+b,color = 'C'+str(i),lw=0.5,ls='--') 
                ax.set_ylim(-.07,0.07)
            except:
                pass
        
        ax.axhline(0,ls='--',color='k',lw=0.3)
        ax.set_title(f'Cross thickness strain, n = {n_i}')
        ax.set_xlabel('Normalized Thickness [mm/mm]')
        ax.set_ylabel('hoop strain [mm/mm]')
        ax.set_xlim(0,1)

    def _save_side_image_strain(self,n,angles_array,extrap_points_array,reg_points_array,nearest_values_array,loc_array_top,loc_array_side):
        try:
            df_list = []
            columns = ['n','angles','extrap_points','nearest_values','reg_points','loc_side','loc_top']
            for i,_ in enumerate(angles_array):
                angles = angles_array[i]
                n_list = n[i]* np.ones(len(angles))
                extrap_points = extrap_points_array[i]
                nearest_values = nearest_values_array[i]
                reg_points = reg_points_array[i]    
                loc_side = loc_array_side[i]
                loc_top = loc_array_top[i]  
                
                df = pd.DataFrame([n_list,angles,extrap_points,nearest_values,reg_points,loc_side,loc_top]).transpose()
                df.columns = columns
                df_list.append(df)
            data = pd.concat(df_list)
        except:
            print('Failed to return data as a DataFrame. Returning as a dictionary instead')
            data = {'angles':angles_array,
                    'extrap_points':extrap_points_array,
                    'nearest_values':nearest_values_array,
                    'reg_points':reg_points_array,
                    'loc_side':loc_array_side,
                    'loc_top':loc_array_top}

        return data

##################################################################

if __name__ == '__main__':  # only runs if this script is the main script
    print('Please go to example script, RPSA_example.py')
    
    

    





















































    
    
    
    
    
    
    
    
    
