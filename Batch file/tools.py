# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 10:11:01 2020

@author: jkf
"""
Rsun=16*60/0.165

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from skimage.transform import warp#, SimilarityTransform
from skimage.transform import AffineTransform, SimilarityTransform#,EuclideanTransform
from skimage.measure import ransac
import scipy.fftpack as fft
from scipy.signal import savgol_filter as sf, medfilt2d, medfilt

func=SimilarityTransform#AffineTransform

###### Optical Flow Alignment Function
def align_opflow(im1org,im2org,winsize=11,step=5,r_t=2,arrow=0,mask=1,sample=20):
    """
        Align two images using optical flow with RANSAC-based parameter estimation

        Parameters:
        im1org, im2org : input images to align
        winsize : optical flow window size
        step : sampling step for feature points
        r_t : RANSAC residual threshold
        arrow : flag to visualize flow vectors
        mask : region mask for valid pixels
        sample : number of points per grid cell for sampling

        Returns:
        d : transformation parameters [scale, rotation, tx, ty]
        model : best transformation model
        flag : inlier ratio
        flow : computed optical flow field
        err : parameter estimation errors
        """
    # Preprocess and normalize images
    s1=mask.copy()
    im1=im1org.copy()
    im2=im2org.copy()
    im1=im1/np.mean(im1[s1])*10000
    im2=im2/np.mean(im2[s1])*10000

    # Compute optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(im1, im2, flow=None, pyr_scale=0.5, levels=5, winsize=winsize, iterations=10, poly_n=5, poly_sigma=1.2, flags=0)
    flow=flow*mask[:,:,np.newaxis]

    # Sample valid points with step interval
    h,w=im1.shape
    x1, y1 = np.meshgrid(np.arange(w), np.arange(h))
    x2=x1.astype('float32')+flow[:,:,0]
    y2=y1.astype('float32')+flow[:,:,1]
    x1=x1[mask][::step]
    y1=y1[mask][::step]
    x2=x2[mask][::step]
    y2=y2[mask][::step]

    # Prepare source and target coordinates
    src=(np.vstack((x1.flatten(),y1.flatten())).T)
    dst=(np.vstack((x2.flatten(),y2.flatten())).T)

    # Uniform sampling of points
    indices = sample_points_evenly(src, dst, sample, 1)
    src = src[indices]
    dst = dst[indices]
    ######
    # Calculate displacement vectors
    s = dst - src
    Dlt0=((np.abs(s[:,0])>0)*1.0 + (np.abs(s[:,1])>0)) >0
    try:
        if Dlt0.sum()>0:
            dst=dst[Dlt0]
            src=src[Dlt0]
            s=s[Dlt0]

            # RANSAC estimation of transformation parameters
            model, D= ransac((src, dst), func, min_samples=4,residual_threshold=r_t, max_trials=200) #如果要考虑旋转，可以使用这个函数。但旋转也会带来更大的累计误差。慎重使用。同时返回量是一个齐次矩阵。代码要改很多

            # Visualization of optical flow vectors
            if arrow==1:
                plt.figure()
                showim(im1)
                x=src[D,0]
                y=src[D,1]
                fx=s[D,0]
                fy=-s[D,1]
                plt.quiver(x,y,fx,fy,color='r',scale=0.2,scale_units='dots',minshaft=2)

            # Calculate estimation statistics
            flag=D.sum()/Dlt0.sum() #Percentage of effective control points
            d=[model.scale, model.rotation,model.translation[0],model.translation[1]]
            print(round(flag,3),D.sum(),d)

            # Residual analysis
            residuals = model.residuals(src[D], dst[D])
            sq = [x ** 2 for x in residuals]
            sigma2 = sum(sq) / (D.sum() - 4)


            # Directional residuals
            rv = residuals_2(d, src[D], dst[D])
            sq1 = [x ** 2 for x in rv[:, 0]]
            xsigma2 = sum(sq1) / (D.sum() - 4)

            sq2 = [x ** 2 for x in rv[:, 1]]
            ysigma2 = sum(sq2) / (D.sum() - 4)

            # Parameter uncertainty estimation
            optimal_params = d

            J = jacobian(optimal_params, src[D], dst[D])

            # Calculation of the covariance matrix
            cov_matrix = np.linalg.inv(J.T @ J)
            err = [np.sqrt(cov_matrix[0, 0] * sigma2), np.sqrt(cov_matrix[1, 1] * sigma2),
                   np.sqrt(cov_matrix[2, 2] * xsigma2), np.sqrt(cov_matrix[3, 3] * ysigma2)]
            print('err:',err)

        else:
            d=[1,0,0,0]
            model=1
            flag=0
            err=0


        return d,model,flag,flow,err
    except:
        d=[1,0,0,0]
        model=1 #
        flag=0
        err=0

    return d,model,flag,flow,err


def mode_model(data, residual_threshold=5):
    (src, dst)=data

    model_robust, inliers = ransac((src, dst), func, min_samples=20,
                                       residual_threshold=2, max_trials=100)

    return model_robust


def jacobian(params, src, dst):
    """Calculate Jacobian matrix for similarity transform parameters"""
    s, theta, tx, ty = params
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    J = np.zeros((2 * len(dst), len(params)))

    for i in range(len(src)):
        x, y = src[i]
        transformed_x = s * (np.cos(theta) * x - np.sin(theta) * y) + tx
        transformed_y = s * (np.sin(theta) * x + np.cos(theta) * y) + ty

        J[2 * i, 0] = (np.cos(theta) * x - np.sin(theta) * y)
        J[2 * i, 1] = s * (-np.sin(theta) * x - np.cos(theta) * y)
        J[2 * i, 2] = 1
        J[2 * i, 3] = 0

        J[2 * i + 1, 0] = (np.sin(theta) * x + np.cos(theta) * y)
        J[2 * i + 1, 1] = s * (np.cos(theta) * x - np.sin(theta) * y)
        J[2 * i + 1, 2] = 0
        J[2 * i + 1, 3] = 1

    return J


def transform(params, src):
    """Define the similar transformation model"""

    s, theta, tx, ty = params

    rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],
                                [-np.sin(theta), np.cos(theta)]])

    transformed = np.dot(src, rotation_matrix) * np.array([s, s]) + np.array([tx, ty])  # src[x,y]右乘旋转矩阵，所以旋转矩阵内sin的符号改变
    return transformed


def residuals_2(params, src, dst):
    """Define the residual function"""
    return (transform(params, src) - dst)


def sample_points_evenly(points, dst, grid_size=10, num=1):
    """Uniformly sample points using spatial grid partitioning

    Parameters:
    points : source coordinates
    dst : target coordinates
    grid_size : number of divisions per axis
    num : points to sample per grid

    Returns:
    indices: selected point indices"""

    # Calculate displacement vectors between matched points
    delt = dst - points
    # Determine spatial boundaries of points
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    # Create grid center points (grid_size x grid_size)
    x_grid = np.linspace(min_x, max_x, grid_size)
    y_grid = np.linspace(min_y, max_y, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    grid_centers = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

    # Calculate distance matrix between points and grid centers
    distances = cdist(points, grid_centers)

    # Assign each point to nearest grid cell
    nearest_grid_idx = np.argmin(distances, axis=1)

    # Group points by their grid cell membership
    points_per_grid = np.array([np.where(nearest_grid_idx == i)[0] for i in range(grid_size * grid_size)], dtype=object)

    sampled_indices = []
    # Process each grid cell with points
    for idx_points in points_per_grid:
        if idx_points.size > 0:
            for _ in range(num):
                # Calculate displacement magnitudes for candidates
                r = (delt[idx_points] ** 2).sum(axis=1)
                r_idx = np.abs(r - np.median(r)).argmin()
                sampled_idx = idx_points[r_idx]
                # sampled_idx = np.random.choice(idx_points)

                sampled_indices.append(sampled_idx)

    return np.array(sampled_indices)


def fitswrite(fileout, im, header=None):
    """Write to fits file"""
    from astropy.io import fits
    import os
    if os.path.exists(fileout):
        os.remove(fileout)
    if header is None:
        fits.writeto(fileout, im, output_verify='fix', overwrite=True, checksum=False)
    else:        
        fits.writeto(fileout, im, header, output_verify='fix', overwrite=True, checksum=False)


def fitsread(filein):
    """Read fits file"""
    from astropy.io import fits
    head = '  '
    hdul = fits.open(filein)

    try:
        data0 = hdul[0].data.astype(np.float32)
        head = hdul[0].header
    except:
        hdul.verify('silentfix')
        data0 = hdul[1].data
        head = hdul[1].header

    return removenan(data0), head

def removeray(im, T=0.2):
    """remove ray"""
    c = medfilt2d(im, 3)
    # d=np.abs(removenan(im/c)-1)>T
    d = np.abs(removenan(im-c)) > (T*c)
    out = c*d+(1-d)*im
    return out, d

def removenan(im, key=0):
    """
    remove NAN and INF in an image
    """
    im2 = np.copy(im)
    arr = np.isnan(im2)
    im2[arr] = key
    arr2 = np.isinf(im2)
    im2[arr2] = key

    return im2


def showim(im,k=3,cmap='gray'):
    """Show image"""
    mi = np.max([im.min(), im.mean() - k * im.std()])
    mx = np.min([im.max(), im.mean() + k * im.std()])
    if len(im.shape) == 3:
        plt.imshow(im, vmin=mi, vmax=mx)
    else:
        plt.imshow(im, vmin=mi, vmax=mx, cmap=cmap,interpolation='bicubic')

    return


def zscore2(im):
    """standardization"""
    im = (im - np.mean(im)) / im.std()
    return im

def disk(M, N, r0):
    """Draw a mask circle"""
    X, Y = np.meshgrid(np.arange(int(-(N / 2)), int(N / 2)), np.linspace(-int(M / 2), int(M / 2) - 1, M))
    r = (X) ** 2 + (Y) ** 2
    r = (r ** 0.5)
    im = r < r0
    return im

def immove2(im,dx=0,dy=0):
    """move image"""
    im2=im.copy()
    tform = SimilarityTransform(translation=(dx,dy))
    im2 = warp(im2, tform.inverse, output_shape=(im2.shape[0], im2.shape[1]),mode='constant',cval=0)
    return im2

def imshift(im,translation=[0,0]):

    """
    shift an image by pixels
    """
    translation=(np.array(translation)).astype('int')
    im1 = im.copy()
    im1 = np.roll(im1, translation[0], axis=0)
    im1 = np.roll(im1, translation[1], axis=1)
    return im1

def rmdir(path):
    """Delete Catalog"""
    import shutil  
    isExists = os.path.exists(path)

    if isExists:

        shutil.rmtree(path) 
    return     
        
def mkdir(path):
    """Create a Catalog"""
    path = path.strip()

    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if isExists:
        return False
    else:
        os.makedirs(path)
        return True        
    
   

##整数像元位移

def cc(standimage, compimage, flag=0,win=1):
        """Perform integer image translation"""
        M, N = standimage.shape
        if flag==0:
            standimage = zscore2(standimage)
            s = fft.fft2(standimage)
        else:    
            s=standimage
            
        c = zscore2(compimage)
        c = fft.fft2(c)
    
        sc = s * np.conj(c)*win
        im = np.abs(fft.fftshift(fft.ifft2(sc)))
        cor = im.max()
        if cor == 0:
            return 0, 0, 0

        M0, N0 = np.where(im == cor)
        m, n = M0[0], N0[0]

        m -= M / 2
        n -= N / 2
        # 判断图像尺寸的奇偶
        if np.mod(M, 2): m += 0.5
        if np.mod(N, 2): n += 0.5

        cor/=standimage.size
        return m, n, cor,im
    

def imnorm(im, mx=0, mi=0):
    if mx != 0 and mi != 0:
        pass
    else:
        mi, mx = np.min(im), np.max(im)

    im2 = removenan((im - mi) / (mx - mi))

    arr1 = (im2 > 1)
    im2[arr1] = 1
    arr0 = (im2 < 0)
    im2[arr0] = 0

    return im2


def all_align(im,align,mask=1,winsize=31,step=50,r_t=1,arrow=0,sample=20):
    """Parameters:
    im : ndarray (CxHxW)
        Multi-channel image stack (C=number of channels)
    align : list of tuples
        Alignment relationships between channels, e.g. [(ref1, mov1), (ref2, mov2)]
    mask : ndarray or int, optional
        Binary mask for valid regions (1=valid) or scalar for full coverage (default=1)
    winsize : int, optional
        Optical flow window size (default=31)
    step : int, optional
        Feature point sampling interval (default=50)
    r_t : float, optional
        RANSAC residual threshold (default=1)
    arrow : int, optional
        Flow visualization flag (0=off, 1=on) (default=0)
    sample : int, optional
        Points per grid for uniform sampling (default=20)"""

    channels=im.shape[0]
    tot=len(align)
    M=[]
    xy=[]
    Flag=[]
    Err=[]
    for k in range(len(align)):
        # Construct coefficient vector for this alignment pair
        L=np.zeros(channels)
        n,m=align[k]
        L[n]=1
        L[m]=-1
       
        im1=im[m].copy()
        im2=im[n].copy()

        # Estimate alignment parameters via optical flow
        d,model,flag,flow,err=align_opflow(im1,im2,winsize=winsize,step=step,r_t=r_t,arrow=arrow,mask=mask,sample=sample)
        d=removenan(d)
        d[0]=np.log10(d[0]) #比例尺对数化

        d=np.array(d)
        xy.append(d*flag)
        Err.append(err)
        M.append(L*flag)
        Flag.append(flag)


    
    xy=np.array(xy)
    Flag=np.array(Flag)

    # Form matrices for least-squares solving
    M=np.array(M)
    M[:,0]=0
    x = np.linalg.lstsq(M,xy,rcond=None )[0]

    x[:,0]=10**(x[:,0])
    for i in range(1,channels):
        tform = SimilarityTransform(scale=x[i,0], rotation=x[i,1],translation=[x[i,2],x[i,3]])
        im[i] = warp(im[i], tform, output_shape=(im[i].shape[0], im[i].shape[1]))
    x=np.concatenate((x,Err))


    im=np.array(im)
    return im,x,Flag



def center_im(im, header):
    """Move the image to the center"""

    dx_1 = header['CRPIX1'] - (header['NAXIS1'] + 1) / 2
    dy_1 = header['CRPIX2'] - (header['NAXIS2'] + 1) / 2

    im = immove2(im, -dx_1, -dy_1)

    return im

