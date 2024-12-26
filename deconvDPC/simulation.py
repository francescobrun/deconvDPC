import numpy as np
import skimage
from scipy.signal import convolve2d
import lib_deconv
from skimage.transform import resize, radon, iradon
from scipy.signal import hilbert
import tifffile
from _sl3d import shepp_logan_3d


## Get phantom matrix:
nVoxelZ = 256 #256
nVoxelY = 256 #128
nVoxelX = 256 #64
P = shepp_logan_3d(size_out=[nVoxelZ,nVoxelY,nVoxelX], phantom_type="toft-schabel")
P= P*10

## Simulate CT:
n_proj = 300;
theta = np.linspace(0.0, 180.0, n_proj, endpoint=False)

dset = np.zeros( (nVoxelZ,nVoxelY,n_proj) )

for i in range(P.shape[0]):
    s_i = radon(P[i,:,:], theta=theta, circle=(True))
    dset[i,:,:] = s_i

# Prepare differential kernel:
psf = np.zeros((3, 3))
psf[1,1] = 1
psf[1,2] = -1

# Add gaussian noise:
mean=0
sd=1
size=(256,256)

dset_d = np.zeros( dset.shape )

for i in range(dset.shape[2]):
    gaussian_noise=np.random.normal(mean,sd,size)
    dset_d[:,:,i] = convolve2d(dset[:,:,i], psf, 'same')
    dset_d[:,:,i] = dset_d[:,:,i] + gaussian_noise

# mask
center_x, center_y = dset_d.shape[0]// 2, dset_d.shape[1]// 2  
raggio = dset_d.shape[0]// 2 
y, x = np.ogrid[:dset_d.shape[0], :dset_d.shape[1]]
distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
mask = distance_from_center >= raggio


# HILBERT reconstruction 
path_output="C:\\Users\\giarizzo.WIN\\Documents\\phaseIntegration_deconv\\hilbert_simul\\"
rec = np.zeros( (dset.shape[0],dset.shape[1],dset.shape[1]) )
for i in range(dset.shape[0]):      
    sino_i = np.imag(hilbert(dset_d[i,:,:], axis=0)) 
    rec_i  = iradon(sino_i, theta=theta, filter_name=None)
    rec_i = 10*(rec_i-np.min(rec_i))/(np.max(rec_i)-np.min(rec_i))
    rec_i[mask] = 0
    rec[i,:,:] = rec_i   
    tifffile.imwrite(path_output + "slice_"+ str(i).zfill(4) + ".tif", rec[i,:,:].astype(np.float32)) 

    
## TV deconvolution

path_output="C:\\Users\\giarizzo.WIN\\Documents\\phaseIntegration_deconv\\tv_simul_recon_norm\\"
dset_tv=np.zeros((dset_d.shape))

for i in range(dset_d.shape[2]):
    image=dset_d[:,:,i]   
    Dw_tv = lib_deconv.tv_deconvolution(image, psf, 0.005, 400) 
    dset_tv[:,:,i]=Dw_tv
    
rec=np.zeros( (dset_tv.shape[0],dset_tv.shape[1],dset_tv.shape[1]) )
for i in range(dset_tv.shape[0]):    
    rec_i  = iradon(dset_tv[i,:,:], theta=theta, filter_name='ramp')
    rec[i,:,:] = rec_i
    rec_i = 10*(rec_i-0.007)/(10-0.007)
    rec_i[mask] = 0
    tifffile.imwrite(path_output + "param_0.005\\slice_"+ str(i).zfill(4) + ".tif", rec[i,:,:].astype(np.float32))   


# SPARSE deconvolution

path_output="C:\\Users\\giarizzo.WIN\\Documents\\phaseIntegration_deconv\\sparse_simul_recon\\"
dset_sparse=np.zeros((dset_d.shape))

for i in range(dset_d.shape[2]):
    image=dset_d[:,:,i]   
    Dw_sparse = lib_deconv.deconvSps(image, psf, 0.07, 400) 
    dset_sparse[:,:,i]=Dw_sparse
    
rec=np.zeros( (dset_sparse.shape[0],dset_sparse.shape[1],dset_sparse.shape[1]) )
for i in range(dset_sparse.shape[0]):    
    rec_i  = iradon(dset_sparse[i,:,:], theta=theta, filter_name='ramp')
    rec_i = 10*(rec_i-0.007)/(10-0.007)
    rec_i[mask] = 0
    rec[i,:,:] = rec_i
    tifffile.imwrite(path_output + "param_0.04\\slice_"+ str(i).zfill(4) + ".tif", rec[i,:,:].astype(np.float32))   


# WIENER deconvolution

dset_wiener=np.zeros((dset_d.shape))
path_output="C:\\Users\\giarizzo.WIN\\Documents\\phaseIntegration_deconv\\wiener_simul_recon\\param_0.001\\"

for i in range(dset_d.shape[2]):
    image=dset_d[:,:,i]   
    Dw_wiener = lib_deconv.wiener_deconvolution_fb(image, psf, 0.0000001) 
    dset_wiener[:,:,i]=Dw_wiener
    
rec=np.zeros( (dset_wiener.shape[0],dset_wiener.shape[1],dset_wiener.shape[1]) )
for i in range(dset_wiener.shape[0]):    
    rec_i  = iradon(dset_wiener[i,:,:], theta=theta, filter_name='ramp')
    rec_i = 10*(rec_i-0.007)/(10-0.007)
    rec_i[mask] = 0
    rec[i,:,:] = rec_i
    tifffile.imwrite(path_output + "slice_"+ str(i).zfill(4) + ".tif", rec[i,:,:].astype(np.float32))   
