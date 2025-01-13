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
path="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation\\3dSL\\"
for i in range(P.shape[0]):
    tifffile.imwrite(path+ "slice_"+ str(i).zfill(4) + ".tif", P[i,:,:].astype(np.float32)) 

## Simulate CT:
n_proj = 300;
theta = np.linspace(0.0, 180.0, n_proj, endpoint=False)

dset = np.zeros( (nVoxelZ,nVoxelY,n_proj) )

for i in range(P.shape[0]):
    s_i = radon(P[i,:,:], theta=theta, circle=(True))
    dset[i,:,:] = s_i

path="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation\\CT_simulation\\"
for i in range(dset.shape[2]):
    tifffile.imwrite(path+ "slice_"+ str(i).zfill(4) + ".tif", dset[:,:,i].astype(np.float32)) 

# Prepare differential kernel:
psf = np.zeros((3, 3))
psf[1,1] = 1
psf[1,2] = -1

path_diff="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation\\differential_simulation\\"
path_noise="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation\\differential_noise_simulation\\"

# Add gaussian noise:
mean=0
sd=1
size=(256,256)

dset_d = np.zeros( dset.shape )

for i in range(dset.shape[2]):
    gaussian_noise=np.random.normal(mean,sd,size)
    dset_d[:,:,i] = convolve2d(dset[:,:,i], psf, 'same')
    tifffile.imwrite(path_diff+ "slice_"+ str(i).zfill(4) + ".tif", dset_d[:,:,i].astype(np.float32))
    dset_d[:,:,i] = dset_d[:,:,i] + gaussian_noise
    tifffile.imwrite(path_noise+ "slice_"+ str(i).zfill(4) + ".tif", dset_d[:,:,i].astype(np.float32))

# mask
center_x, center_y = dset_d.shape[0]// 2, dset_d.shape[1]// 2  
radius = dset_d.shape[0]// 2 
y, x = np.ogrid[:dset_d.shape[0], :dset_d.shape[1]]
distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
mask = distance_from_center >= radius


# HILBERT reconstruction 
path_output="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation\\hilbert_simul\\"
rec = np.zeros( (dset.shape[0],dset.shape[1],dset.shape[1]) )
for i in range(dset.shape[0]):      
    sino_i = np.imag(hilbert(dset_d[i,:,:], axis=0)) 
    rec_i  = iradon(sino_i, theta=theta, filter_name=None)
    rec[i,:,:] = rec_i   

rec_complete=rec[127,:,:]
minimum= np.mean(rec_complete[111:121,89:105])
maximum= np.mean(rec_complete[237:244,118:138])

for i in range(dset.shape[0]):
    rec[i,:,:] = (rec[i,:,:]-minimum)/(maximum-minimum)
    rec_i=rec[i,:,:]
    rec_i[mask] = 0
    tifffile.imwrite(path_output + "slice_"+ str(i).zfill(4) + ".tif", rec_i.astype(np.float32)) 

    
## TV deconvolution

path_output="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation\\tv_simul\\param_0.09\\"
dset_tv=np.zeros((dset_d.shape))

for i in range(dset_d.shape[2]):
    image=dset_d[:,:,i]   
    Dw_tv = lib_deconv.tv_deconvolution(image, psf, 0.09, 400) 
    dset_tv[:,:,i]=Dw_tv
    
rec=np.zeros( (dset_tv.shape[0],dset_tv.shape[1],dset_tv.shape[1]) )
for i in range(dset_tv.shape[0]):    
    rec_i  = iradon(dset_tv[i,:,:], theta=theta, filter_name='ramp')
    rec[i,:,:] = rec_i
 
rec_i=rec[127,:,:]
minimum= np.mean(rec_i[111:121,89:105])
maximum= np.mean(rec_i[237:244,118:138])

for i in range(rec.shape[0]):
    rec[i,:,:]= (rec[i,:,:]-minimum)/(maximum-minimum)
    rec[i,:,:][mask] = 0
    tifffile.imwrite(path_output + "slice_"+ str(i).zfill(4) + ".tif", rec[i,:,:].astype(np.float32)) 


# SPARSE deconvolution

path_output="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation\\sparse_simul\\param_0.22\\"
dset_sparse=np.zeros((dset_d.shape))

for i in range(dset_d.shape[2]):
    image=dset_d[:,:,i]   
    Dw_sparse = lib_deconv.deconvSps(image, psf, 0.22, 400) 
    dset_sparse[:,:,i]=Dw_sparse
    
rec=np.zeros( (dset_sparse.shape[0],dset_sparse.shape[1],dset_sparse.shape[1]) )
for i in range(dset_sparse.shape[0]):    
    rec_i  = iradon(dset_sparse[i,:,:], theta=theta, filter_name='ramp')
    rec[i,:,:] = rec_i 

rec_i=rec[127,:,:]
minimum= np.mean(rec_i[111:121,89:105])
maximum= np.mean(rec_i[237:244,118:138])

for i in range(rec.shape[0]):
    rec[i,:,:]= (rec[i,:,:]-minimum)/(maximum-minimum)
    rec[i,:,:][mask] = 0
    tifffile.imwrite(path_output + "slice_"+ str(i).zfill(4) + ".tif", rec[i,:,:].astype(np.float32)) 


# WIENER deconvolution

dset_wiener=np.zeros((dset_d.shape))
path_output="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation\\wiener_simul\\param_0.00001\\"

for i in range(dset_d.shape[2]):
    image=dset_d[:,:,i]   
    Dw_wiener = lib_deconv.wiener_deconvolution_fb(image, psf, 0.00001) 
    dset_wiener[:,:,i]=Dw_wiener
    
rec=np.zeros( (dset_wiener.shape[0],dset_wiener.shape[1],dset_wiener.shape[1]) )
for i in range(dset_wiener.shape[0]):    
    rec_i  = iradon(dset_wiener[i,:,:], theta=theta, filter_name='ramp')
    rec[i,:,:] = rec_i

rec_i=rec[127,:,:]
minimum= np.mean(rec_i[111:121,89:105])
maximum= np.mean(rec_i[237:244,118:138])

for i in range(rec.shape[0]):
    rec[i,:,:]= (rec[i,:,:]-minimum)/(maximum-minimum)
    rec[i,:,:][mask] = 0
    tifffile.imwrite(path_output + "slice_"+ str(i).zfill(4) + ".tif", rec[i,:,:].astype(np.float32)) 
