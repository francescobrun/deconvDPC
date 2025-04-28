import numpy as np
import skimage
from scipy.signal import convolve2d
import lib_deconv
from skimage.transform import resize, radon, iradon
from scipy.signal import hilbert
import tifffile
from _sl3d import shepp_logan_3d
from scipy.ndimage import zoom
import os


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
path_output="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation\\hilbert\\hilbert_simul\\"
if not os.path.exists(path_output):
    os.makedirs(path_output)
rec = np.zeros( (dset.shape[0],dset.shape[1],dset.shape[1]) )

for i in range(dset.shape[0]):      
    im = np.pad(dset_d[i,:,:], ((dset_d.shape[1] // 2, dset_d.shape[1] // 2), (0, 0)), mode='edge')
    sino_i = np.imag(hilbert(im, axis=0)) 
    sino_i = zoom(sino_i, (2,1), order=1) 
    sino_i = np.roll(sino_i, -1, axis=0)  
    sino_i=np.pad(sino_i, ((0, 0), (dset.shape[1]/2,dset.shape[1]/2 )), mode='edge')
    rec_i = iradon(sino_i, theta=theta, filter_name=None) / 2.0 
    rec_i = zoom(rec_i, (0.5,0.5), order=1) 
    rec_i = rec_i[128:384, 128:384]
    rec_i[mask] = 0                       
    rec[i,:,:] = rec_i   
    tifffile.imwrite(path_output + "slice_"+ str(i).zfill(4) + ".tif", rec[i,:,:].astype(np.float32)) 


# Forward after HILBERT
path_output= "D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation\\hilbert\\forward_after_hilbert\\"
if not os.path.exists(path_output):
    os.makedirs(path_output)

projs = np.zeros( (dset.shape[0],dset.shape[1],n_proj), dtype=np.float32 )
for i in range(dset.shape[0]):
    rec_i = rec[i,:,:]
    sino_i  = radon(rec_i, theta=theta)  # Forward projection
    sino_i[ sino_i < 0 ] = 0 # non-negativity
    projs[i,:,:] = sino_i
    
for i in range(projs.shape[2]):      
    tifffile.imwrite(path_output+"proj_"+ str(i).zfill(4) + ".tif", projs[:,:,i].astype(np.float32)) 


## TV deconvolution

dset_tv_norm = np.zeros((dset_d.shape), dtype=np.float32)
dset_tv_no_norm = np.zeros((dset_d.shape), dtype=np.float32)

path_output_norm="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\new_image\\300_proj\\sigma_1\\TV\\deconv_normalized\\"
if not os.path.exists(path_output_norm):
    os.makedirs(path_output_norm)

for i in range(dset_d.shape[2]):
    dset_tv_no_norm[:,:,i] = lib_deconv.tv_deconvolution(dset_d[:,:,i], psf, 0.025, 400) 

    # Normalize according to the corresponding projection after hilbert:
    dset_tv_norm[:,:,i] = lib_deconv.percentile_normalization(dset_tv_no_norm[:,:,i], projs[:,:,i]).astype(np.float32)
    tifffile.imwrite(os.path.join(path_output_norm,"proj_"+ str(i).zfill(4) + ".tif"), dset_tv_norm[:,:,i].astype(np.float32))

path_output_norm= "D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\new_image\\300_proj\\sigma_1\\TV\\TV_reconstruction\\"
if not os.path.exists(path_output_norm):
    os.makedirs(path_output_norm)
    
for i in range(rec.shape[0]):   
    im=  np.pad(dset_tv_norm[i,:,:], ((dset_d.shape[1] // 2, dset_d.shape[1] // 2), (0, 0)), mode='edge')
    rec_i  = iradon(im, theta=theta, filter_name='ramp')
    rec_i= rec_i[128:384,128:384]
    rec_i[mask] = 0
    tifffile.imwrite(os.path.join(path_output_norm,"slice_"+ str(i).zfill(4) + ".tif"), rec_i.astype(np.float32))  


# SPARSE deconvolution

dset_sparse_norm = np.zeros((dset_d.shape), dtype=np.float32)
dset_sparse_no_norm = np.zeros((dset_d.shape), dtype=np.float32)

path_output_norm="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\new_image\\300_proj\\sigma_1\\sparse\\deconv_normalized\\"
if not os.path.exists(path_output_norm):
    os.makedirs(path_output_norm)

for i in range(dset_d.shape[2]):
    dset_sparse_no_norm[:,:,i] = lib_deconv.deconvSps(dset_d[:,:,i], psf, 0.2, 400) 
    
    # Normalize according to the corresponding projection after hilbert:
    dset_sparse_norm[:,:,i] = lib_deconv.percentile_normalization(dset_sparse_no_norm[:,:,i], projs[:,:,i]).astype(np.float32)
    tifffile.imwrite(os.path.join(path_output_norm,"proj_"+ str(i).zfill(4) + ".tif"), dset_sparse_norm[:,:,i].astype(np.float32))

path_output_norm= "D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\new_image\\300_proj\\sigma_1\\sparse\\sparse_reconstruction\\"
if not os.path.exists(path_output_norm):
    os.makedirs(path_output_norm)
    
for i in range(rec.shape[0]):   
    im=  np.pad(dset_sparse_norm[i,:,:], ((dset_d.shape[1] // 2, dset_d.shape[1] // 2), (0, 0)), mode='edge')
    rec_i  = iradon(im, theta=theta, filter_name='ramp')
    rec_i= rec_i[128:384,128:384]
    rec_i[mask] = 0
    tifffile.imwrite(os.path.join(path_output_norm,"slice_"+ str(i).zfill(4) + ".tif"), rec_i.astype(np.float32))  


# WIENER deconvolution

dset_wiener_no_norm = np.zeros((dset_d.shape),dtype=np.float32)
dset_wiener_norm = np.zeros((dset_d.shape),dtype=np.float32)

path_output_norm="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\new_image\\300_proj\\sigma_1\\wiener\\deconv_normalized\\"
if not os.path.exists(path_output_norm):
    os.makedirs(path_output_norm)

for i in range(dset_d.shape[2]):

    dset_wiener_no_norm[:,:,i] = lib_deconv.wiener_deconvolution_fb(dset_d[:,:,i], psf, 0.0001).astype(np.float32)
    
    # Normalize according to the corresponding projection after hilbert:
    dset_wiener_norm[:,:,i] = lib_deconv.percentile_normalization(dset_wiener_no_norm[:,:,i], projs[:,:,i]).astype(np.float32)
    tifffile.imwrite(os.path.join(path_output_norm,"proj_"+ str(i).zfill(4) + ".tif"), dset_wiener_norm[:,:,i].astype(np.float32))

path_output_norm= "D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\new_image\\300_proj\\sigma_1\\wiener\\wiener_reconstruction\\"
if not os.path.exists(path_output_norm):
    os.makedirs(path_output_norm)
    
for i in range(rec.shape[0]):   
    im=  np.pad(dset_wiener_norm[i,:,:], ((dset_d.shape[1] // 2, dset_d.shape[1] // 2), (0, 0)), mode='edge')
    rec_i  = iradon(im, theta=theta, filter_name='ramp')
    rec_i= rec_i[128:384,128:384]
    rec_i[mask] = 0
    tifffile.imwrite(os.path.join(path_output_norm,"slice_"+ str(i).zfill(4) + ".tif"), rec_i.astype(np.float32))   
