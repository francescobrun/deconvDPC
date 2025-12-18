import numpy as np
import skimage
from scipy.signal import convolve2d
import lib_deconv
from skimage.transform import resize, radon, iradon
from scipy.signal import hilbert
import tifffile
from _sl3d import shepp_logan_3d
from scipy.ndimage import zoom, rotate
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import glob


def process_slice1(i, param):
    res = lib_deconv.deconvSps(dset_d[:, :, i], psf, param, 400)
    res_norm = lib_deconv.percentile_normalization(res, projs[:, :, i]).astype(np.float32)
    return i, res_norm

def process_and_save(param, output_path):
    process_func = partial(process_slice1, param=param)

    with ProcessPoolExecutor(max_workers=32) as executor:
        for i, res_norm in executor.map(process_func, range(dset_d.shape[2])):
            out_file = os.path.join(output_path, f"proj_{i:04d}.tif")
            tifffile.imwrite(out_file, res_norm.astype(np.float32))

def process_slice2(i, param):
    res = lib_deconv.tv_deconvolution(dset_d[:, :, i], psf, param, 400)
    res_norm = lib_deconv.percentile_normalization(res, projs[:, :, i]).astype(np.float32)
    return i, res_norm

def process_and_save2(param, output_path):
    process_func = partial(process_slice2, param=param)

    with ProcessPoolExecutor(max_workers=32) as executor:
        for i, res_norm in executor.map(process_func, range(dset_d.shape[2])):
            out_file = os.path.join(output_path, f"proj_{i:04d}.tif")
            tifffile.imwrite(out_file, res_norm.astype(np.float32))

def process_slice3(i, param):
    res = lib_deconv.wiener_deconvolution_fb(dset_d[:, :, i], psf, param)
    res_norm = lib_deconv.percentile_normalization(res, projs[:, :, i]).astype(np.float32)
    return i, res_norm

def process_and_save3(param, output_path):
    process_func = partial(process_slice3, param=param)

    with ProcessPoolExecutor(max_workers=32) as executor:
        for i, res_norm in executor.map(process_func, range(dset_d.shape[2])):
            out_file = os.path.join(output_path, f"proj_{i:04d}.tif")
            tifffile.imwrite(out_file, res_norm.astype(np.float32))





## GET PHANTOM MATRIX

nVoxelZ = 256 
nVoxelY = 256 
nVoxelX = 256 
P = shepp_logan_3d(size_out=[nVoxelZ,nVoxelY,nVoxelX], phantom_type="toft-schabel")

path="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation\\3dSL\\"
for i in range(P.shape[0]):
    tifffile.imwrite(path+ "slice_"+ str(i).zfill(4) + ".tif", P[i,:,:].astype(np.float32)) 


# SIMULATE CT

n_proj = 150
theta = np.linspace(0, 180, n_proj, endpoint=False)
nZ, nY, nX = P.shape
dset = np.zeros((nY, nX, n_proj))

for i, ang in enumerate(theta):
    P_rot = rotate(P, angle=ang, axes=(0,2), reshape=False, order=1)
    proj = np.mean(P_rot, axis=0)
    dset[:,:,i] = proj

path="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\poisson_noise\\150proj\\high_noise\\CT_simulation\\"
if not os.path.exists(path):
    os.makedirs(path)
for i in range(dset.shape[2]):
    tifffile.imwrite(path+ "slice_"+ str(i).zfill(4) + ".tif", dset[:,:,i].astype(np.float32)) 


# Control - iradon phantom

path= "D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\poisson_noise\\150proj\\phantom\\"
if not os.path.exists(path):
    os.makedirs(path)

rec_fin=np.zeros((nY, nX, nX))
    
for i in range(dset.shape[0]):   
    im=  np.pad(dset[i,:,:], ((dset.shape[1] // 2, dset.shape[1] // 2), (0, 0)), mode='edge')
    rec_i  = iradon(im, theta=theta, filter_name='ramp')
    rec_i= rec_i[128:384,128:384] 
    rec_fin[:,:,i]=rec_i

for i in range (dset.shape[0]):
    tifffile.imwrite(os.path.join(path,"slice_"+ str(i).zfill(4) + ".tif"), rec_fin[i,:,:].astype(np.float32)) 

# Prepare differential kernel:
psf = np.zeros((3, 3))
psf[1,1] = 1
psf[1,2] = -1


# DIFFERENTIAL PROJECTIONS AND POISSON NOISE ADDITION

path_diff="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\poisson_noise\\150proj\\high_noise\\differential_noise_simulation\\"
path_noise="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\poisson_noise\\150proj\\high_noise\\noise_simulation\\"
if not os.path.exists(path_diff):
    os.makedirs(path_diff)
if not os.path.exists(path_noise):
    os.makedirs(path_noise)

N= 30000   #N=65000 
dset_d = np.zeros( dset.shape )

for i in range (dset.shape[2]):
    # Attenuation projection
    att = dset[:,:,i]
    # Calculate transmission image (Lambert–Beer)
    T = np.exp(-att)
    # Photon count image
    I = T * N

    #Add Poisson noise
    I_noise = np.random.poisson(I)

    # Back to transmission
    T_noise = I_noise.astype(np.float64) / N
    # Back to attenuation
    eps = np.finfo(np.float64).eps
    T_noise[T_noise <= eps] = eps
    att_noise = -np.log(T_noise)

    tifffile.imwrite(path_noise+ "slice_"+ str(i).zfill(4) + ".tif", att_noise.astype(np.float32))

    #differential projections
    dset_d[:,:,i] = convolve2d(att_noise, psf, 'same')
    tifffile.imwrite(path_diff+ "slice_"+ str(i).zfill(4) + ".tif", dset_d[:,:,i].astype(np.float32))


# MASK

center_x, center_y = dset_d.shape[0]// 2, dset_d.shape[1]// 2  
radius = dset_d.shape[0]// 2 
y, x = np.ogrid[:dset_d.shape[0], :dset_d.shape[1]]
distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
mask = distance_from_center >= radius


# HILBERT  
path_output="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\poisson_noise\\150proj\\high_noise\\hilbert\\hilbert_simul\\"
if not os.path.exists(path_output):
    os.makedirs(path_output)
rec = np.zeros( (dset.shape[0],dset.shape[1],dset.shape[1]) )

for i in range(dset.shape[0]):      
    im = np.pad(dset_d[i,:,:], ((dset_d.shape[1] // 2, dset_d.shape[1] // 2), (0, 0)), mode='edge')
    sino_i = np.imag(hilbert(im, axis=0)) 
    sino_i = zoom(sino_i, (2,1), order=1) 
    sino_i = np.roll(sino_i, -1, axis=0)  
    
    rec_i = iradon(sino_i, theta=theta, filter_name=None) / 2.0 
    rec_i = zoom(rec_i, (0.5,0.5), order=1) 
    rec_i = rec_i[128:384, 128:384]
    rec_i[mask] = 0                       
    rec[i,:,:] = rec_i   
    tifffile.imwrite(path_output + "slice_"+ str(i).zfill(4) + ".tif", rec[i,:,:].astype(np.float32)) 


# Forward after HILBERT
path_output= "D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\poisson_noise\\150proj\\high_noise\\hilbert\\forward_after_hilbert\\"
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



if __name__ == '__main__':
    

    # SPARSE deconvolution

    dset_sparse_norm = np.zeros((dset_d.shape), dtype=np.float32)
    dset_sparse_no_norm = np.zeros((dset_d.shape), dtype=np.float32)

    path_output_norm1="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\poisson_noise\\150proj\\high_noise\\sparse\\deconv_normalized\\"
    if not os.path.exists(path_output_norm1):
        os.makedirs(path_output_norm1)

    process_and_save(0.0002, path_output_norm1)
    
        
    # TV deconvolution

    dset_tv_norm = np.zeros((dset_d.shape), dtype=np.float32)
    dset_tv_no_norm = np.zeros((dset_d.shape), dtype=np.float32)

    path_output_norm1="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\poisson_noise\\150proj\\high_noise\\TV\\deconv_normalized\\"
    if not os.path.exists(path_output_norm1):
        os.makedirs(path_output_norm1)

    process_and_save2(0.05, path_output_norm1)

    
    # WIENER deconvolution

    dset_wiener_no_norm = np.zeros((dset_d.shape),dtype=np.float32)
    dset_wiener_norm = np.zeros((dset_d.shape),dtype=np.float32)

    path_output_norm1="D:\\FromPEPItoGiada\\deconvDPC\\phantom_simulation_calibrated\\poisson_noise\\150proj\\high_noise\\wiener\\deconv_normalized\\"
    if not os.path.exists(path_output_norm1):
        os.makedirs(path_output_norm1)

    process_and_save3(0.0001, path_output_norm1)
