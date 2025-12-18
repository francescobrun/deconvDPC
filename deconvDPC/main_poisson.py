import numpy as np
import tifffile
import glob
import lib_deconv
from scipy.signal import hilbert, medfilt2d
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def process_slice1(i, param):
    res = lib_deconv.deconvSps(projections2[:,:,i], psf, param, 400)
    res = res*(-1)
    return i, res

def process_and_save(param, output_path):
    process_func = partial(process_slice1, param=param)

    with ProcessPoolExecutor(max_workers=16) as executor:
        for i, res in executor.map(process_func, range(projections2.shape[2])):
            out_file = os.path.join(output_path, f"proj_{i:04d}.tif")
            tifffile.imwrite(out_file, res.astype(np.float32))

def process_slice2(i, param):
    res = lib_deconv.tv_deconvolution(projections2[:,:,i], psf, param, 400)
    res = res*(-1)
    return i, res

def process_and_save2(param, output_path):
    process_func = partial(process_slice2, param=param)

    with ProcessPoolExecutor(max_workers=16) as executor:
        for i, res in executor.map(process_func, range(projections2.shape[2])):
            out_file = os.path.join(output_path, f"proj_{i:04d}.tif")
            tifffile.imwrite(out_file, res.astype(np.float32))




# LOAD DATA

n_proj = 720;

crop_left = 380
crop_right = 165
crop_top= 9
crop_bottom = 140


path_input= "D:\\FromPEPItoGiada\\raw\\wire_flour_coffee_salt_fit_4paramsON_1shot\\"
path_projection2= "D:\\FromPEPItoGiada\\deconvDPC\\experimental_data\\1shot\\projections\\"

if not os.path.exists(path_projection2):
    os.makedirs(path_projection2)


image1=tifffile.imread("D:\\FromPEPItoGiada\\raw\\wire_flour_coffee_salt_fit_4paramsON_1shot\\REF_0001.tif")
image1 = image1[crop_top:image1.shape[0]-crop_bottom+1,crop_left:image1.shape[1]-crop_right+1]

projections=np.zeros((image1.shape[0],image1.shape[1],n_proj))
im_dir = glob.glob(f"{path_input}*REF_*")

for i in range(projections.shape[2]):
    im=tifffile.imread(im_dir[i])
    im = im[crop_top:im.shape[0]-crop_bottom+1,crop_left:im.shape[1]-crop_right+1]
    projections[:,:,i]=im



# REGULARIZATION DATA

projections2=np.zeros((image1.shape[0],image1.shape[1],n_proj))
for i in range (projections.shape[2]):
    corr=np.mean(projections[:,:,i],axis=1)
    corr=np.tile(corr,(np.size(projections[:,:,i],1),1)).T
    projections2[:,:,i]=projections[:,:,i]-corr
    projections2[:,:,i]=lib_deconv._regularization(projections[:,:,i])
    tifffile.imwrite(path_projection2 + "proj_"+ str(i).zfill(4) + ".tif", projections2[:,:,i].astype(np.float32))


#point spread function

psf=np.zeros((3,3))
psf[1,1]=1
psf[1,2]=-1



# HILBERT

path_output="D:\\FromPEPItoGiada\\deconvDPC\\experimental_data\\1shot\\hilbert\\hilbert_projections\\"
if not os.path.exists(path_output):
    os.makedirs(path_output)
hilbert_dset = np.zeros( (projections2.shape[0],projections2.shape[1],projections2.shape[2]) )

for k in range(projections2.shape[0]): 
    sino_k = np.imag(hilbert(projections2[k,:,:], axis=0))
    hilbert_dset[k,:,:]=sino_k
  
for k in range(hilbert_dset.shape[2]):
    tifffile.imwrite(path_output + "slice_"+ str(k).zfill(4) + ".tif", hilbert_dset[:,:,k].astype(np.float32))



# WIENER deconvolution

path_output="D:\\FromPEPItoGiada\\deconvDPC\\experimental_data\\1shot\\wiener\\wiener_deconvolution\\"
if not os.path.exists(path_output):
    os.makedirs(path_output)
wiener_dset=lib_deconv._wiener_deconvolution(projections2)

for i in range(wiener_dset.shape[2]):
    tifffile.imwrite(path_output + "slice_"+ str(i).zfill(4) + ".tif", wiener_dset[:,:,i].astype(np.float32))



if __name__ == '__main__':
    

    # SPARSE deconvolution

    path_output1="D:\\FromPEPItoGiada\\deconvDPC\\experimental_data\\1shot\\sparse\\sparse_deconv_0.0002\\"
    if not os.path.exists(path_output1):
        os.makedirs(path_output1)

    process_and_save(0.0002, path_output1)

    

    # TV deconvolution

    path_output="D:\\FromPEPItoGiada\\deconvDPC\\experimental_data\\1shot\\tv\\tv_deconv_0.05\\"
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    process_and_save2(0.05, path_output)
    
