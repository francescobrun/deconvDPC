import numpy as np
import tifffile
import glob
import lib_deconv
from scipy.signal import hilbert, medfilt2d


#read data

n_proj = 720;

crop_left = 380
crop_right = 165
crop_top= 9
crop_bottom = 140

path_input= "D:\\FromPEPItoGiada\\wire_flour_coffee_salt\\"
path_projection2= "D:\\FromPEPItoGiada\\deconvDPC\\projections\\"
image1=tifffile.imread("D:\\FromPEPItoGiada\\wire_flour_coffee_salt\\REF_corr_0001.tif")
image1 = image1[crop_top:image1.shape[0]-crop_bottom+1,crop_left:image1.shape[1]-crop_right+1]

projections=np.zeros((image1.shape[0],image1.shape[1],n_proj))
im_dir = glob.glob(f"{path_input}*REF_corr_*")

for i in range(projections.shape[2]):
    im=tifffile.imread(path_input+im_dir[i])
    im = im[crop_top:im.shape[0]-crop_bottom+1,crop_left:im.shape[1]-crop_right+1]
    projections[:,:,i]=im


#regularization data

projections2=np.zeros((image1.shape[0],image1.shape[1],720))
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


# WIENER deconvolution

path_output="D:\\FromPEPItoGiada\\deconvDPC\\wiener_deconvolution\\"
wiener_dset=lib_deconv._wiener_deconvolution(projections2)

for i in range(wiener_dset.shape[2]):
    tifffile.imwrite(path_output + "slice_"+ str(i).zfill(4) + ".tif", wiener_dset[:,:,i].astype(np.float32))
    

# HILBERT

path_output="D:\\FromPEPItoGiada\\deconvDPC\\hilbert_projections\\"
hilbert_dset = np.zeros( (projections2.shape[0],projections2.shape[1],projections2.shape[2]) )

for k in range(projections2.shape[0]): 
    sino_k = np.imag(hilbert(projections2[k,:,:], axis=0))
    #sino_k=(sino_k-np.min(sino_k))/(np.max(sino_k)-np.min(sino_k))
    hilbert_dset[k,:,:]=sino_k
  
for k in range(hilbert_dset.shape[2]):
    tifffile.imwrite(path_output + "slice_"+ str(k).zfill(4) + ".tif", hilbert_dset[:,:,k].astype(np.float32))



# SPARSE deconvolution

path_output="D:\\FromPEPItoGiada\\deconvDPC\\sparse_deconvolution_0.07\\"
dset_s=np.zeros((projections2.shape))

for i in range (projections2.shape[2]):
    dset_s[:,:,i]=lib_deconv.deconvSps(projections2[:,:,i], psf, 0.2, 400)
    #dset_s[:,:,i]=(dset_s[:,:,i]-np.min(dset_s[:,:,i]))/(np.max(dset_s[:,:,i])/np.min(dset_s[:,:,i]))
    tifffile.imwrite(path_output + "slice_"+ str(i).zfill(4) + ".tif", dset_s[:,:,i].astype(np.float32))    



# TV deconvolution

path_output="D:\\FromPEPItoGiada\\deconvDPC\\tv_deconvolution_0.7\\"
dset_tv=np.zeros((projections2.shape))

for i in range (projections2.shape[2]):
    dset_tv[:,:,i]=lib_deconv.tv_deconvolution(projections2[:,:,i], psf, 0.025, 400)
    #dset_tv[:,:,i]=(dset_tv[:,:,i]-np.min(dset_tv[:,:,i]))/(np.max(dset_tv[:,:,i])/np.min(dset_tv[:,:,i]))
    tifffile.imwrite(path_output + "slice_"+ str(i).zfill(4) + ".tif", dset_tv[:,:,i].astype(np.float32))    
