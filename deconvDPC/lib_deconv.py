import numpy as np
from scipy.ndimage import median_filter, generic_filter, zoom
from numpy.fft import fftshift, ifftshift, fft2,ifft2
from scipy.signal import convolve2d
from scipy.ndimage import convolve

def percentile_normalization(source, reference, p_low=1, p_high=99):
    src_low, src_high = np.percentile(source, [p_low, p_high])
    ref_low, ref_high = np.percentile(reference, [p_low, p_high])
   
    # Normalize source to match reference contrast range
    source_norm = (source - src_low) / (src_high - src_low)
    source_scaled = source_norm * (ref_high - ref_low) + ref_low
   
    return source_scaled

def _regularization (image):
    
    corr=np.mean(image,axis=1)
    corr=np.tile(corr,(np.size(image,1),1)).T
    image2=image-corr
    
    #median filter (out=filtered)
    nan_mask=np.isnan(image2)
    im=np.where(nan_mask,np.nanmedian(image2),image2)
    filtered=median_filter(im,size=5)
    
    def local_std(block):
        return np.nanstd(block)
    
    #local stantard deviation (out=sd)
    sd=generic_filter(filtered, local_std, footprint=np.ones((5, 5)))
    out=image2
    out[np.abs(out-filtered)>30.*sd]=np.nan
    
    #change nan values
    image2[~np.isfinite(out)] = filtered[~np.isfinite(out)]
    
    return out



def _wiener_deconvolution (image_dataset):

    # deconvolution parameters
    image2=image_dataset[:,:,100]
    [r,c]= image2.shape[0],image2.shape[1]
    img3=np.pad(image2, pad_width=((r//2, r//2),(c//2, c//2)),  mode='constant', constant_values=0)

    #fourier transform
    F = np.fft.fftshift(np.fft.fft2(img3))
    F=np.pad(F, pad_width=((F.shape[0]//2, F.shape[0]//2),(F.shape[1]//2, F.shape[1]//2)),  mode='constant', constant_values=0)

    [y,x]=F.shape
    ulim = np.arange(-x/2 + 1, x/2, step=1)
    ulim=ulim*(2*np.pi/(x*1))
    ulim=ulim[np.newaxis,:]
    vlim = np.arange(-y/2 + 1, y/2, step=1)
    vlim=vlim*(2*np.pi/(y*1))
    vlim=vlim[np.newaxis,:]
    [u,v]=np.meshgrid(ulim,vlim)

    # initial parameters
    v0_arr = 0.0035 #cutoff frequency in vertical direction
    n=1 # order of the denominator
    s_arr = 0.5E7  #strength of the filter

    #psf and regularization parameter
    D=2j*np.sin(u)
    SNR=(s_arr/((1+v/v0_arr)**(2*n)))
    
    wiener_dset=np.zeros((image_dataset.shape[0],992,image_dataset.shape[2]))
    denom=np.abs(D)**2 + (1/(SNR))
    D=np.pad(D, pad_width=((1, 0), (1, 0)), mode='constant', constant_values=0)
    denom=np.pad(denom, pad_width=((1, 0), (1, 0)), mode='constant', constant_values=0)
    [r,c]= image_dataset.shape[0],image_dataset.shape[1]
    
    for i in range(image_dataset.shape[2]):
        image=np.pad(image_dataset[:,:,i], pad_width=((r//2, r//2),(c//2, c//2)),  mode='constant', constant_values=0)
        F = np.fft.fftshift(np.fft.fft2(image))
        F=np.pad(F, pad_width=((F.shape[0]//2, F.shape[0]//2),(F.shape[1]//2, F.shape[1]//2)), mode='constant', constant_values=0)
        phi=np.fft.ifft2(np.fft.ifftshift(((np.conj(D)/np.maximum(denom, np.finfo(np.float64).eps)*F))))
        phi=zoom(phi, (0.5,0.5))
        phi=phi[121:365,495:1487]
        diff_phi=(np.real(phi))*(-1)
        wiener_dset[:,:,i]=diff_phi
        
    return wiener_dset



def _deconvL2_w(I, filt1, we, max_it=200, weight_x=None, weight_y=None, weight_xx=None, weight_yy=None, weight_xy=None):
    n, m = I.shape

    hfs1_x1 = (filt1.shape[1] - 1) // 2
    hfs1_x2 = (filt1.shape[1]) // 2
    hfs1_y1 = (filt1.shape[0] - 1) // 2
    hfs1_y2 = (filt1.shape[0]) // 2

    hfs_x1 = hfs1_x1
    hfs_x2 = hfs1_x2
    hfs_y1 = hfs1_y1
    hfs_y2 = hfs1_y2

    m += hfs_x1 + hfs_x2
    n += hfs_y1 + hfs_y2
    N = m * n
    mask = np.zeros((n, m))
    mask[hfs_y1:n - hfs_y2, hfs_x1:m - hfs_x2] = 1

    if weight_x is None:
        weight_x = np.ones((n, m - 1))
        weight_y = np.ones((n - 1, m))
        weight_xx = np.zeros((n, m - 2))
        weight_yy = np.zeros((n - 2, m))
        weight_xy = np.zeros((n - 1, m - 1))
            
    x = np.pad(I, ((hfs_y1, hfs_y2), (hfs_x1, hfs_x2)), mode='edge')

    b = convolve2d(x * mask, filt1, mode='same')

    dxf = np.atleast_2d(np.array([1, -1]))
    dyf = np.atleast_2d(np.array([[1], [-1]]))
    dyyf = np.atleast_2d(np.array([[-1], [2], [-1]]))
    dxxf = np.atleast_2d(np.array([[-1, 2, -1]]))
    dxyf = np.array([[-1, 1], [1, -1]])

    Ax = convolve2d(convolve2d(x, np.flipud(np.fliplr(filt1)), mode='same') * mask, filt1, mode='same')

    Ax += we * convolve2d(weight_x * convolve2d(x, np.flipud(np.fliplr((dxf))), mode='valid'), dxf)
    Ax += we * convolve2d(weight_y * convolve2d(x, np.flipud(np.fliplr(dyf)), mode='valid'), dyf)
    Ax += we * (convolve2d(weight_xx * convolve2d(x, np.flipud(np.fliplr(dxxf)), mode='valid'), dxxf))
    Ax += we * (convolve2d(weight_yy * convolve2d(x, np.flipud(np.fliplr(dyyf)), mode='valid'), dyyf))
    Ax += we * (convolve2d(weight_xy * convolve2d(x, np.flipud(np.fliplr(dxyf)), mode='valid'), dxyf))

    r = b - Ax

    for iter in range(1, max_it + 1):
        rho = np.sum(r ** 2)

        if iter > 1:
            beta = rho / rho_1
            p = r + beta * p
        else:
            p = r

        Ap = convolve2d(convolve2d(p, np.flipud(np.fliplr(filt1)), mode='same') * mask, filt1, mode='same')
        Ap += we * convolve2d(weight_x * convolve2d(p, np.flipud(np.fliplr(dxf)), mode='valid'), dxf)
        Ap += we * convolve2d(weight_y * convolve2d(p, np.flipud(np.fliplr(dyf)), mode='valid'), dyf)
        Ap += we * (convolve2d(weight_xx * convolve2d(p, np.flipud(np.fliplr(dxxf)), mode='valid'), dxxf))
        Ap += we * (convolve2d(weight_yy * convolve2d(p, np.flipud(np.fliplr(dyyf)), mode='valid'), dyyf))
        Ap += we * (convolve2d(weight_xy * convolve2d(p, np.flipud(np.fliplr(dxyf)), mode='valid'), dxyf))

        q = Ap
        alpha = rho / np.sum(p * q)
        x = x + alpha * p
        r = r - alpha * q
        rho_1 = rho

    return x



def deconvSps(I, filt1, we, max_it=200):
    # Note: size(filt1) is expected to be odd in both dimensions
    
    filt1 = filt1 * -1
    
    n, m = I.shape
    
    hfs1_x1 = (filt1.shape[1] - 1) // 2
    hfs1_x2 = (filt1.shape[1]) // 2
    hfs1_y1 = (filt1.shape[0] - 1) // 2
    hfs1_y2 = (filt1.shape[0]) // 2

    hfs_x1 = hfs1_x1
    hfs_x2 = hfs1_x2
    hfs_y1 = hfs1_y1
    hfs_y2 = hfs1_y2

    m += hfs_x1 + hfs_x2
    n += hfs_y1 + hfs_y2
    N = m * n
    mask = np.zeros((n, m))
    mask[hfs_y1:n - hfs_y2, hfs_x1:m - hfs_x2] = 1

    tI = I.copy()
    I = np.zeros((n, m))
    I[hfs_y1:n - hfs_y2, hfs_x1:m - hfs_x2] = tI
    x = I.copy()

    dxf = np.atleast_2d(np.array([1, -1]))
    dyf = np.atleast_2d(np.array([[1], [-1]]))
    dyyf = np.atleast_2d(np.array([[-1], [2], [-1]]))
    dxxf = np.atleast_2d(np.array([[-1, 2, -1]]))
    dxyf = np.array([[-1, 1], [1, -1]])

    weight_x = np.ones((n, m - 1))
    weight_y = np.ones((n - 1, m))
    weight_xx = np.ones((n, m - 2))
    weight_yy = np.ones((n - 2, m))
    weight_xy = np.ones((n - 1, m - 1))

    x = _deconvL2_w(x[hfs_y1:n - hfs_y2, hfs_x1:m - hfs_x2], filt1, we, max_it, weight_x, weight_y, weight_xx, weight_yy, weight_xy)

    w0 = 0.1
    exp_a = 0.8
    thr_e = 0.01

    for t in range(2):
        dy = convolve2d(x, np.fliplr(np.flipud(dyf)), mode='valid')
        dx = convolve2d(x, np.fliplr(np.flipud(dxf)), mode='valid')
        dyy = convolve2d(x, np.fliplr(np.flipud(dyyf)), mode='valid')
        dxx = convolve2d(x, np.fliplr(np.flipud(dxxf)), mode='valid')
        dxy = convolve2d(x, np.fliplr(np.flipud(dxyf)), mode='valid')

        weight_x = w0 * np.maximum(np.abs(dx), thr_e) ** (exp_a - 2)
        weight_y = w0 * np.maximum(np.abs(dy), thr_e) ** (exp_a - 2)
        weight_xx = 0.25 * w0 * np.maximum(np.abs(dxx), thr_e) ** (exp_a - 2)
        weight_yy = 0.25 * w0 * np.maximum(np.abs(dyy), thr_e) ** (exp_a - 2)
        weight_xy = 0.25 * w0 * np.maximum(np.abs(dxy), thr_e) ** (exp_a - 2)

        x = _deconvL2_w(I[hfs_y1:n - hfs_y2, hfs_x1:m - hfs_x2], filt1, we, max_it, weight_x, weight_y, weight_xx, weight_yy, weight_xy)

    x = x[hfs_y1:n - hfs_y2, hfs_x1:m - hfs_x2]
    
    # Correct one pixel shift:
    x = np.concatenate(( x[:, 1:], x[:, -1:]), axis=1)

    return x

    

def _grad(M, bound="sym", order=1):
    """
        grad - gradient, forward differences
        
          [gx,gy] = grad(M, options);
        or
          g = grad(M, options);
        
          options.bound = 'per' or 'sym'
          options.order = 1 (backward differences)
                        = 2 (centered differences)
        
          Works also for 3D array.
          Assme that the function is evenly sampled with sampling step 1.
        
          See also: div.
        
          Copyright (c) Gabriel Peyre
    """    


    # retrieve number of dimensions
    nbdims = np.ndim(M)
    
    
    if bound == "sym":  
        nx = np.shape(M)[0]
        if order == 1:
            fx = M[np.hstack((np.arange(1,nx),[nx-1])),:] - M
        else:
            fx = (M[np.hstack((np.arange(1,nx),[nx-1])),:] - M[np.hstack(([0],np.arange(0,nx-1))),:])/2.
            # boundary
            fx[0,:] = M[1,:]-M[0,:]
            fx[nx-1,:] = M[nx-1,:]-M[nx-2,:]
            
        if nbdims >= 2:
            ny = np.shape(M)[1]
            if order == 1:
                fy = M[:,np.hstack((np.arange(1,ny),[ny-1]))] - M
            else:
                fy = (M[:,np.hstack((np.arange(1,ny),[ny-1]))] - M[:,np.hstack(([0],np.arange(ny-1)))])/2.
                # boundary
                fy[:,0] = M[:,1]-M[:,0]
                fy[:,ny-1] = M[:,ny-1]-M[:,ny-2]
    
        if nbdims >= 3:
            nz = np.shape(M)[2]
            if order == 1:
                fz = M[:,:,np.hstack((np.arange(1,nz),[nz-1]))] - M
            else:
                fz = (M[:,:,np.hstack((np.arange(1,nz),[nz-1]))] - M[:,:,np.hstack(([0],np.arange(nz-1)))])/2.
                # boundary
                fz[:,:,0] = M[:,:,1]-M[:,:,0]
                fz[:,:,ny-1] = M[:,:,nz-1]-M[:,:,nz-2]            
    else:
        nx = np.shape(M)[0]
        if order == 1:
            fx = M[np.hstack((np.arange(1,nx),[0])),:] - M
        else:
            fx = (M[np.hstack((np.arange(1,nx),[0])),:] - M[np.hstack(([nx-1],np.arange(nx-1))),:])/2.
            
        if nbdims >= 2:
            ny = np.shape(M)[1]
            if order == 1:
                fy = M[:,np.hstack((np.arange(1,ny),[0]))] - M
            else:
                fy = (M[:,np.hstack((np.arange(1,ny),[0]))] - M[:,np.hstack(([ny-1],np.arange(ny-1)))])/2.
        
        if nbdims >= 3:
            nz = np.shape(M)[2]
            if order == 1:
                fz = M[:,:,np.hstack((np.arange(1,nz),[0]))] - M
            else:
                fz = (M[:,:,np.hstack((np.arange(1,nz),[0]))] - M[:,:,np.hstack(([nz-1],np.arange(nz-1)))])/2.   
   
    if nbdims==2:
        fx = np.concatenate((fx[:,:,np.newaxis],fy[:,:,np.newaxis]), axis=2)
    elif nbdims==3:
        fx = np.concatenate((fx[:,:,:,np.newaxis],fy[:,:,:,np.newaxis],fz[:,:,:,np.newaxis]),axis=3)
    
    return fx



def _div(Px, Py=None, bound="sym", order=1):
    """
        div - divergence operator

        fd = div(Px,Py, options);
        fd = div(P, options);

          options.bound = 'per' or 'sym'
          options.order = 1 (backward differences)
                        = 2 (centered differences)

          Note that the -div and grad operator are adjoint
          of each other such that 
              <grad(f),g>=<f,-div(g)>

          See also: grad.

        Copyright (c) 2007 Gabriel Peyre
    """

    Pz = None
    ny = None
    fy = None
    fz = None

    # retrieve number of dimensions
    nbdims = np.ndim(Px)
    if nbdims >= 3:
        if nbdims == 3:
            Py = Px[:, :, 1]
            Px = Px[:, :, 0]
            nbdims = 2
        else:
            Pz = Px[:, :, :, 2]
            Py = Px[:, :, :, 1]
            Px = Px[:, :, :, 0]
            nbdims = 3

    if bound == "sym":
        nx = np.shape(Px)[0]
        if order == 1:
            fx = Px - Px[np.hstack(([0], np.arange(0, nx-1))), :]
            fx[0, :] = Px[0, :]                        # boundary
            fx[nx-1, :] = -Px[nx-2, :]

            if nbdims >= 2:
                ny = np.shape(Py)[1]
                fy = Py - Py[:, np.hstack(([0], np.arange(0, ny-1)))]
                fy[:, 0] = Py[:, 0]                    # boundary
                fy[:, ny-1] = -Py[:, ny-2]

            if nbdims >= 3:
                nz = np.shape(Pz)[2]
                fz = Pz - Pz[:, :, np.hstack(([0], np.arange(0, nz-1)))]
                fz[:, :, 0] = Pz[:, :, 0]                # boundary
                fz[:, :, nz-1] = -Pz[:, :, nz-2]
        else:
            fx = (Px[np.hstack((np.arange(1, nx), [nx-1])), :] -
                  Px[np.hstack(([0], np.arange(0, nx-1))), :])/2.
            fx[0, :] = + Px[1, :]/2. + Px[0, :]           # boundary
            fx[1, :] = + Px[2, :]/2. - Px[0, :]
            fx[nx-1, :] = - Px[nx-1, :]-Px[nx-2, :]/2.
            fx[nx-2, :] = + Px[nx-1, :]-Px[nx-3, :]/2.

            if nbdims >= 2:
                ny = np.shape(Py)[1]
                fy = (Py[:, np.hstack((np.arange(1, ny), [ny-1]))] -
                      Py[:, np.hstack(([0], np.arange(0, ny-1)))])/2.
                fy[:, 0] = + Py[:, 1]/2. + Py[:, 0]       # boundary
                fy[:, 1] = + Py[:, 2]/2. - Py[:, 0]
                fy[:, ny-1] = - Py[:, ny-1]-Py[:, ny-2]/2.
                fy[:, ny-2] = + Py[:, ny-1]-Py[:, ny-3]/2.

            if nbdims >= 3:
                nz = np.shape(Pz)[2]
                fz = (Pz[:, :, np.hstack((np.arange(1, nz), [nz-1]))] -
                      Pz[:, :, np.hstack(([0], np.arange(0, nz-1)))])/2.
                fz[:, :, 0] = + Pz[:, :, 1]/2. + Pz[:, :, 0]  # boundary
                fz[:, :, 1] = + Pz[:, :, 2]/2. - Pz[:, :, 0]
                fz[:, :, ny-1] = - Pz[:, :, nz-1]-Pz[:, :, nz-2]/2.
                fz[:, :, ny-2] = + Pz[:, :, nz-1]-Pz[:, :, nz-3]/2.
    else:
        if order == 1:
            nx = np.shape(Px)[0]
            fx = Px-Px[np.hstack(([nx-1], np.arange(0, nx-1))), :]

            if nbdims >= 2:
                ny = np.shape(Py)[1]
                fy = Py-Py[:, np.hstack(([ny-1], np.arange(0, ny-1)))]

            if nbdims >= 3:
                nz = np.shape(Pz)[2]
                fz = Pz-Pz[:, :, np.hstack(([nz-1], np.arange(0, nz-1)))]

        else:
            nx = np.shape(Px)[0]
            fx = (Px[np.hstack((np.arange(1, nx), [0])), :]) - \
                (Px[np.hstack(([nx-1], np.arange(0, nx-1))), :])

            if nbdims >= 2:
                ny = np.shape(Py)[1]
                fy = (Py[:, np.hstack((np.arange(1, ny), [0]))]) - \
                    (Py[:, np.hstack(([ny-1], np.arange(0, ny-1)))])

            if nbdims >= 3:
                nz = np.shape(Pz)[2]
                fz = (Pz[:, :, np.hstack((np.arange(1, nz), [0]))]) - \
                    (Pz[:, :, np.hstack(([nz-1], np.arange(0, nz-1)))])

    # gather result
    if nbdims == 3:
        fd = fx+fy+fz

    elif nbdims == 2:
        fd = fx+fy

    else:
        fd = fx

    return fd



def _tv_denoise(y, epsilon, lambda_, iter):
    """
    Total variation denoising using iterative gradient descent.

    Args:
    - y: Input image (2D NumPy array)
    - epsilon: Regularization parameter (float)
    - lambda_: Regularization parameter (float)
    - iter: Number of iterations (int)

    Returns:
    - fTV: Denoised image (2D NumPy array)
    """
    # Compute the step size for diffusion
    tau = 1.9 / (1 + lambda_ * 8 / epsilon)
    
    # Initialize denoised image
    fTV = y.copy()
    
    for i in range(iter):
        # Compute image gradient
        #Gx, Gy = np.gradient(fTV)
        #Gr = np.stack((Gx, Gy), axis=2)
        Gr = _grad(fTV)
        
        # Compute the norm of the gradient
        d = np.sqrt(np.sum(Gr ** 2, axis=2))
        deps = np.sqrt(epsilon ** 2 + d ** 2)
        
        # Compute the divergence of the normalized gradient
        #G0x = -convolve(Gr[..., 0] / deps, np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]))
        #G0y = -convolve(Gr[..., 1] / deps, np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]))
        #G0 = G0x + G0y
        G0 = -_div( Gr / np.expand_dims(deps, axis=-1) )
        
        # Compute the gradient of the denoised image
        G = fTV - y + lambda_ * G0
        
        # Update the denoised image
        fTV = fTV - tau * G
    
    return fTV


def tv_deconvolution(y, h, a, iter=400):
    """
    Plug and play algorithm for image deconvolution.

    Args:
    - y: Input image (2D NumPy array)
    - h: Point spread function (2D NumPy array)
    - a: Regularization parameter (float)
    - iter: Number of iterations (int)

    Returns:
    - x: Deconvolved image (2D NumPy array)
    """
    # Initialize variables
    z = y.copy()
    x = y.copy()
    
    m, n = y.shape
    
    # Pre-compute Fourier transforms:
    Fk = fft2(h, s=(2 * m, 2 * n))
    Fy = fft2(y, s=(2 * m, 2 * n))

    for i in range(iter):

        #Compute Fourier transform:
        Fz = fft2(z, s=(2 * m, 2 * n))

        # Update x
        x = np.real(ifft2((np.conj(Fk) * Fy + a * Fz) / (np.conj(Fk) * Fk + a)))
        x = x[:m, :n]  # Crop

        # Perform TV denoising:
        epsilon = 1e-2
        lam = 0.02
        iter_tv = 5
        #z = denoise_tv_chambolle(x, weight=1.0/lam, eps=epsilon, max_num_iter=iter_tv)
        z = _tv_denoise(x, epsilon, lam, iter_tv)

        # Display the result (for visualization only)
        # plt.imshow(x, cmap='gray')
        # plt.title("Plug and play iteration: " + str(i))
        # plt.axis('off')
        # plt.show(block=False)
        # plt.pause(0.1)

    # Correct for one pixel shift:
    out = np.zeros_like(x)
    out[1:, 1:] = x[:-1, :-1]
    
    return out


def wiener_deconvolution_fb(image, psf, noise_var):
    
    # Fourier transform of the input image and the point spread function (PSF)
    F = np.fft.fft2(image)
    H = np.fft.fft2(psf, s=image.shape)

    # Wiener deconvolution
    denom = np.abs(H) ** 2 + noise_var
    G = np.conj(H) / np.maximum(denom, np.finfo(np.float64).eps) # avoid division by zero     
    X = G * F

    # Inverse Fourier transform to get the restored image
    x = np.fft.ifft2(X).real
    
    # Correct one pixel shift:
    x = np.concatenate(( x[:, :1], x[:, :-1]), axis=1)
    x = np.concatenate(( x[:1, :], x[:-1, :]), axis=0)

    return x
