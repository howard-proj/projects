import numpy as np

def filterData(raw_data):
    # Filtering the data with a gaussian filtered
    sigma_gauss = 5 # width of gaussian
    t = len(raw_data)/10000 # How long the data is in seconds

    dx = 1/10000
    Nq = 1/(2*dx)     # maximum frequency
    df = 1/t   # frequency interval

    f_fft = np.arange(-Nq,Nq,df)

    ## DO FFT
    data_f = np.fft.fftshift(np.fft.fft(raw_data)) # FFT of data
    ## GAUSSIAN FILTER
    gauss_filter = np.exp(-(f_fft)**2/sigma_gauss**2)   # gaussian filter used
    data_f_filtered= data_f*gauss_filter    # gaussian filter spectrum in frquency domain
    data_t_filtered = np.fft.ifft(np.fft.ifftshift(data_f_filtered))    # bring filtered signal in time domain
    return np.real(data_t_filtered)
