import numpy as np
from statsmodels.tsa.stattools import acf


"""Define fitting function"""
def autocorr_decay(dk,A,tau,B):  # define curvefit function
    return A*(np.exp(-(dk/tau))+B)

"""Calculate Intrisic timescale"""
TR = 0.72            # sec, HCP S1200 TR
maxlag = 50  # set range of max timestep (timepoint)
maxlag2sec = maxlag * TR  # convert max lag scale, timepoint to seconds
# print(maxlag2sec)
xdata = np.arange(1,maxlag+1) * TR  # drop fisrt element. Self-AutoCorr corrcoef is always 1
delay_thresh = 100  # Delay threshold (sec)

"""
input data : ts_list_clean, BOLD timeseries 
"""

input_data = ts_list_clean  # (Nsubj, ROI, timepoint)
NumSubj = ts_list_clean.shape[0]
NumVox = ts_list_clean.shape[1]
TimePoint = ts_list_clean.shape[2]

ITS_list = np.zeros((NumSubj, NumVox))  # tau
ITS_Autocorr_list = np.zeros((NumSubj, NumVox, maxlag))  # R

for subj in range(len(input_data)):
    if subj%10 == 0:
        print(subj, '', end=':', flush=True)
        print('')

    TsDemean_L = input_data[subj].T  # (timepoint, ROI)
    # TimePoint = TsDemean_L.shape[0]
    # NumVox = TsDemean_L.shape[1]

    for i in range(NumVox):
        # tscale = sc.signal.correlate(ts_list_clean[0].T[:,i], ts_list_clean[0].T[:,i], mode='full')
        
        acf_value  = acf(TsDemean_L[:, i], nlags=maxlag, fft=True)
        acf_value = acf_value[1:]  # drop first autocorr because Self-AutoCorr corrcoef is always 1

        # start = len(tscale)//2+1  # drop (x-tau)(x) part, remain only (x)(x-tau) part
        # stop = start + maxlag
        # xdata = np.arange(len(tscale))

        # tscale = tscale[start:stop]  # clip data
        # print(tscale)
        ITS_Autocorr_list[subj, i] = acf_value

        try:
            B=0  # initialize B value
            repeat_num = 0  # count repeat num
            while B==0:  # When B=0 is failed to fitting, retry calculate A, tau, B variable
                A, tau, B = sc.optimize.curve_fit(autocorr_decay, xdata, acf_value, p0=[0,np.random.rand(1)[0]+0.01,0], bounds=(([0,0,-np.inf],[np.inf,np.inf,np.inf])), method='trf')[0]
                ITS_list[subj, i] = tau
                

                repeat_num+=1
                if repeat_num ==20:
                    print(subj, i, 'No ITS value')
                    ITS_list[subj, i] = np.nan
                    break

        except:
            print(subj, i, 'No ITS value')
            ITS_list[subj, i] = np.nan
            
ITS_list[np.where(ITS_list>delay_thresh)]=0  # threshold with delay_thresh