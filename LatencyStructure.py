import os, sys
from os.path import join
import scipy as sc
import numpy as np
from scipy.signal import correlate
import time

# compute latency structure tau matrix

TR = 0.72            # sec, HCP S1200 TR
freq_samp = 30       # interpolation factor for TR sampling
delay_thresh = 5    # Delay threshold (sec)
interpolate = False
TRtoSecond = True
Save = False

"""
input data : ts_list_clean, BOLD timeseries 
"""

for subj in range(len(ts_list_clean))[160:]:
    start_time = time.time()
    subj_ID = sub_list_hcp[subj]  # set subject ID
    print(f'{subj} / {len(ts_list_clean)}  {subj_ID}', end=' ', flush=True)
    
    # [NOTE] if save another path (Not main_path), set save_path
    save_path = join(data_path, 'HCP', str(subj_ID), 'MNINonLinear/Results')
    
    TsDemean_L = ts_list_clean[subj].T  # (timepoint, ROI)
    TimePoint = TsDemean_L.shape[0]
    NumVox = TsDemean_L.shape[1]

    TauMat = np.zeros((NumVox, NumVox))  # set tau template
    ExtremumMat = np.zeros((NumVox, NumVox))  # set signal extremum amplitude template
    
    for i in range(NumVox):
        for j in range(NumVox):
            # compute cross covariance
            cc = correlate(TsDemean_L[:, i], TsDemean_L[:, j], mode='full')  # cross-correlation with lags
            lags = sc.signal.correlation_lags(len(TsDemean_L[:, i]), len(TsDemean_L[:, j]), mode='full')  # extract lags
            
            if interpolate:
                # interpolation (TR to seconds)
                cc = np.interp(np.linspace(0, len(cc), len(cc)*freq_samp), np.arange(len(cc)), cc)  # interpolate cross-covariance
                lags = np.arange(-(TimePoint - 1) * freq_samp, ((TimePoint - 1) * freq_samp) + freq_samp)  # Interpolate lags
                TR = TR/freq_samp  # Sampling time after interpolation
            
            # compute tau
            abs_argmax_idx = np.argmax(np.abs(cc))  # find extremum (consider both positive and negative values)
            ExtremumMat[i,j] = cc[abs_argmax_idx]
            
            if TRtoSecond:
                tau = lags[abs_argmax_idx] * TR  # convert element unit TR timepotin to seconds
            else:
                tau = lags[abs_argmax_idx]
            TauMat[i, j] = tau
            
            # threshold the TauMat with delay threshold
            TauMatThr = TauMat.copy()
            TauMatThr[np.where(np.abs(TauMat) > delay_thresh)] = 0
        
            
    # TauProj = TauMat.mean(axis=1)  # latency projections of the TD
    # TauThrProj = TauMatThr.mean(axis=1)
    
    """Save each subject folder in Area HCP dataset"""
    np.save(join(save_path, 'tau_mat_ctxsctx_Scha300_Surface.npy'), TauMat)
    np.save(join(save_path, 'tau_mat_thr_ctxsctx_Scha300_Surface.npy'), TauMatThr)
    np.save(join(save_path, 'tau_extremum_mat_ctxsctx_Scha300_Surface.npy'), ExtremumMat)
    
    print(f' {time.time() - start_time} seconds!')
