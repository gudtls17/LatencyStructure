import numpy as np
from brainspace import gradient

# make functional connectivity

def functional_connectivity_generator(data):
    """_summary_

    Args:
        data (_type_): (subject, ROI, BOLD timepoint)

    Returns:
        (subject, ROI, ROI)
    """
    for i in range(len(data)):
        if i%100 == 0:
            print(i, '', end='', flush=True)
        
        ts = data[i]
        conn_mat =np.nan_to_num(np.where(np.eye(len(ts)) ==1, 0, np.corrcoef(ts)))
        conn_mat_rtoz =  np.nan_to_num(np.arctanh(conn_mat),nan=0.0)
        
        
        if i == 0:
            conn_mat_list = np.expand_dims(conn_mat, axis=0)
            conn_mat_rtoz_list = np.expand_dims(conn_mat_rtoz, axis=0)
        else:
            conn_mat_list = np.concatenate((conn_mat_list, np.expand_dims(conn_mat, axis=0)), axis=0)
            conn_mat_rtoz_list = np.concatenate((conn_mat_rtoz_list, np.expand_dims(conn_mat_rtoz, axis=0)), axis=0)
            
    return conn_mat_list, conn_mat_rtoz_list

# make afiinity matrix (individual)

def gradient_generator(data, data_ref, sparsity: float=0.9, comp_num: int=5):
    """_summary_

    Args:
        data (_type_): (subject, ROI, ROI), FC list
        data_ref (_type_): (ROI, gradient), groupmena gradient
        comp_num : eigenvalue component number

    Returns:
        (subject, ROI, gradient)
    """

    sparsity = sparsity
    comp_num = comp_num
    # emb_dm = gradient.embedding.DiffusionMaps(n_components = comp_num, random_state=42)  # diffusion map
    emb_dm = gradient.embedding.PCAMaps(n_components = comp_num, random_state=42)  # PCA

    k = str(int(100-sparsity*100))
    k = k.zfill(2)

    print(f'Top {k}')

    for i in range(len(data)):
        if i%100 == 0:
            print(i, '', end='', flush=True)
        
        fc_rtoz = data[i]  # select 1 subj data
            
        aff_fc_rtoz = gradient.compute_affinity(fc_rtoz, kernel = 'cosine', sparsity = sparsity)
        
        emb_dm.fit(aff_fc_rtoz)
        lam, grad = [None]*1, [None]*1
        lam[0], grad[0] = emb_dm.lambdas_ , emb_dm.maps_    # calculate eigenvalue and eigenvector

        pa = gradient.ProcrustesAlignment(n_iter=10)        # Procrustes alignment
        pa.fit(grad, reference=data_ref)
        aligned = np.array(pa.aligned_)
        
        if i == 0:
            gradient_list = aligned
        else:
            gradient_list = np.concatenate((gradient_list, aligned), axis=0)
            
    return gradient_list