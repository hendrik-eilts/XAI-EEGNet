import os
import mne
import umap
import torch
import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import global_settings as gs
from crp.DFT_LRP import dft_lrp
import matplotlib.pyplot as plt
import training.utils as train_utils
import matplotlib.patches as mpatches
from mne_icalabel import label_components
from crp.attribution import CondAttribution
from sklearn.cluster import DBSCAN
from relevance.results import Results
from mne_icalabel import label_components
from mne_connectivity.viz import plot_connectivity_circle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from zennit.composites import EpsilonPlus, EpsilonAlpha2Beta1, EpsilonAlpha2Beta1Flat, EpsilonPlusFlat
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from tqdm import tqdm
from copy import copy


# ---------------------------------------------------------------------------------------

def prepare_data_for_filter(batch_size, layer, path_store_data, params, class_label, dh, num_filters, composite, create_new=False, use_which_data="train", 
                            num_samples_to_select=5, reversed_classes=False, select_correct=None, select_samples_by_class=True):

    try:
        if create_new:
            subjects_done_list = []
            total_rel_per_filter = []
            selected_samples_per_subject = None
        else:
            subjects_done_list = joblib.load(os.path.join(path_store_data, "subjects_done_list"))
            total_rel_per_filter = joblib.load(os.path.join(path_store_data, "total_rel_per_filter"))
            selected_samples_per_subject = joblib.load(os.path.join(path_store_data, "selected_samples_per_subject"))
    except:
        subjects_done_list = []
        total_rel_per_filter = []
        selected_samples_per_subject = None

    if selected_samples_per_subject is None:
        selected_samples_per_subject = {}
        for subj in range(dh.num_subjects):
            selected_samples_per_subject[subj] = []

    index = len(total_rel_per_filter)

    if use_which_data == "train-test":
        X_time, y, sample_subject_map = get_data(dh, select_correct, select_samples_by_class, class_label, params, reversed_classes, use_which_data='train-test', composite=composite)
        X_time.requires_grad = True

    # --------------------------------------------------------------

    for subj in range(dh.num_subjects):

        if subj in subjects_done_list:
            continue

        if use_which_data != "train-test":
            X_time, y, sample_subject_map = get_data(dh, select_correct, select_samples_by_class, class_label, params, reversed_classes, 
                                                           use_which_data=use_which_data, subj=subj, composite=composite)
            X_time.requires_grad = True
        
        conditions = [{layer: 0, "y": class_label}] 

        model = train_utils.load_model(dh, params, X_time.shape[-2], X_time.shape[-1], subj, return_dict=False)
        attribution = CondAttribution(model, sum_over_channels=False)

        model.eval()

        total_rel_per_filter_of_subject = []

        for f_idx in range(0, num_filters):

            if class_label == 'both':
                conditions = [{layer: f_idx, "y": [0,1]}] 
            else:
                conditions = [{layer: f_idx, "y": class_label}] 

            R_filter_list = []
            AM_filter_list = []

            # --------------------

            R_time = torch.empty_like(X_time)

            for b in range(0, X_time.shape[0], batch_size):
                R_time[b:b+batch_size], AM_filter, R_filter, _ = attribution(X_time[b:b+batch_size], conditions, composite)

                if layer != "b3_flatten":
                    R_filter_list.append(R_filter[layer])
                    AM_filter_list.append(AM_filter[layer])

            # --------------------
                    
            if layer != "b3_flatten":
                R_filter_list = torch.cat(R_filter_list, dim=0)
                AM_filter_list = torch.cat(AM_filter_list, dim=0)

                # (96, 16, 1, 97)
                R_per_sample = R_filter_list.detach().clone()
                # (96, 16*97)
                R_per_sample = R_per_sample.view(R_per_sample.shape[0], -1)
                # (96,)
                R_per_sample = torch.sum(R_per_sample, dim=1)
                R_per_sample = torch.abs(R_per_sample)

                # normalize
                R_per_sample /= torch.sum(R_per_sample)

                # sort by value (descending)
                indices = torch.argsort(R_per_sample, descending=True)
                R_per_sample = R_per_sample[indices]
            else:

                # (n, 32, 384)
                R_per_sample = R_time.detach().clone()
                # (n,32*284)
                R_per_sample = R_per_sample.view(R_per_sample.shape[0], -1)
                # (n,)
                R_per_sample = torch.sum(R_per_sample, dim=1)
                R_per_sample = torch.abs(R_per_sample)

                # normalize
                R_per_sample /= torch.sum(R_per_sample)

                # sort by value (descending)
                indices = torch.argsort(R_per_sample, descending=True)
                R_per_sample = R_per_sample[indices]

            # ---------------------------

            # select k most relevant samples
            if 0 < num_samples_to_select <= 1.0: # via threshold

                sum_rel = 0
                select_k = 0

                for rel_of_sample in R_per_sample:
                    sum_rel += rel_of_sample
                    select_k += 1

                    if sum_rel > num_samples_to_select:
                        break 

                # print(select_k)
                    
                indices = indices[:select_k]

            else: # via fixed value
                indices = torch.argsort(R_per_sample, descending=True)[:num_samples_to_select]

            indices_np = indices.cpu().detach().numpy()

            for sample_index in indices_np:
                try:
                    s = sample_subject_map[sample_index]
                except:
                    s = 0

                selected_samples_per_subject[s].append(sample_index)

            # --------------------

            rel_of_filter = torch.sum(R_per_sample)

            total_rel_per_filter_of_subject.append(rel_of_filter.cpu().detach().numpy())

            R_time_subset = R_time[indices]

            # -----------------------------------------

            X_time_subset = X_time[indices]
            y_time_subset = y[indices_np]

            if layer != "b3_flatten":
                R_filter_list = R_filter_list[indices]
                AM_filter_list = AM_filter_list[indices]

                AM_filter_list = AM_filter_list[:, f_idx, :, :]
                AM_filter_list = torch.mean(AM_filter_list, dim=0)
                AM_filter_list = AM_filter_list.view(-1) 
                joblib.dump(AM_filter_list, os.path.join(path_store_data, f"filter_activation_map_{subj}_{f_idx}"))

                R_filter_list = R_filter_list[:, f_idx, :, :]
                R_filter_list = torch.mean(R_filter_list, dim=0)
                R_filter_list = R_filter_list.view(-1) 
                joblib.dump(R_filter_list, os.path.join(path_store_data, f"filter_activation_map_rel_{subj}_{f_idx}"))

            # ---------------------------------

            dftlrp = dft_lrp.DFTLRP(X_time.shape[-1], leverage_symmetry=True, precision=32, 
                                        create_stdft=False, create_inverse=False, cuda=torch.cuda.is_available())

            R_freq_subset = np.empty((X_time_subset.shape[0], X_time_subset.shape[1], 193))
            X_freq_subset = np.empty((X_time_subset.shape[0], X_time_subset.shape[1], 193))

            for b in range(0, X_time_subset.shape[0], batch_size):
                X_freq_subset[b:b+batch_size], R_freq_subset[b:b+batch_size] = dftlrp.dft_lrp(R_time_subset[b:b+batch_size], X_time_subset[b:b+batch_size], real=False, short_time=False)

            X_freq_subset = torch.abs(torch.from_numpy(X_freq_subset).type(torch.cuda.FloatTensor))

            # ------------------------

            description_of_data = "R_time_select"
            joblib.dump(R_time_subset.cpu().detach().numpy(), os.path.join(path_store_data, f"{description_of_data}_{subj}_{f_idx}"))

            description_of_data = "X_time_select"
            joblib.dump(X_time_subset.cpu().detach().numpy(), os.path.join(path_store_data, f"{description_of_data}_{subj}_{f_idx}"))

            description_of_data = "y_select"
            joblib.dump(y_time_subset, os.path.join(path_store_data, f"{description_of_data}_{subj}_{f_idx}"))

            description_of_data = "X_freq_select"
            joblib.dump(X_freq_subset.cpu().detach().numpy(), os.path.join(path_store_data, f"{description_of_data}_{subj}_{f_idx}"))
    
            description_of_data = "R_freq_select"
            joblib.dump(R_freq_subset, os.path.join(path_store_data, f"{description_of_data}_{subj}_{f_idx}"))

            del X_freq_subset, R_freq_subset

            index += 1

        total_rel_per_filter += total_rel_per_filter_of_subject
        subjects_done_list.append(subj)
        joblib.dump(subjects_done_list, os.path.join(path_store_data, "subjects_done_list"))
        joblib.dump(total_rel_per_filter, os.path.join(path_store_data, "total_rel_per_filter"))
        joblib.dump(selected_samples_per_subject, os.path.join(path_store_data, "selected_samples_per_subject"))

    joblib.dump(total_rel_per_filter, os.path.join(path_store_data, "total_rel_per_filter_final"))
    joblib.dump(selected_samples_per_subject, os.path.join(path_store_data, "selected_samples_per_subject"))

# ---------------------------------------------------------------------------------------

def freq_band_name_to_latex(symbol):

    greek_mapping = {
        'alpha': r'$\alpha$',
        'beta': r'$\beta$',
        'gamma': r'$\gamma$',
        'delta': r'$\delta$',
        'theta': r"$\theta$",
    }
    
    return greek_mapping.get(symbol.lower(), symbol)

# ---------------------------------------------------------------------------------------

def compute_ICAs(data, channel_names, sampling_rate, plot=True, raw=False):

    if not raw:
        if len(data.shape) > 2: 
            data = np.reshape(data, (data.shape[1], -1))

        info = mne.create_info(channel_names, sampling_rate, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        raw.set_montage('standard_1020')  
    else:
        raw = data

    ica = mne.preprocessing.ICA(
        n_components=None,
        max_iter="auto",
        method="fastica",
        # method="infomax",
        random_state=0,
        # fit_params=dict(extended=True),
    )

    picks = np.arange(len(channel_names))
    ica.fit(raw) 

    if plot:
        plt.figure(figsize=(4,4))
        ica.plot_components(inst=raw, picks=picks, show=False, size=0.5) # , float=0.9) # , axes=ax)
        plt.figure(figsize=(5,5))
        plt.show()

    return raw, ica

# ---------------------------------------------------------------------------------------

# def get_channels_by_brain_region(ch_names, brain_region=""):
#     brain_region = brain_region.lower()

#     if brain_region == "all":
#         return ch_names, [i for i in range(0, len(ch_names))]
#     elif "frontal" in brain_region:
#         start_ch = "F"
#     elif "occipital" in brain_region:
#         start_ch = "O"
#     elif "temporal" in brain_region:
#         start_ch = "T"
#     elif "parietal" in brain_region:
#         start_ch = "P"
#     elif "central" in brain_region:
#         start_ch = "C"
#     elif "motor_cortex" in brain_region:

#         pass

#     else: 
#         print("brain region not recognized")
#         return ch_names, [i for i in range(0, len(ch_names))]

#     if "left" in brain_region:
#         rem = ["1"]
#     elif "right" in brain_region:
#         rem = ["0"]
#     elif "middle" in brain_region:
#         rem = ["Z"]
#     else:
#         rem = ["0","1","Z"]

#     channels = []
#     for ch_name in ch_names:

#         ch_name_ = ch_name.upper()

#         if ch_name_.startswith(start_ch):
#             if ch_name[-1].isdigit():
#                 rm = int(ch_name[-1]) % 2

#                 if str(rm) in rem: 
#                     channels.append(ch_name)
#             else:
                
#                 if ch_name_[-1] in rem:

#                     channels.append(ch_name)

#     # channels = [ch_name for ch_name in ch_names if ch_name.upper().startswith(start_ch) and ch_name[-1].isdigit() and int(ch_name[-1]) % 2 in rem]
#     indices = [i for i, ch_name in enumerate(ch_names) if ch_name in channels]

#     return channels, indices


# ---------------------------------------------------------------------------------------

def get_channels_by_brain_region1(ch_names, brain_region=""):
    brain_region = brain_region.lower()

    if brain_region == "all":
        return ch_names, [i for i in range(0, len(ch_names))]
    elif "frontal" in brain_region:
        start_ch = "F"
    elif "occipital" in brain_region:
        start_ch = "O"
    elif "temporal" in brain_region:
        start_ch = "T"
    elif "parietal" in brain_region:
        start_ch = "P"
    elif "central" in brain_region:
        start_ch = "C"
    elif "motor_cortex" in brain_region:

        pass

    else: 
        print("brain region not recognized")
        return ch_names, [i for i in range(0, len(ch_names))]

    if "left" in brain_region:
        rem = ["1"]
    elif "right" in brain_region:
        rem = ["0"]
    elif "middle" in brain_region:
        rem = ["Z"]
    else:
        rem = ["0","1","Z"]

    channels = []
    for ch_name in ch_names:

        ch_name_ = ch_name.upper()

        if ch_name_.startswith(start_ch):
            if ch_name[-1].isdigit():
                rm = int(ch_name[-1]) % 2

                if str(rm) in rem: 
                    channels.append(ch_name)
            else:
                
                if ch_name_[-1] in rem:

                    channels.append(ch_name)

    # channels = [ch_name for ch_name in ch_names if ch_name.upper().startswith(start_ch) and ch_name[-1].isdigit() and int(ch_name[-1]) % 2 in rem]
    indices = [i for i, ch_name in enumerate(ch_names) if ch_name in channels]

    return channels, indices

# ---------------------------------------------------------------------------------------

def load_data_and_compute_relevance(dh, params, class_label, title="", proportion_of_samples=1.0, discard_negative_rel=False, reversed_classes=False, model_id_of_subj=None, subject="all",
                      select_correct=None, select_samples_by_class=True, composite=None, freq_domain=True, use_which_data="train-test"):

    X_time, y, _ = get_data(dh, select_correct, select_samples_by_class, class_label, params, reversed_classes, subj=subject, composite=composite, use_which_data=use_which_data)

    model = train_utils.load_model(dh, params, X_time.shape[-2], X_time.shape[-1], model_id_of_subj, return_dict=False)

    attribution = CondAttribution(model, sum_over_channels=False)
    composite = EpsilonAlpha2Beta1() 
    conditions = [{"y": class_label}] 

    X_time.requires_grad = True

    R_time = torch.empty_like(X_time)

    batch_size = 1

    for b in range(0, X_time.shape[0], batch_size):
        R_time[b:b+batch_size], _, _, _ = attribution(X_time[b:b+batch_size], conditions, composite)
    del _

    # -------------------------------

    a,b,c = R_time.shape
    R_time2d = R_time.view(a, b*c)
    R_per_sample = R_time2d
    R_per_sample = torch.sum(R_per_sample, dim=1)

    R_per_sample = torch.abs(R_per_sample)
    del R_time2d

    n_samples = int(len(X_time) * proportion_of_samples)
    # print(n_samples)

    title += f", {int(proportion_of_samples*100)}% of samples (sorted by relevance)"

    indices = torch.argsort(R_per_sample, descending=True)[:n_samples]

    X_time = X_time[indices]
    R_time = R_time[indices]

    if not freq_domain:
        if discard_negative_rel:
            R_time[R_time < 0] = 0

        return X_time, R_time

    # -------------------------------

    dftlrp = dft_lrp.DFTLRP(X_time.shape[-1], leverage_symmetry=True, precision=32, create_stdft=False, create_inverse=False, cuda=torch.cuda.is_available())

    R_freq = np.empty((R_time.shape[0], R_time.shape[1], 193))
    X_freq = np.empty((X_time.shape[0], X_time.shape[1], 193))

    for b in range(0, X_time.shape[0], batch_size):
        X_freq[b:b+batch_size], R_freq[b:b+batch_size] = dftlrp.dft_lrp(R_time[b:b+batch_size], X_time[b:b+batch_size], real=False, short_time=False)

    X_freq = torch.abs(torch.from_numpy(X_freq).type(torch.cuda.FloatTensor))

    X_freq = X_freq**2

    R_freq = torch.from_numpy(R_freq).type(torch.cuda.FloatTensor)

    if discard_negative_rel:
        R_freq[R_freq < 0] = 0

    if torch.is_tensor(X_freq):
        X_freq = X_freq.cpu().detach().numpy()

    if torch.is_tensor(R_freq):
        R_freq = R_freq.cpu().detach().numpy()
    
    return X_freq, R_freq

# -----------------------------------------------------------------------------------------

def compute_embedding(data, layer):
    data = np.array(data)
    metric = "cosine"
    reducer = umap.UMAP(random_state=42, n_neighbors=data.shape[0], min_dist=0, init="random", metric=metric)
    embedding = reducer.fit_transform(data)

    return embedding

# -----------------------------------------------------------------------------------------
    
def cluster_embedding(embedding, eps, min_samples, title="", plot=True):
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embedding)
    unique_labels = np.unique(cluster_labels)

    cluster_indices = [np.where(cluster_labels == label)[0] for label in unique_labels] 

    unique_labels_original = copy(unique_labels)
    cluster_indices_original = copy(cluster_indices)

    if plot:
        plt.figure(figsize=(4, 4))
        c_idx = 0
        for i, label in enumerate(unique_labels):
            if label == -1:
                plt.scatter(embedding[cluster_indices[i]][:, 0], embedding[cluster_indices[i]][:, 1], label='NC', alpha=0.25)
            else:
                plt.scatter(embedding[cluster_indices[i]][:, 0], embedding[cluster_indices[i]][:, 1], label=f'Cluster {c_idx+1}')
                c_idx += 1

        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        fig = plt.figure()
        fig.patch.set_facecolor('white')

        plt.show()

    # ------------------------------------------------------

    unique_labels = list(np.unique(cluster_labels))

    cluster_indices = [np.where(cluster_labels == label)[0] for label in unique_labels]

    cluster_indices.append([i for i in range(len(cluster_labels))])

    unique_labels.append("all")

    unique_labels_ = []

    for l in unique_labels:
        if l == -1:
            unique_labels_.append('NC')
        elif l == 'all':
            unique_labels_.append('all')
        else:
            unique_labels_.append(str(l+1))

    return cluster_indices, cluster_labels, unique_labels_, cluster_indices_original, unique_labels_original

# -----------------------------------------------------------------------------------------

def plot_clusters(embedding_c0, embedding_c1, cluster_indices_c0, cluster_indices_c1, unique_labels_c0, unique_labels_c1, title="", 
                  class_label_map=None, cluster_color_map=None, ax=None):

    if ax is None:
        fig, axes = plt.subplots(1,2, figsize=(8,4)) # , sharey="row", sharex="row")
        ax_was_none = True
    else:
        axes = ax
        ax_was_none = False

    class_labels = [0,1]
    embeddings = [embedding_c0, embedding_c1]
    indices = [cluster_indices_c0, cluster_indices_c1]
    labels = [unique_labels_c0, unique_labels_c1]

    for idx, class_label in enumerate(class_labels):

        c_idx = 0

        ax = axes[idx]

        embedding = embeddings[idx]
        cluster_indices = indices[idx]
        unique_labels = labels[idx]

        for i, label in enumerate(unique_labels):

            if label == -1:
                label = "NC"
            else:
                label = str(label+1)

            color = cluster_color_map[label]

            if label == "NC":
                ax.scatter(embedding[cluster_indices[i]][:, 0], embedding[cluster_indices[i]][:, 1], label='NC', alpha=1, color=color)
            else:
                ax.scatter(embedding[cluster_indices[i]][:, 0], embedding[cluster_indices[i]][:, 1], label=f'C{c_idx+1}', alpha=1, color=color)
                c_idx += 1

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
        ax.grid(False)
        
        if class_label_map is not None:
            ax.set_title(f"Class: {class_label_map[class_label]}")
        else: 
            ax.set_title(f"Class {class_label}")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Calculate highest and lowest values along both dimensions
        x_min, x_max = np.min(embedding[:, 0]), np.max(embedding[:, 0])
        y_min, y_max = np.min(embedding[:, 1]), np.max(embedding[:, 1])

        margin_percentage = 0.2
        # Calculate the desired margins
        x_margin = margin_percentage * (x_max - x_min)
        y_margin = margin_percentage * (y_max - y_min)

        # Set the x and y limits with margins
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # legend = ax.legend(loc='upper right', ncols=len(unique_labels)//2)
        legend = ax.legend(ncols=len(unique_labels)//2)

        legend.get_frame().set_alpha(0.4)  # Set the transparency to 0

# -----------------------------------------------------------------------------------------

def plot_cluster_subplot(embedding, cluster_indices, unique_labels, class_label, ax, title="", class_label_map=None, show_xlabel=True, show_ylabel=True, show_legend=False, cluster_color_map=None):

    c_idx = 0

    for i, label in enumerate(unique_labels):

        if label == -1:
            label = "NC"
        else:
            label = str(label+1)

        color = cluster_color_map[label]

        if label == -1:
            ax.scatter(embedding[cluster_indices[i]][:, 0], embedding[cluster_indices[i]][:, 1], label='Noise', alpha=0.25, color=color)
        else:
            ax.scatter(embedding[cluster_indices[i]][:, 0], embedding[cluster_indices[i]][:, 1], label=f'Cluster {c_idx+1}', alpha=0.5, color=color)
            c_idx += 1

    if show_xlabel:
        ax.set_xlabel('Dimension 1')
    
    if show_ylabel:
        ax.set_ylabel('Dimension 2')
    
    ax.grid(False)
    
    ax.set_title(title)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Calculate highest and lowest values along both dimensions
    x_min, x_max = np.min(embedding[:, 0]), np.max(embedding[:, 0])
    y_min, y_max = np.min(embedding[:, 1]), np.max(embedding[:, 1])

    margin_percentage = 0.2
    # Calculate the desired margins
    x_margin = margin_percentage * (x_max - x_min)
    y_margin = margin_percentage * (y_max - y_min)

    # Set the x and y limits with margins
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    if show_legend:
        ax.legend(loc="upper right", fontsize="small")

# -----------------------------------------------------------------------------------------

def find_correct_vs_incorrect(dh, CV_params, mode, subject="all"):

    subj_models = []

    num_time_points = 384

    for s in range(dh.num_subjects):
        model = train_utils.load_model(dh, CV_params, num_chn=len(dh.channel_names), num_data_points=num_time_points,
                                                            subject=s, trainings_mode=mode, return_dict=False, show_errors=True)
        subj_models.append(model)

    correct_list = []

    if subject == 'all':
        subjects = [subj for subj in range(0, dh.num_subjects)]
    else:
        subjects = [subject]

    for subj in subjects:

        X_subj, y_subj, _ = dh.get_transformed_data(subj)
        X_subj = torch.tensor(X_subj, requires_grad=True).float()

        y_subj = torch.from_numpy(np.array(y_subj)).type(torch.cuda.LongTensor)

        if torch.cuda.is_available():
            X_subj = X_subj.type(torch.cuda.FloatTensor)

        outputs = model(X_subj)

        _, predicted = torch.max(outputs.data, 1)
        
        correct = predicted == y_subj
        correct_list += list(correct.cpu().detach().numpy())

    return np.array(correct_list)

# -----------------------------------------------------------------------------------------

def get_samples_by_class(X, y, class_label, reversed_classes=False):

    if class_label != "both":
        if reversed_classes:
            class_label_ = 0 if class_label == 1 else 1 
        else:
            class_label_ = class_label
        
        indices = y == class_label_

        X = X[indices]
        y = y[indices]

    return X, y, indices

# -----------------------------------------------------------------------------------------

def get_correct_or_incorrect_samples(X, y, dh, CV_params, mode, subj='all', correct=True):

    correct_list = find_correct_vs_incorrect(dh, CV_params, mode, subject=subj)
    indices = np.where(correct_list == correct)

    X = X[indices]
    y = y[indices]

    return X, y, indices

# -----------------------------------------------------------------------------------------

def get_data(dh, select_correct, select_samples_by_class, class_label, CV_params, reversed_classes, use_which_data='train-test', subj=0, composite=None):
   
    if use_which_data == "train-test":
        X, y, sample_subject_map = pool_data(dh, CV_params, freq_domain=False, composite=composite, batch_size=128, return_labels=True)
    elif use_which_data == "train":
        X, y, sample_subject_map = pool_data(dh, CV_params, freq_domain=False, composite=composite, batch_size=128, return_labels=True, exclude=[subj])
    else:
        X, y, _ = dh.get_transformed_data(subj)    
        sample_subject_map = [subj for _ in range(len(y))]

    sample_subject_map = np.array(sample_subject_map)

    y = np.array(y)
    X = torch.from_numpy(X).type(torch.cuda.FloatTensor)

    if select_correct is not None:
        X, y, indices = get_correct_or_incorrect_samples(X, y, dh, CV_params, subj=subj, correct=select_correct)
        sample_subject_map = sample_subject_map[indices] 

    if select_samples_by_class:
        X, y, indices = get_samples_by_class(X, y, class_label, reversed_classes=reversed_classes)
        sample_subject_map = sample_subject_map[indices]

    return X, y, sample_subject_map

# -----------------------------------------------------------------------------------------

def pool_data(dh, CV_params, freq_domain=False, composite=None, return_labels=False, batch_size=512, exclude=None):

    if exclude is None:
        exclude = []

    data = []

    conditions = [{'y':[0,1]}]
    labels = []

    sample_subject_map = []

    for subj in range(dh.num_subjects):

        if subj in exclude:
            # print("pool_data: skip subject", subj)
            continue

        X_subj, y, _ = dh.get_transformed_data(subj)
        X_subj = torch.tensor(X_subj, requires_grad=True).float()

        if not isinstance(y, list):
            y = list(y)
        labels += y

        sample_subject_map += [subj for _ in range(len(y))]

        if torch.cuda.is_available():
            X_subj = X_subj.type(torch.cuda.FloatTensor)

        if freq_domain:            
            model = train_utils.load_model(dh, CV_params, X_subj.shape[-2], X_subj.shape[-1], mode, subj, return_dict=False)
            attribution = CondAttribution(model, no_param_grad=True, sum_over_channels=False)

            R_time = torch.zeros_like(X_subj)

            for b in range(0, X_subj.shape[0], batch_size):
                R_time[b:b+batch_size], _, _, _ = attribution(X_subj[b:b+batch_size], conditions, composite) 
            del _

            dftlrp = dft_lrp.DFTLRP(X_subj.shape[-1], leverage_symmetry=True, precision=32, create_stdft=False, create_inverse=False, cuda=torch.cuda.is_available())
          
            R_time = R_time.cpu().detach().numpy()

            a,b,_ = X_subj.shape
            X_subj_freq = np.empty((a,b,193))

            for b in range(0, X_subj.shape[0], batch_size):
                X_subj_freq[b:b+batch_size], _ = dftlrp.dft_lrp(R_time[b:b+batch_size], X_subj[b:b+batch_size], real=False, short_time=False)
            del _

            X_subj = np.abs(X_subj_freq)

        if torch.is_tensor(X_subj):
            X_subj = X_subj.cpu().detach().numpy()

        data.append(X_subj)

    data = np.concatenate(data, axis=0)

    if return_labels:
        return data, labels, sample_subject_map
    else:
        return data, sample_subject_map
    
# -------------------------------------------------------------------------------------