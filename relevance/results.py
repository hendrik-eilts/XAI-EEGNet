import os
import mne
import torch
import joblib
import matplotlib 
import numpy as np
import matplotlib.pyplot as plt
import relevance.utils as rel_utils
import training.utils as train_utils
import matplotlib.patches as mpatches

from matplotlib.colors import ListedColormap

from copy import copy
from scipy.spatial.distance import pdist
from mne_icalabel import label_components
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import relevance.utils as utils
import matplotlib.pyplot as plt
from training import utils as train_utils
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
from mne.filter import filter_data
from matplotlib.colors import LinearSegmentedColormap


class Results:

    def __init__(self, base_result_path, CV_params, num_folds, sampling_rate, class_labels, layer, num_filter, select_samples_by_class,
                 use_which_data, num_selected_samples, ds_name, reversed_classes, select_correct, channel_names, 
                 use_what_for_similarity, testing=False):

        self.base_result_path = base_result_path
        self.use_what_for_similarity = use_what_for_similarity
        self.sampling_rate = sampling_rate
        self.class_labels = class_labels
        self.layer = layer
        self.num_filter = num_filter
        self.num_CVs = len(CV_params)
        self.CV_params = CV_params
        self.num_folds = num_folds
        self.select_samples_by_class = select_samples_by_class
        self.use_which_data = use_which_data
        self.num_selected_samples = num_selected_samples
        self.ds_name = ds_name
        self.reversed_classes = reversed_classes
        self.select_correct = select_correct
        self.channel_names = channel_names
        self.testing = testing

        self.correlation_dict_c0_R = None
        self.correlation_dict_c1_R = None

        self.correlation_dict_c0_X = None
        self.correlation_dict_c1_X = None

        self.data = {}
        self.cluster_data = {}
        self.ica_components = {}

        self.cluster_keywords = ["embedding", "indices", "labels", "unique_labels", "unique_labels_plotting", "indices_plotting"]
        self.data_keywords = ["X_sim", "X_time_files", "X_freq_files", "R_time_files", "R_freq_files", "y_files"] 
        
        for label in self.class_labels:
            self.ica_components[label] = {}

            for cv_idx in range(self.num_CVs):
                self.ica_components[label][cv_idx] = {}

        for kw in self.cluster_keywords:
            self.cluster_data[kw] = {}

            for label in self.class_labels:
                self.cluster_data[kw][label] = {}

                for cv_idx in range(self.num_CVs):
                    self.cluster_data[kw][label][cv_idx] = []

        for kw in self.data_keywords:
            self.data[kw] = {}

            for label in self.class_labels:
                self.data[kw][label] = {}

                for cv_idx in range(self.num_CVs):
                    self.data[kw][label][cv_idx] = {}

                    for fold_idx in range(self.num_folds): 
                        self.data[kw][label][cv_idx][fold_idx] = []

        self._prepare_paths()
        self._collect_data()
        
    # ---------------------------------------------------------------------

    def set_data_class_cv_fold(self, data, data_identifier, class_label, cv_idx, fold_idx):
        self.data[data_identifier][class_label][cv_idx][fold_idx] = data

    # ---------------------------------------------------------------------

    def get_data_class_cv_fold(self, data_identifier, class_label, cv_idx, fold_idx):
            return self.data[data_identifier][class_label][cv_idx][fold_idx]

    # ---------------------------------------------------------------------

    def set_data_class_cv(self, data, data_identifier, class_label, cv_idx):

        if data_identifier in self.cluster_keywords:
            self.cluster_data[data_identifier][class_label][cv_idx] = data
        else:
            self.data[data_identifier][class_label][cv_idx] = data

    # ---------------------------------------------------------------------

    def get_data_class_cv(self, data_identifier, class_label, cv_idx):

        if data_identifier in self.cluster_keywords:
            return self.cluster_data[data_identifier][class_label][cv_idx]
        else:
            folds = []

            for fold_idx in range(self.num_folds):
                data = self.data[data_identifier][class_label][cv_idx][fold_idx]
                folds += data

            return folds

    # ---------------------------------------------------------------------
                
    def _get_data_of_cluster(self, data_identifier, class_label, cv_idx, cluster_idx):

        data = self.get_data_class_cv(data_identifier, class_label, cv_idx)

        indices = self.get_data_class_cv("indices", class_label, cv_idx)
        unique_labels = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))

        if cluster_idx in unique_labels:
            index = unique_labels.index(cluster_idx)
        else:
            print("not a valid cluster index:", cluster_idx, ", valid: ", unique_labels)
            return None

        indices = indices[index]
        data = [data[i] for i in indices]

        return data

    # ---------------------------------------------------------------------

    def _get_num_filter_per_subject_of_cluster(self, class_label, cv_idx, cluster_idx):

        indices = self.get_data_class_cv("indices", class_label, cv_idx)
        unique_labels = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))

        if cluster_idx in unique_labels:
            index = unique_labels.index(cluster_idx)
        else:
            print("not a valid cluster index:", cluster_idx, ", valid: ", unique_labels)
            return None

        indices = indices[index]

        filter_person_relation = []

        for fold_idx in range(self.num_folds):
            for _ in range(self.num_filter):
                filter_person_relation.append(fold_idx)

        filter_person_relation = [filter_person_relation[i] for i in indices]

        num_filter_per_subject = [0 for _ in range(self.num_folds)]
        for p in filter_person_relation:
            num_filter_per_subject[p] += 1

        return num_filter_per_subject, filter_person_relation

    # ---------------------------------------------------------------------
                
    def get_samples_of_cluster(self, class_label, cv_idx, cluster_idx, time_domain=True, relevance=False):

        data_identifier = "_time_files" if time_domain else "_freq_files"
        data_identifier = "R" + data_identifier if relevance else "X" + data_identifier

        filenames_X = self._get_data_of_cluster(data_identifier, class_label, cv_idx, cluster_idx)
        filenames_y = self._get_data_of_cluster("y_files", class_label, cv_idx, cluster_idx)

        all_samples = []
        all_labels = []

        for fn_X, fn_y in zip(filenames_X, filenames_y):

            samples = joblib.load(fn_X)
            labels = joblib.load(fn_y)

            all_samples.append(samples)
            all_labels += list(labels)

        all_samples = np.concatenate(all_samples, axis=0)

        return all_samples, all_labels
        
    # ---------------------------------------------------------------------     

    def compute_correlations(self, CV_params, calc_correlations_new=True, discard_negative=True, use_relevance=True, use_X=True, plot=False):

        if use_relevance:
            if self.correlation_dict_c0_R is None or calc_correlations_new:
                print("Compute correlations for class 0")
                self.correlation_dict_c0_R = self._calc_correlations(class_label=0, CV_params=CV_params, plot=plot, discard_negative=discard_negative, use_relevance=True)

            if self.correlation_dict_c1_R is None or calc_correlations_new:
                print("Compute correlations for class 1")
                self.correlation_dict_c1_R = self._calc_correlations(class_label=1, CV_params=CV_params, plot=plot, discard_negative=discard_negative, use_relevance=True)

        if use_X:
            if self.correlation_dict_c0_X is None or calc_correlations_new:
                print("Compute correlations for class 0")
                self.correlation_dict_c0_X = self._calc_correlations(class_label=0, CV_params=CV_params, plot=plot, discard_negative=discard_negative, use_relevance=False)

            if self.correlation_dict_c1_X is None or calc_correlations_new:
                print("Compute correlations for class 1")
                self.correlation_dict_c1_X = self._calc_correlations(class_label=1, CV_params=CV_params, plot=plot, discard_negative=discard_negative, use_relevance=False)

    # ---------------------------------------------------------------------

    @staticmethod
    def assemble_path(CV_params, base_result_path, num_samples_to_select, ds_name, layer, class_label,  
                    select_samples_by_class, reversed_classes, select_correct,  
                    filter_sim_sample_based, use_which_data, testing=False):
        
        nr = CV_params['nr']

        if testing:
            testing_str = "_testing"
        else: 
            testing_str = ""

        if select_samples_by_class:
            sample_select_str = "_samplesOfClass"
        else:
            sample_select_str = ""

        use_which_data = "_" + use_which_data

        if filter_sim_sample_based:
            filter_sim_sample_based_str = "_sampleBased"
        else:
            filter_sim_sample_based_str = ""

        if select_correct == True:
            select_correct_str = "_correct"
        elif select_correct == False:
            select_correct_str = "_incorrect"
        else:
            select_correct_str = ""

        if reversed_classes:
            sample_select_str += "Reversed"

        if class_label == "both":
            sample_select_str = ""

        if "seed" in CV_params:
            seed_str = f"_seed{CV_params['seed']}"
        else:
            seed_str = ""

        path_base = os.path.join(base_result_path, f"results_relevance_{num_samples_to_select}{use_which_data}{testing_str}", ds_name, layer)

        path_for_cv = os.path.join(path_base, f"class_{class_label}{sample_select_str}" +  
                        f"{select_correct_str}{filter_sim_sample_based_str}{seed_str}_{nr}")
        
        return path_base, path_for_cv

    # ---------------------------------------------------------------------

    def plot_distribution_of_selected_samples(self):

        for cv_idx in range(self.num_CVs):

            for class_label in self.class_labels:

                suptitle = f"Class {class_label} - CV iteration {cv_idx}"
                path = self.paths_to_CV_iteration[class_label][cv_idx]
                # path_get_data_dict[class_label][cv_idx]

                selected_samples_per_subject = joblib.load(os.path.join(path, "selected_samples_per_subject"))

                x_labels = []
                num_samples_per_subj = []
                num_unique_per_subj = []

                for key, value in selected_samples_per_subject.items():
                    x_labels.append(key)

                    num_samples_per_subj.append(len(value))
                    num_unique_per_subj.append(len(set(value)))

                width_per_person = 6/10
                width = self.num_folds * width_per_person 
                print(width_per_person)
                _, axes = plt.subplots(1,2, figsize=(width,2.5), sharey="row")

                for i in range(0, 2):
                    ax = axes[i]

                    if i == 0: 
                        values = num_samples_per_subj
                        title = "Num selected samples"
                    else: 
                        values = num_unique_per_subj
                        title = "Num unique samples selected"

                    ax.bar(x_labels, values)
                    ax.set_xticks(x_labels)
                    ax.set_xlabel("Test Subjects")
                    ax.set_ylabel("Amount of samples")
                    ax.set_title(title)

                plt.suptitle(suptitle, y=1.1)
                plt.show()

    # ---------------------------------------------------------------------

    def plot_single_filter(self):

        tmp = joblib.load(os.path.join(self.paths_to_CV_iteration[0][0], "total_rel_per_filter_final"))
        num_total = len(tmp)
        num_filters_per_model = num_total // self.num_folds
        del tmp

        folds = [i for i in range(self.num_folds)]

        include_cvs = [0]
        include_folds = [2]

        for class_label in self.class_labels:
            print("-"*20, class_label, "-"*20)

            for cv in self.CV_params:
                cv_nr = cv['nr']

                if cv_nr not in include_cvs:
                    continue

                for fold in folds:
                    if fold not in include_folds:
                        continue

                    path_store_data = self.paths_to_CV_iteration[class_label][cv_nr]

                    titles = []
                    X_freq_list = []
                    R_freq_list = []
                    rel_per_filter = []

                    for f in range(num_filters_per_model):

                        X_freq = joblib.load(os.path.join(path_store_data, f"X_freq_select_{fold}_{f}"))
                        R_freq = joblib.load(os.path.join(path_store_data, f"R_freq_select_{fold}_{f}"))

                        X_freq = np.abs(X_freq)**2
                        discard_negative_rel = True

                        if discard_negative_rel:
                            R_freq[R_freq < 0] = 0

                        title = f"CV: {cv['nr']}, Fold: {fold}, Filter {f}"

                        titles.append(title)
                        X_freq_list.append(X_freq)
                        R_freq_list.append(R_freq)

                        rel_per_filter.append(np.sum(R_freq))

                    rel_per_filter = np.array(rel_per_filter)

                    indices = np.flip(np.argsort(rel_per_filter))

                    rel_per_filter = rel_per_filter[indices]
                    titles = [titles[i] for i in indices]

                    x_labels = [str(i) for i in indices]

                    plt.bar(x_labels, rel_per_filter)
                    plt.xlabel('Filters')
                    plt.ylabel('Relevance')
                    
                    plt.title(f"Class {class_label}, CV {cv['nr']}, Fold {fold}")

                    plt.show()

                    for f in range(0, 3): # num_filters_per_model):

                        X_freq = X_freq_list[f]
                        R_freq = R_freq_list[f]
                        title = titles[f]

                        bands = [(0,4,"delta"), (4,8,"theta"), (8,12,"alpha"), (12,30,"beta"), (30,60,"gamma")]

                        fig, axes = plt.subplots(2, len(bands), figsize=(6, 4)) 


                        _, X_freq = self._plot_freq_bands_topo(X_freq, self.channel_names, bands, axes[0], title=title, global_scale=True, norm=False, log_scale=False, scale_bands=False, cmap="Reds", 
                                                    label="", colorbar_offset=(0.0,0.5), vlim=None, mean=False, negative_lim=False)

                        _, R_freq = self._plot_freq_bands_topo(R_freq, self.channel_names, bands, axes[1], title="", global_scale=True, norm=False, log_scale=False, scale_bands=False, cmap="Reds", 
                                                    label="", colorbar_offset=(0.0,0.2), vlim=None, mean=False, negative_lim=False)

                        # self._plot_freq_bands_topo_combined(X_freq, R_freq, self.channel_names, bands, axes[2], title="", global_scale=True, norm=False, log_scale=False, scale_bands=False, cmap="Reds", 
                        #                             label="", colorbar_offset=(0.0,0.2), vlim=None, mean=False, negative_lim=False)

                        plt.show()

    # -------------------------------------------------------------------------------------------------------

    def plot_cluster_signal(self, class_label, cv_idx, 
                            title=None, exclude_cluster=[], sig_log_scale=False, 
                            sig_global_scale=True, rel_global_scale=True, psd=True, sig_scale_bands=False, rel_scale_bands=False, sig_norm=True, rel_norm=True,
                            cluster_order=None, rel_common_scale=False, sig_common_scale=False, sig_mean=True, rel_mean=False, rel_vlim=None, sig_vlim=None, negative_lim=False,
                            rel_cmap="bwr", remove_neg=False, scale_per_band=False, bands=None, scale_1_div_f=True, plot_ideal_topo=False):

        title_was_none = title is None

        bands = [(0,4,"delta"), (4,8,"theta"), (8,12,"alpha"), (12,30,"beta"), (30,60,"gamma")]

        scale_factors = [] # 2, 6, 10, 21, 45]

        first_scalar = (bands[0][0]+bands[0][1])/2

        for band in bands:
            val = (band[0]+band[1])/(2*first_scalar)
            scale_factors.append(val)
        
        cidx_keys = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))

        if exclude_cluster is None:
            exclude_cluster = []

        for clstr in exclude_cluster:
            if clstr in cidx_keys:
                cidx_keys.remove(clstr)

        if cluster_order is not None:
            cidx_keys = np.array(cidx_keys)[cluster_order]

        for cidx_key in cidx_keys:

            print(cidx_key)

            X_freq, y = self.get_samples_of_cluster(class_label, cv_idx, cidx_key, time_domain=False)
            R_freq, _ = self.get_samples_of_cluster(class_label, cv_idx, cidx_key, time_domain=False, relevance=True)

            if psd:
                X_freq = X_freq**2

            if title_was_none:
                if cidx_key == "NC":
                    title_ = cidx_key

                elif cidx_key == "all": 
                    title_ = "All Filters"
                else:
                    title_ = f"Cluster {cidx_key}" 
            else:
                title_ = title

            # fig, axes = plt.subplots(3, len(bands), figsize=(12, 6)) 
                
            if plot_ideal_topo:
                fig, axes = plt.subplots(3, len(bands), figsize=(9, 5)) 
            else:
                fig, axes = plt.subplots(2, len(bands), figsize=(6, 5))
            

            if sig_log_scale:
                label = "log_10(PSD)"
            else:
                label = "PSD"

            colorbar_offset = (0.1, 0)

            s_vlim, X_freq = self._plot_freq_bands_topo(X_freq, self.channel_names, ax=axes[0], global_scale=sig_global_scale, log_scale=sig_log_scale, 
                                        scale_bands=sig_scale_bands, bands=bands, norm=sig_norm, title=title_, label=label, colorbar_offset=(0.0,0.5), 
                                        vlim=sig_vlim, mean=sig_mean, negative_lim=False, cmap="Reds", scale_per_band=scale_per_band, scale_1_div_f=scale_1_div_f, 
                                        scale_factors=scale_factors)
            
            colorbar_offset = (0, 0.25)
            
            r_vlim, R_freq = self._plot_freq_bands_topo(R_freq, self.channel_names, ax=axes[1], global_scale=rel_global_scale, log_scale=False, 
                                        scale_bands=rel_scale_bands, bands=bands, norm=rel_norm, title=title_, label="Relevance", colorbar_offset=colorbar_offset,
                                        vlim=rel_vlim, mean=rel_mean, negative_lim=negative_lim, cmap=rel_cmap, scale_per_band=False, scale_1_div_f=scale_1_div_f, 
                                        scale_factors=scale_factors, remove_neg=remove_neg)

            if plot_ideal_topo:
                self._plot_freq_bands_topo_combined(X_freq, R_freq, self.channel_names, ax=axes[2], bands=bands, title=title_, label="Ideal", colorbar_offset=(0.0, 0.2))

            plt.show()
            plt.tight_layout()

    # ---------------------------------------------------------------------

    def _plot_freq_bands_topo_combined(self, X_freq, R_freq, ch_names, bands, ax, title="", cmap=None, label="", colorbar_offset=[0,0]):

        X_all = []

        for band_idx in range(len(X_freq)):

            ax_band = ax[band_idx]

            X_band = np.empty_like(X_freq[band_idx])

            for chn in range(X_freq[band_idx].shape[0]):

                X_val = X_freq[band_idx][chn]
                R_val = R_freq[band_idx][chn]

                if R_val > 0:
                    res = X_val * R_val
                else:
                    res = (1-X_val) * np.abs(R_val)

                X_band[chn] = res
            
            X_all.append(X_band)

            self._plot_topography(X_band, 120, ch_names, title="", label="", plot=True, sp_size=2, show_ch_names=False, vlim=(0,1), ax=ax_band, cmap=cmap)

            ax_band.set_title("")

            # if i == len(bands)-1 and global_scale:
            #     if suptitle:
            #         cax = plt.axes([1.0 + colorbar_offset[0], 0.0 + colorbar_offset[1], 0.01, 0.2]) 
            #     else:
            #         cax = plt.axes([1.0 + colorbar_offset[0], 0.0 + colorbar_offset[1], 0.01, 0.2]) 
            #     plt.colorbar(im, cax=cax)


    # ---------------------------------------------------------------------
                    
    def plot_cluster_relevance_subplot(self, class_label, cv_idx, ax, exclude_cluster=[], colors=None, show_xlabel=True, bar_width=0.5):
        
        rels = []
        abs_rels = []

        cidx_keys = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))

        for clstr in exclude_cluster:
            if clstr in cidx_keys:
                cidx_keys.remove(clstr)

        for cidx_key in cidx_keys:
            rel = self._calc_cluster_relevance(class_label, cv_idx, cidx_key)
            rels.append(rel)
            abs_rels.append(np.abs(rel))

        labels = cidx_keys 
        x = np.arange(len(labels))

        rels = np.array(rels)
        abs_rels = np.array(abs_rels)

        data = abs_rels

        for i in range(len(data)):
            color, alpha = colors[i]
            ax.bar(x[i], data[i], color=color, alpha=alpha, width=bar_width)

        ax.set_xlim(left=-0.5, right=len(x)-0.5)

        if show_xlabel:
            ax.set_xlabel('Cluster')

        ax.set_ylabel('Mean Relevance')
        ax.set_xticks(x, labels)
        
    # ---------------------------------------------------------------------
    
    def plot_cluster_coherence_subplot(self, class_label, cv_idx, ax, colors, exclude_cluster=[], show_xlabel=True, bar_width=0.5):
        
        coherence_per_cluster = []

        cidx_keys = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))
        
        for clstr in exclude_cluster:
            print(clstr, exclude_cluster)
            if clstr in cidx_keys:
                cidx_keys.remove(clstr)

        for cidx_key in cidx_keys:
            coh = self._calc_cluster_coherence(class_label, cv_idx, cidx_key)
            coherence_per_cluster.append(coh)

        labels = cidx_keys
        coherence_list = []

        for coherence in coherence_per_cluster:
            coherence_list.append(coherence)

        coherence_list = np.array(coherence_list)
        labels = np.array(labels)

        x = np.arange(len(labels))

        ax.set_xlim(left=-0.5, right=len(x)-0.5)

        for idx, (xi, cl) in enumerate(zip(x, coherence_list)):
            color, alpha = colors[idx]
            ax.bar(xi, cl, width=bar_width, color=color, alpha=alpha)

        # ax.bar(x, coherence_list, width=0.3, color='blue', alpha=0.5)
        ax.set_ylabel("Coherence")
        
        if show_xlabel:
            ax.set_xlabel("Cluster")
        ax.set_xticks(x, labels)
        
    # ---------------------------------------------------------------------
   
    def plot_cluster_amount_of_filter_subplot(self, class_label, cv_idx, ax, colors, exclude_cluster=[], show_xlabel=True, bar_width=0.5):
        
        num_filter = []

        cidx_keys = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))

        for clstr in exclude_cluster:
            if clstr in cidx_keys:
                cidx_keys.remove(clstr)

        rotation_angles = []
        for cidx_key in cidx_keys:
            rotation_angles.append(0)
            nf, _ = self._get_num_filter_per_subject_of_cluster(class_label, cv_idx, cidx_key)
            nf = np.sum(nf)
            num_filter.append(nf)

        num_filter = np.array(num_filter)

        labels = cidx_keys 
        x = np.arange(len(labels))

        for i in range(len(num_filter)):
            color, alpha = colors[i]

            ax.bar(x[i], num_filter[i], color=color, alpha=alpha, width=bar_width)

        if show_xlabel:
            ax.set_xlabel("Cluster")
    
        ax.set_ylabel("Amount of filters")
        ax.set_xticks(x, labels)
        
    # ---------------------------------------------------------------------

    def plot_cluster_class_label_distribution(self, class_label, cv_idx, title="", normalize=True, exclude_cluster=[]):

        data  = []

        cidx_keys = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))

        for clstr in exclude_cluster:
            if clstr in cidx_keys:
                cidx_keys.remove(clstr)

        for idx in range(len(cidx_keys)):
            
            cidx_key = cidx_keys[idx]

            _, y = self.get_samples_of_cluster(class_label, cv_idx, cidx_key, time_domain=True, relevance=False)

            if normalize:
                norm = len(y)
            else:
                norm = 1

            num_0 = (len(y)-np.sum(y)) / norm
            num_1 = np.sum(y) / norm

            data.append([num_0, num_1])

        data_array = np.array(data)
        num_bars = len(data[0])
        bar_width = 0.30
        bar_distance = 0.25

        group_positions = np.arange(len(data))*1.5

        fig, ax = plt.subplots(figsize=(3,3))

        colors = ["blue", "red"]

        for i in range(num_bars):
            bar_positions = group_positions + i * (bar_width + bar_distance)
            ax.bar(bar_positions, data_array[:, i], bar_width, label=f'Class {i}', color=colors[i])

        ax.set_xticks(group_positions + ((num_bars - 1) * (bar_width + bar_distance)) / 2)

        cluster_labels = cidx_keys 
        
        ax.set_xticklabels(cluster_labels, rotation=0)

        # Set labels and title
        ax.set_xlabel('Clusters')
        if normalize:
            ax.set_ylabel('Proportion of class labels')
            ax.set_ylim((0, 1))
        else:
            ax.set_ylabel('Number of labels')

        ax.set_title('Distribution of class labels in clusters')

        ax.legend()
        
        plt.show()

    # ---------------------------------------------------------------------

    def plot_cluster_distribution_stacked_subplot(self, class_label, cv_idx, ax, cluster_color_map=None, exclude_cluster=[], show_xticks=True):

        cidx_keys = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))

        for clstr in exclude_cluster:
            if clstr in cidx_keys:
                cidx_keys.remove(clstr)

        colors = []

        list_of_lists = []
        legend_labels = []
        label_colors = {}

        for idx, cluster_idx in enumerate(cidx_keys):

            colors.append(cluster_color_map[cluster_idx])
            
            filenames = self._get_data_of_cluster("X_time_files", class_label, cv_idx, cluster_idx)

            num_filter_per_fold = [0]*self.num_folds

            for fn in filenames:
                fold = fn.split("_")[-2]
                num_filter_per_fold[int(fold)] += 1

            list_of_lists.append(num_filter_per_fold)
            
            if cluster_idx == "NC":
                legend_label = cluster_idx
            else:
                legend_label = f"C{cluster_idx}"
            
            legend_labels.append(legend_label)
            label_colors[legend_label] = colors[idx]

        x = np.arange(self.num_folds)
        labels = [str(xi+1) for xi in x]

        bottom = np.zeros(self.num_folds)  # Bottom position for each bar, initialized to zeros

        for i, list_ in enumerate(list_of_lists):
            ax.bar(x, list_, bottom=bottom, color=colors[i], alpha=1, label=legend_labels[i])
            bottom += np.array(list_)  # Update bottom positions for the next stacked bar

        ax.set_xlim(left=-1, right=self.num_folds)

        ax.legend(title='')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        if not show_xticks:
            ax.set_xticks([])
            ax.set_xlabel("")
        else:
            ax.set_xticks(x, labels)
            ax.set_xlabel("Models")

        ax.set_ylabel('#Filters')

    # ---------------------------------------------------------------------

    def plot_cluster_distribution_stacked(self, class_label, cv_idx, title="", ylim=None, exclude_cluster=[], figsize=(7, 1), cluster_color_map=None, ax=None):

        cidx_keys = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))

        for clstr in exclude_cluster:
            if clstr in cidx_keys:
                cidx_keys.remove(clstr)

        colors = []

        list_of_lists = []
        legend_labels = []
        label_colors = {}

        for idx, cluster_idx in enumerate(cidx_keys):

            colors.append(cluster_color_map[cluster_idx])
            
            filenames = self._get_data_of_cluster("X_time_files", class_label, cv_idx, cluster_idx)

            num_filter_per_fold = [0]*self.num_folds

            for fn in filenames:
                fold = fn.split("_")[-2]
                num_filter_per_fold[int(fold)] += 1

            list_of_lists.append(num_filter_per_fold)
            legend_labels.append(f"Cluster {cluster_idx}")
            label_colors[f"Cluster {cluster_idx}"] = colors[idx]

        x = np.arange(self.num_folds)
        labels = [str(xi) for xi in x]

        _, ax = plt.subplots(1,1, figsize=figsize)

        bottom = np.zeros(self.num_folds)  # Bottom position for each bar, initialized to zeros

        for i, list_ in enumerate(list_of_lists):
            ax.bar(x, list_, bottom=bottom, color=colors[i], alpha=0.5, label=legend_labels[i])
            bottom += np.array(list_)  # Update bottom positions for the next stacked bar

        ax.legend(title='')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel("Models")
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_ylabel('Amount of filters')
        
        if ylim is not None:
            ax.set_ylim(*ylim)

        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------

    def plot_cluster_distribution(self, class_label, cv_idx, title="", ylim=None, exclude_cluster=[], figsize=(7, 3), cluster_color_map=None):

        cidx_keys = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))

        for clstr in exclude_cluster:
            if clstr in cidx_keys:
                cidx_keys.remove(clstr)

        list_of_lists = []
        legend_labels = []
        label_colors = {}

        colors = []

        for idx, cluster_idx in enumerate(cidx_keys):
            
            filenames = self._get_data_of_cluster("X_time_files", class_label, cv_idx, cluster_idx)

            num_filter_per_fold = [0]*self.num_folds

            color = cluster_color_map[cluster_idx]

            for fn in filenames:
                fold = fn.split("_")[-2]
                num_filter_per_fold[int(fold)] += 1

            list_of_lists.append(num_filter_per_fold)
            colors.append(color)
            if cluster_idx != "NC":
                legend_labels.append(f"Cluster {cluster_idx}")
            else:
                legend_labels.append(f"{cluster_idx}")

            label_colors[f"Cluster {cluster_idx}"] = color

        # -----------------------------------------------------

        def create_colormap(color):
            cmap = LinearSegmentedColormap.from_list('custom_cmap', [(1, 1, 1), color])
            return cmap

        values = np.array(list_of_lists)

        # Create the colormaps for each row
        cmaps = [create_colormap(color) for color in colors]

        # Plot the values for each row using the corresponding colormap
        fig, axs = plt.subplots(len(values), 1, figsize=(12, len(values)*0.5), gridspec_kw = {'wspace':0, 'hspace':0})

        for i, (row_values, cmap) in enumerate(zip(values, cmaps)):

            heatmap = axs[i].imshow([row_values], cmap=cmap, aspect='auto')

            if i == len(values)-1:
                custom_labels = [str(v+1) for v in range(len(row_values))]
                axs[i].set_xticks(np.arange(len(row_values)))
                axs[i].set_xticklabels(custom_labels)
            else:
                axs[i].set_xticks([])

            axs[i].set_yticks([])

            axs[i].grid(False)

            # axs[i].set_title(f'Row {i+1}')
            # axs[i].set_yticks([])
            # axs[i].set_xticks(np.arange(len(row_values)))
    
            axs[i].set_ylabel(cidx_keys[i], rotation=0, labelpad=20, y=0.4)

            for y in range(row_values.shape[0]):
                
                val = row_values[y]

                if val == 0:
                    axs[i].text(y, 0, '0', color='red', ha='center', va='center')
                else:
                    if val <= 3:
                        axs[i].text(y, 0, f'{val}', color='gray', ha='center', va='center')

        plt.subplots_adjust(wspace=None, hspace=None)
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------

    def compute_ica(self, class_label, cv_idx, exclude_cluster=[], overwrite_if_exists=False, band=False):

        cluster_keys = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))

        for clstr in exclude_cluster:
            if clstr in cluster_keys:
                cluster_keys.remove(clstr)

        for cluster_idx in cluster_keys:
            
            compute_components = True

            cluster_idx_str = cluster_idx

            if band != False:
                cluster_idx_str = f"{cluster_idx}_{band}"

            if cluster_idx_str in self.ica_components[class_label][cv_idx].keys():
                if not overwrite_if_exists:
                    compute_components = False

            if compute_components:
                ica, labels, proba = self._calc_ica_on_cluster(class_label, cv_idx, cluster_idx, time_domain=True, band=band)
                self.ica_components[class_label][cv_idx][cluster_idx_str] = (ica, labels, proba)

    # ---------------------------------------------------------------------

    def plot_ica(self, class_label, cv_idx, exclude_cluster=[], figsize=(6,6), title="", time=True, band=False, ica_title_fontsize=None):

        info = mne.create_info(self.channel_names, self.sampling_rate, ch_types='eeg')
        cidx_keys = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))

        for clstr in exclude_cluster:
            if clstr in cidx_keys:
                cidx_keys.remove(clstr)

        for cluster_idx in cidx_keys:
            print(" "*5, "Cluster:", cluster_idx)
  
            cluster_idx_str = cluster_idx

            if not time:
                cluster_idx_str = f"{cluster_idx_str}_freq"

            if band is not None and band is not False:
                cluster_idx_str = f"{cluster_idx_str}_{band}"

            try:
                ica, labels, proba = self.ica_components[class_label][cv_idx][cluster_idx_str]
            except Exception as e:
                print(f"ICA components not computed for class label {class_label}, CV nr. {cv_idx}, Cluster {cluster_idx_str}")
                return 

            R_time, _ = self.get_samples_of_cluster(class_label, cv_idx, cluster_idx, time_domain=False, relevance=True)

            info = mne.create_info(self.channel_names, self.sampling_rate, ch_types='eeg')
            R_epochs = mne.EpochsArray(R_time, info)
            R_epochs.set_montage('standard_1020')  

            R_ica = ica.get_sources(R_epochs).get_data()
            R_ica = np.abs(R_ica)
            R_ica = np.sum(R_ica, axis=0)
            R_ica_plot = np.sum(R_ica, axis=1)
            R_ica_rank = np.abs(R_ica_plot)

            indices = np.flip(np.argsort(R_ica_rank))

            labels_sorted = [(i, labels[i]) for i in indices]
            rels_sorted_rank = R_ica_rank[indices]
            rels_sorted_plot = R_ica_plot[indices]

            rels_sorted_rank /= np.sum(rels_sorted_rank)

            count = 4
            rels_sorted_rank = rels_sorted_rank[:count]
            rels_sorted_rank *= 100

            # --------------------------------

            colors = []

            flip = False

            for rel in rels_sorted_plot:
                
                if flip:
                    if rel < 0:
                        colors.append('blue')
                    else:
                        colors.append('red')
                else:
                    colors.append("teal")

            def add_leading_zeros(id):
                id = str(id)
                if len(id) < 3:
                    id = (3-len(id))*"0" + id 
                return id

            labels = [f"{add_leading_zeros(id)}_{name}" for (id, name) in labels_sorted[:count]]

            # --------------------------------

            fig = plt.figure(figsize=figsize)

            gs = gridspec.GridSpec(1, 5) # , height_ratios=[3.5, 3.5])

            if band is None or band is False:
                title = f"Cluster {cluster_idx}"
            else:
                title = f"Cluster {cluster_idx} ({band})"


            plt.suptitle(title, y=1.55)

            ax1 = plt.subplot(gs[0, 0])  
            ax2 = plt.subplot(gs[0, 1])
            ax3 = plt.subplot(gs[0, 2])
            ax4 = plt.subplot(gs[0, 3])
            ax5 = plt.subplot(gs[0, 4])

            axes = [ax2, ax3, ax4, ax5]

            ax = ax1
            y = np.arange(len(labels))

            ax.xaxis.grid(False)
            ax.yaxis.grid(True, alpha=0.5)

            y = np.flip(y)
            rels_sorted_rank = np.flip(rels_sorted_rank)

            ax.barh(y, rels_sorted_rank, height=0.5, label='', align='center', color=colors, alpha=0.75) 
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Relevance (%)')
            # ax.set_title("ICA Components")
            ica.plot_components(picks=indices[:count], plot_std=True, sensors=False, axes=axes) 

            plt.show()

    # ---------------------------------------------------------------------     

    def compute_functional_groups(self, freq_signal, brain_regions=None, bands=None, only_positive=False, mean=True):

        # print(freq_signal.shape)
        # (32, 193)

        if brain_regions is None:
            brain_regions = ["left temporal", "right temporal", "left parietal", "right parietal", 
                            "left occipital", "right occipital", "left central", "right central", 
                            "left frontal", "right frontal"]
        if bands is None:
            bands = [(0,4,"delta"), (4,8,"theta"), (8,12,"alpha"), (12,30,"beta"), (30,60,"gamma")] 

        if only_positive:
            freq_signal[freq_signal < 0] = 0

        f = 3
        values = []

        for low, high, band in bands:
            low, high = int(low*f), int(high*f) 

            for br in brain_regions:

                _, indices = rel_utils.get_channels_by_brain_region(self.channel_names, brain_region=br)
                rel_br = freq_signal[indices,:]

                rel_fb = rel_br[:, low:high]  # select freq bins

                if mean: 
                    rel_fb = np.mean(rel_fb)

                values.append(rel_fb)
        
        return values

    # ---------------------------------------------------------------------     


    def plot_cluster_functional_grouping2(self, class_label, cv_idx, brain_regions=None, bands=None, cluster_idx=0, 
                                         aggregation="mean", only_positive=False, abs=True, title="", axes=None, from_ideal_topo=False, use_X=False):

        cidx_keys = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))
        
        if cluster_idx not in cidx_keys:
            print(f"the specified cluster {cluster_idx} does not exist")
            return
        
        R_freq1, _ = self.get_samples_of_cluster(0, cv_idx, cluster_idx, time_domain=False, relevance=True)
        R_freq2, _ = self.get_samples_of_cluster(1, cv_idx, cluster_idx, time_domain=False, relevance=True)

        rels = {}

        if only_positive: 
            R_freq1[R_freq1 < 0] = 0
            R_freq2[R_freq2 < 0] = 0

        for low, high, band in bands:
            low, high = int(low*3), int(high*3) 
            
            rels[band] = []
            colors = []

            for (_, channels, color) in brain_regions:
                colors.append(color)

                indices = [i for i, ch_name in enumerate(self.channel_names) if ch_name in channels]
                rel_br1 = R_freq1[:,indices,:]
                rel_fb1 = rel_br1[:, :, low:high]  # select freq bins

                rel_br2 = R_freq2[:,indices,:]
                rel_fb2 = rel_br2[:, :, low:high]  # select freq bins
          
                if aggregation == "mean":
                    rel_fb1 = np.mean(rel_fb1)
                    rel_fb2 = np.mean(rel_fb2)

                elif aggregation == "max":
                    rel_fb1 = np.max(rel_fb1)
                    rel_fb2 = np.max(rel_fb2)

                elif aggregation == "sum":
                    rel_fb1 = np.sum(rel_fb1)
                    rel_fb2 = np.sum(rel_fb2)

                elif aggregation == "median":
                    rel_fb1 = np.median(rel_fb1)
                    rel_fb2 = np.median(rel_fb2)

                rel_fb = np.abs(rel_fb1 - rel_fb2)

                rels[band].append(rel_fb)

        for i, band in enumerate(bands):
            try:
                ax = axes[i]
            except:
                ax = axes
                
            for bidx, (region, _, color) in enumerate(brain_regions):
                ax.bar(region, rels[band[2]][bidx], color=color)

            ax.set_title(f"$\\{band[2]}$")
            if i == 0:
                ax.set_ylabel(f"{aggregation} relevance")
            ax.tick_params(axis='x', rotation=90)  # Adjust the rotation angle as needed
            ax.xaxis.grid(False)


    # -------------------------------------------------------------------------------------------------

    def plot_cluster_functional_grouping(self, class_label, cv_idx, brain_regions=None, bands=None, cluster_idx=0, 
                                         aggregation="mean", only_positive=False, abs=True, title="", axes=None, from_ideal_topo=False, use_X=False):

        cidx_keys = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))
        
        if cluster_idx not in cidx_keys:
            print(f"the specified cluster {cluster_idx} does not exist")
            return
        
        R_freq, _ = self.get_samples_of_cluster(class_label, cv_idx, cluster_idx, time_domain=False, relevance=True)
        X_freq, _ = self.get_samples_of_cluster(class_label, cv_idx, cluster_idx, time_domain=False, relevance=False)

        if from_ideal_topo:

            R_freq = R_freq / np.max(np.abs(R_freq))
            X_freq = X_freq / np.max(np.abs(X_freq))

            mask = R_freq > 0
            R_freq = np.where(mask, R_freq*X_freq, (1-X_freq)*np.abs(R_freq))

            print(np.min(R_freq), np.max(R_freq))

        if use_X:
            R_freq = X_freq

        if abs:
            R_freq = np.abs(R_freq)

        if only_positive: 
            R_freq[R_freq < 0] = 0

        total_relevance = np.mean((np.sum(R_freq)))

        rels = {}
        rels_unaggregated = {}

        for low, high, band in bands:
            low, high = int(low*3), int(high*3) 
            
            rels[band] = []
            rels_unaggregated[band] = []
            colors = []

            for (_, channels, color) in brain_regions:
                colors.append(color)
                indices = [i for i, ch_name in enumerate(self.channel_names) if ch_name in channels]
                rel_br = R_freq[:,indices,:]
                rel_fb = rel_br[:, :, low:high]  # select freq bins

                if aggregation == "mean":
                    rel_fb = np.mean(rel_fb)

                elif aggregation == "max":
                    rel_fb = np.max(rel_fb)

                elif aggregation == "sum":
                    rel_fb = np.sum(rel_fb)

                elif aggregation == "median":
                    rel_fb = np.median(rel_fb)

                # if abs:
                #     rel_fb = np.abs(rel_fb)
                
                # rel_fb /= total_relevance
                # rel_fb *= 100

                rels[band].append(rel_fb)

                # ------------
                rel_fb = rel_br[:, :, low:high]  # select freq bins
                rels_unaggregated[band].append(rel_fb)

        for i, band in enumerate(bands):
            try:
                ax = axes[i]
            except:
                ax = axes
                
            for bidx, (region, _, color) in enumerate(brain_regions):
                ax.bar(region, rels[band[2]][bidx], color=color)

            ax.set_title(f"$\\{band[2]}$")
            if i == 0:
                ax.set_ylabel(f"Mean Relevance")
            ax.tick_params(axis='x', rotation=90)  # Adjust the rotation angle as needed
            ax.xaxis.grid(False)

        return rels_unaggregated

    # ---------------------------------------------------------------------     

    def plot_correlation_barchart(self, ax=None, colors=None, xticks=None, use_relevance=True):

        if use_relevance:
            if self.correlation_dict_c0_R is None or self.correlation_dict_c1_R is None:
                print("Correlations on relevance have not been computed yet. Compute correlations by calling compute_correlation(..)")
                return
        else:
            if self.correlation_dict_c0_X is None or self.correlation_dict_c1_X is None:
                print("Correlations on relevance have not been computed yet. Compute correlations by calling compute_correlation(..)")
                return
            
        within_group_labels = ['Same', 'Different']

        if use_relevance:
            corr_dicts = [self.correlation_dict_c0_R, self.correlation_dict_c1_R]
        else:
            corr_dicts = [self.correlation_dict_c0_X, self.correlation_dict_c1_X]

        if xticks is None:
            outer_group_labels = []
        else:
            outer_group_labels = xticks

        data = []

        for corr_dict in corr_dicts:
            corrs = []
            corrs.append(np.mean(corr_dict['CV_x_CV_corr_list']))
            corrs.append(np.mean(corr_dict['Fold_x_Fold_corr_list']))
            data.append(corrs)

        # ---------------------

        data = np.array(data)

        num_bars = len(data[0])

        bar_width = 0.30
        bar_distance = 0.25
        group_positions = np.arange(len(data))*1.5

        if ax is None:
            fig, ax = plt.subplots(figsize=(3,3))

        if colors is None:
            colors = ["blue", "red"]

        for i in range(num_bars):
            label = within_group_labels[i]

            bar_positions = group_positions + i * (bar_width + bar_distance)
            bars = ax.bar(bar_positions, data[:, i], bar_width, label=label, color=colors[i])

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.0, f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_ylim(0, 1)
        ax.set_xticks(group_positions + ((num_bars - 1) * (bar_width + bar_distance)) / 2)

        # cluster_labels = cidx_keys 
        ax.set_xticklabels(outer_group_labels, rotation=0)

        ax.set_xlabel('Classes')
        ax.set_ylabel('Mean Correlation')
        # ax.set_title('Correlation: across CVs vs. across Folds', pad=10)
        ax.legend(loc='lower right')

        return data
    
    # ---------------------------------------------------------------------

    def plot_correlation_matrices(self, CV_params, class_label, axes=None, use_relevance=True, over_folds=True, tick_size=9):

        if class_label == 0:
            if use_relevance:
                corr_dict = self.correlation_dict_c0_R
            else:
                corr_dict = self.correlation_dict_c1_X
        else: 
            if use_relevance:
                corr_dict = self.correlation_dict_c1_R
            else:
                corr_dict = self.correlation_dict_c1_X

        if corr_dict is None:
            print("Correlations on relevance have not been computed yet. Compute correlations by calling compute_correlation(..)")
            return

        # -----------------------------------------------------
                
        if over_folds: 
            label = "Fold"
            title = "CV"
            key_mat = "Fold_x_Fold_matrices"
        else: 
            label = "CV"
            title = "Fold"
            key_mat = "CV_x_CV_matrices"


        corr_mats = corr_dict[key_mat]
        heatmaps = []

        for idx, corr_mat in enumerate(corr_mats):

            title_ = f"{title} {idx}"

            if not over_folds:
                ticklabels = [f"{params['nr']}" for params in CV_params]
            else:
                ticklabels = [subj for subj in range(self.num_folds)]

            if axes is None:
                _, ax = plt.subplots(1, 1, figsize=(2.75, 2.5))
            else:
                ax = axes[idx]

            sns.set(font_scale=0.75)

            ax.set_title(f"{title_}")
            hm = sns.heatmap(corr_mat, annot=False, cmap="coolwarm", linewidths=0.5, xticklabels=ticklabels, 
                            yticklabels=ticklabels, ax=ax, vmin=-1, vmax=1, cbar=False)
            
            hm.set_xticklabels(hm.get_xticklabels(), rotation=0, ha="right", fontsize=tick_size)
            hm.set_yticklabels(hm.get_yticklabels(), rotation=0, ha="right", fontsize=tick_size)

            ax.set_xlabel(label)
            ax.set_ylabel(label)

            heatmaps.append(hm)

        return heatmaps

    # -------------------------------------------------------------------------------------

    def plot_LRP_baseline(self, dh, params, class_label, title="", proportion_of_samples=1.0, discard_negative_rel=False, reversed_classes=False, model_id_of_subj=None, subject="all",
                        select_correct=None, select_samples_by_class=True, composite=None, use_which_data="train-test"):

        X_freq, R_freq = utils.load_data_and_compute_relevance(dh, params, class_label, title, proportion_of_samples=proportion_of_samples, 
                                                        discard_negative_rel=discard_negative_rel, reversed_classes=reversed_classes, model_id_of_subj=model_id_of_subj, subject=subject,
                                                        select_correct=select_correct, select_samples_by_class=select_samples_by_class, composite=composite, freq_domain=True,
                                                        use_which_data=use_which_data)

        bands = [(0,4,"delta"), (4,8,"theta"), (8,12,"alpha"), (12,30,"beta"), (30,60,"gamma")]

        fig, axes = plt.subplots(2, len(bands), figsize=(6, 4)) 

        self._plot_freq_bands_topo(X_freq, self.channel_names, bands, axes[0], title=title, global_scale=True, norm=False, log_scale=False, scale_bands=True, cmap="Reds", 
                                    label="", colorbar_offset=(0.0,0.5), vlim=None, mean=False, negative_lim=False)

        self._plot_freq_bands_topo(R_freq, self.channel_names, bands, axes[1], title="", global_scale=True, norm=False, log_scale=False, scale_bands=False, cmap="Reds", 
                                    label="", colorbar_offset=(0.0,0.2), vlim=None, mean=False, negative_lim=False)
        plt.show()

    # -------------------------------------------------------------------------------------

    def plot_sensors_of_brain_regions(self, brain_regions, cmap=None, figsize=(5,5)):
        
            plt.rc('font', size=10) 

            info = mne.create_info(self.channel_names, sfreq=128, ch_types='eeg')
            info.set_montage('standard_1020')

            fig, ax = plt.subplots(1,1, figsize=figsize)

            all_indices = []
            colors = []

            for _, (_, channels, color) in enumerate(brain_regions):
                indices = [i for i, ch_name in enumerate(self.channel_names) if ch_name in channels]
                all_indices.append(indices)
                colors.append(color)

            cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

            fig = mne.viz.plot_sensors(info, show_names=True, ch_type='eeg', to_sphere=True, axes=ax, ch_groups=all_indices, 
                                        pointsize=125, linewidth=0, sphere="auto", cmap=cmap)
            ax.set_facecolor("white")
            plt.show()
            return fig

    # -------------------------------------------------------------------------------------

    def _plot_freq_bands_topo_original(self, signal, ch_names, bands, ax, title="", global_scale=True, norm=True, log_scale=False, scale_bands=False, cmap=None, 
                            label="", colorbar_offset=[0,0], vlim=None, mean=False, negative_lim=True, scale_per_band=False, scale_1_div_f=False, scale_factors=None):
        suptitle = title

        # if cmap is None:
        #     if np.min(signal) < 0 or negative_lim:
        #         cmap = "bwr"
        #     elif not negative_lim:
        #         cmap = "Reds"
        #     else:
        #         cmap = "Reds"

        if len(signal.shape) == 3:
            if mean:
                signal = np.mean(signal, axis=0)
            else:
                signal = np.sum(signal, axis=0)

        signal_bands = []

        mult = signal.shape[-1] // 60

        min_per_band = []
        max_per_band = []

        # -------------------------------
        # sum over bins, then find min and max per band

        for (a, b, _) in bands:
            freq = np.copy(signal[:, mult*a:mult*b])
            if mean:
                freq = np.mean(freq, axis=1)   
            else:
                freq = np.sum(freq, axis=1)   

            signal_bands.append(freq)
            min_per_band.append(np.min(freq))
            max_per_band.append(np.max(freq))

        freq_max = np.max(max_per_band)
        freq_min = np.min(min_per_band)

        # total_sum = np.sum(signal_bands)
        # print("total_sum: ", total_sum)

        # -------------------------------
        # divide by global max (max value considering all bands)

        if global_scale and norm:
            for i in range(len(signal_bands)):
                # freq_max = total_sum
                signal_bands[i] /= freq_max
                max_per_band[i] /= freq_max
                min_per_band[i] /= freq_max

        # -------------------------------
        # scale bands: bring each band to the same amplitude

        if scale_bands:
            # scalars = [math.floor(1/val) for val in max_per_band]

            max_total = np.max(max_per_band)
            scalar_strs = []

            if scale_1_div_f and scale_factors is not None:
                scalars = scale_factors

                for scalar in scalars:
                    scalar_strs.append(f"{scalar:.1f}")

                for i in range(len(signal_bands)):
                    # signal_bands[i] /= max_total
                    signal_bands[i] *= scalars[i]

                max_val = -np.inf
                for sb in signal_bands:
                    if np.max(np.abs(sb)) > max_val:
                        max_val = np.max(np.abs(sb))

                for i in range(len(signal_bands)):
                    # signal_bands[i] /= max_total
                    signal_bands[i] /= max_val
            else:
                scalars = []

                for val in max_per_band:
                    scalar = max_total / val
                    scalars.append(scalar)
                    scalar_strs.append(f"{scalar:.1f}")

                for i in range(len(signal_bands)):
                    signal_bands[i] *= scalars[i]


        for i in range(len(bands)):
            ax_band = ax[i]

            band = bands[i]

            freq = signal_bands[i]

            if not global_scale and norm:
                freq_max = np.max(np.abs(freq))

            if i == 0:
                label_ = label
            else:
                label_ = ""

            title = utils.freq_band_name_to_latex(band[2])
        
            if vlim is None:
                if global_scale:
                    if negative_lim:
                        vlim = (-np.max(signal_bands), np.max(signal_bands))
                    else:
                        vlim = (0, np.max(signal_bands))
                else:
                    if negative_lim:
                        vlim = (-np.max(freq), np.max(freq))
                    else:
                        vlim = (0, np.max(freq))

                if log_scale:
                    vlim = (vlim[0]+1, vlim[1])

            if scale_bands and not scale_per_band:
                title += f" ($\\times$ {scalar_strs[i]})"

            if scale_per_band:
                vlim=(None, None)

            im = self._plot_topography(freq, 120, ch_names, title=title, label=label_, plot=True, sp_size=2, show_ch_names=False, vlim=vlim, ax=ax_band, cmap=cmap)

            if not scale_per_band:
                if i == len(bands)-1 and global_scale:
                    if suptitle:
                        cax = plt.axes([1.0 + colorbar_offset[0], 0.0 + colorbar_offset[1], 0.01, 0.2]) 
                    else:
                        cax = plt.axes([1.0 + colorbar_offset[0], 0.0 + colorbar_offset[1], 0.01, 0.2]) 
                    plt.colorbar(im, cax=cax)
            else:
                step = 0.22
                pos = step + i * step
                cax = plt.axes([pos + colorbar_offset[0], 0.0 + colorbar_offset[1], 0.01, 0.2]) 
                plt.colorbar(im, cax=cax)

        if suptitle:
            plt.suptitle(suptitle, y=0.9)

        return vlim, signal

 
    # -----------------------------------------------------------------------------------------

    def _plot_freq_bands_topo(self, signal, ch_names, bands, ax, title="", global_scale=True, norm=True, log_scale=False, scale_bands=False, cmap=None, 
                            label="", colorbar_offset=[0,0], vlim=None, mean=False, negative_lim=True, scale_per_band=False, scale_1_div_f=False, scale_factors=None, remove_neg=False):
        suptitle = title

        if len(signal.shape) == 3:
            if mean:
                signal = np.mean(signal, axis=0)
            else:
                signal = np.sum(signal, axis=0)

        signal_bands = []

        mult = signal.shape[-1] // 60

        min_per_band = []
        max_per_band = []

        # -------------------------------
        # sum over bins, then find min and max per band

        for (a, b, _) in bands:
            freq = np.copy(signal[:, mult*a:mult*b])

            if remove_neg:
                freq[freq < 0] = 0

            if mean:
                freq = np.mean(freq, axis=1)   
            else:
                freq = np.sum(freq, axis=1)  
 

            signal_bands.append(freq)
            min_per_band.append(np.min(freq))
            max_per_band.append(np.max(freq))

        freq_max = np.max(max_per_band)
        freq_min = np.min(min_per_band)

        # -------------------------------
        # divide by global max (max value considering all bands)

        if global_scale and norm:
            for i in range(len(signal_bands)):
                # freq_max = total_sum
                signal_bands[i] /= freq_max
                max_per_band[i] /= freq_max
                min_per_band[i] /= freq_max

        # -------------------------------
        # scale bands: bring each band to the same amplitude

        if scale_bands:

            max_total = np.max(max_per_band)
            scalar_strs = []

            if scale_1_div_f and scale_factors is not None:
                scalars = scale_factors

                for scalar in scalars:
                    scalar_strs.append(f"{scalar:.1f}")

                for i in range(len(signal_bands)):
                    signal_bands[i] *= scalars[i]

                max_val = -np.inf
                for sb in signal_bands:
                    if np.max(np.abs(sb)) > max_val:
                        max_val = np.max(np.abs(sb))

                for i in range(len(signal_bands)):
                    signal_bands[i] /= max_val


            else:
                scalars = []

                for val in max_per_band:
                    scalar = max_total / val
                    scalars.append(scalar)
                    scalar_strs.append(f"{scalar:.1f}")

                for i in range(len(signal_bands)):
                    signal_bands[i] *= scalars[i]

        # -------------------------------

        freqs = []

        for i in range(len(bands)):
            ax_band = ax[i]

            band = bands[i]

            freq = signal_bands[i]

            if not global_scale and norm:
                freq_max = np.max(np.abs(freq))

            if i == 0:
                label_ = label
            else:
                label_ = ""

            title = utils.freq_band_name_to_latex(band[2])
        
            if vlim is None:
                if global_scale:
                    if negative_lim:
                        vlim = (-np.max(signal_bands), np.max(signal_bands))
                    else:
                        vlim = (0, np.max(signal_bands))
                else:
                    if negative_lim:
                        vlim = (-np.max(freq), np.max(freq))
                    else:
                        vlim = (0, np.max(freq))

            if scale_bands:
                title += f" ($\\times$ {scalar_strs[i]})"

            im = self._plot_topography(freq, 120, ch_names, title=title, label=label_, plot=True, sp_size=2, show_ch_names=False, vlim=vlim, ax=ax_band, cmap=cmap)

            freqs.append(freq)
            if i == len(bands)-1 and global_scale:
                if suptitle:
                    cax = plt.axes([1.0 + colorbar_offset[0], 0.0 + colorbar_offset[1], 0.01, 0.2]) 
                else:
                    cax = plt.axes([1.0 + colorbar_offset[0], 0.0 + colorbar_offset[1], 0.01, 0.2]) 
                plt.colorbar(im, cax=cax)

        if suptitle:
            plt.suptitle(suptitle, y=0.8)

        return vlim, freqs

    # -----------------------------------------------------------------------

    def _plot_topography(self, signal, sampling_rate, ch_names=None, show_ch_names=True, title="", label="",  plot=True, vlim=(-1,1), sp_size=3, ax=None, cmap="RdBu_r"):
        
        if ch_names is None:
            ch_names = []

        ax.set_title(title)

        if label:
            ax.set_ylabel(label, rotation=90, labelpad=20)
        
        info = mne.create_info(ch_names, sfreq=sampling_rate, ch_types='eeg')
        info.set_montage('standard_1020')

        if not show_ch_names:
            ch_names = None

        im, cm = mne.viz.plot_topomap(signal, info, names=ch_names, show=False, axes=ax, vlim=vlim, cmap=cmap,
                                    image_interp="linear")

        plt.tight_layout()

        if plot:
            plt.plot()
        else:
            plt.close()

        return im 

    # -----------------------------------------------------------------------
    
    def _calc_ica_on_cluster(self, class_label, cv_idx, cluster_idx, time_domain=True, band=False):

        X, y = self.get_samples_of_cluster(class_label, cv_idx, cluster_idx, time_domain=time_domain)
        
        # # 8-12 Hz
        # low = 3*8
        # high = 3*12
        
        if band == "alpha":
            low = 8
            high = 12
        elif band == "beta":
            low = 12
            high = 30
        elif band == "gamma":
            low = 30
            high = 60
        else:
            low = 0
            high = 60

        info = mne.create_info(self.channel_names, self.sampling_rate, ch_types='eeg')
        epochs = mne.EpochsArray(X, info)
        epochs.set_montage('standard_1020')  

        if time_domain and band != False:
            print("filter time")
            epochs = epochs.copy().filter(l_freq=low, h_freq=high)

        ica = mne.preprocessing.ICA(
            n_components=None,
            max_iter="auto",
            method="infomax",  # Use the "extended Infomax" algorithm as specified by ICLabel 
            random_state=0,
        )

        picks = np.arange(len(self.channel_names)-1)
        ica.fit(epochs) 

        ic_labels = label_components(epochs, ica, method="iclabel")
        
        labels = ic_labels["labels"]
        proba = ic_labels["y_pred_proba"]
        
        return ica, labels, proba

    # ---------------------------------------------------------------------

    def _collect_data(self):
        
        tmp = joblib.load(os.path.join(self.paths_to_CV_iteration[0][0], "total_rel_per_filter_final"))
        num_total = len(tmp)
        num_filters_per_model = num_total // self.num_folds
        del tmp

        c=0
        fp_dict = {}
        for s in range(self.num_folds):
            for _ in range(num_filters_per_model):
                fp_dict[c] = s
                c += 1

        for class_label in self.class_labels:
            
            for cv_idx in range(self.num_CVs):

                for fold_idx in range(self.num_folds): 

                    path_store_data = self.paths_to_CV_iteration[class_label][cv_idx]

                    for f in range(num_total):

                        if fold_idx != fp_dict[f]:
                            continue

                        try:
                            if self.layer != "b3_flatten":

                                fn  = os.path.join(path_store_data, f"filter_activation_map_{fold_idx}_{f % num_filters_per_model}")
                                X_sim = joblib.load(fn)

                                fn  = os.path.join(path_store_data, f"filter_activation_map_rel_{fold_idx}_{f % num_filters_per_model}")
                                X_sim_rel = joblib.load(fn)
                            else:
                                X_sim = X_sim_rel = None

                            # samples_freq
                            fn  = os.path.join(path_store_data, f"X_freq_select_{fold_idx}_{f % num_filters_per_model}")
                            X_sim_samples = joblib.load(fn)

                            if self.use_what_for_similarity in ["samples_psd", "samples_rel_combined"]:
                                X_sim_samples = np.abs(X_sim_samples)**2 

                            X_sim_samples = np.mean(X_sim_samples, axis=0)
                            a, b = X_sim_samples.shape
                            X_sim_samples = X_sim_samples[:,:192].reshape(a, b // 12, 12)
                            X_sim_samples = X_sim_samples.mean(axis=-1)  
                            X_sim_samples = X_sim_samples.flatten()

                            X_sim_samples /= np.max(X_sim_samples)

                            # samples_rel
                            fn  = os.path.join(path_store_data, f"R_freq_select_{fold_idx}_{f % num_filters_per_model}")
                            R_sim_samples = joblib.load(fn)

                            if self.use_what_for_similarity == "samples_rel_pos":
                                R_sim_samples[R_sim_samples < 0] = 0
                                # R_sim_samples = R_sim_samples**2

                            R_sim_samples = np.mean(R_sim_samples, axis=0)
                            a, b = R_sim_samples.shape

                            # functional_groups = self.compute_functional_groups(R_sim_samples)

                            R_sim_samples = R_sim_samples[:,:192].reshape(a, b // 12, 12)
                            R_sim_samples = R_sim_samples.mean(axis=-1)  
                            R_sim_samples = R_sim_samples.flatten()

                            R_sim_samples /= np.max(np.abs(R_sim_samples))

                        except Exception as e:
                            print(">>> skip filter, no data:", f, e)

                            continue

                        if self.use_what_for_similarity == "combined":
                            # X_sim = torch.cat([X_sim, X_sim_rel])
                            X_sim = torch.mul(X_sim, X_sim_rel)

                        elif self.use_what_for_similarity == "activation_map_relevance":
                            X_sim = X_sim_rel
                        elif self.use_what_for_similarity == "samples_freq":
                            X_sim = X_sim_samples
                        elif self.use_what_for_similarity == "samples_rel":
                            X_sim = R_sim_samples
                        elif self.use_what_for_similarity == "samples_rel_pos":
                            X_sim = R_sim_samples
                        elif self.use_what_for_similarity == "samples_psd":
                            X_sim = X_sim_samples
                        elif self.use_what_for_similarity == "samples_rel_combined":
                            X_sim = np.concatenate([X_sim_samples, R_sim_samples])
                        # elif self.use_what_for_similarity == "samples_rel_functional_groups":
                        #    X_sim = functional_groups
                        else:
                           pass

                        if torch.is_tensor(X_sim):
                            X_sim = X_sim.cpu().detach().numpy()

                        self.data["X_sim"][class_label][cv_idx][fold_idx].append(X_sim)
                        self.data["X_freq_files"][class_label][cv_idx][fold_idx].append(os.path.join(path_store_data, f"X_freq_select_{fold_idx}_{f % num_filters_per_model}"))
                        self.data["X_time_files"][class_label][cv_idx][fold_idx].append(os.path.join(path_store_data, f"X_time_select_{fold_idx}_{f % num_filters_per_model}"))

                        self.data["R_freq_files"][class_label][cv_idx][fold_idx].append(os.path.join(path_store_data, f"R_freq_select_{fold_idx}_{f % num_filters_per_model}"))
                        self.data["R_time_files"][class_label][cv_idx][fold_idx].append(os.path.join(path_store_data, f"R_time_select_{fold_idx}_{f % num_filters_per_model}"))
                        self.data["y_files"][class_label][cv_idx][fold_idx].append(os.path.join(path_store_data, f"y_select_{fold_idx}_{f % num_filters_per_model}"))

    # ---------------------------------------------------------------------
                    
    def _prepare_paths(self):
        self.paths_to_CV_iteration = {}

        for class_label in self.class_labels:
        
            self.paths_to_CV_iteration[class_label] = {}

            for i in range(self.num_CVs):

                _, path_for_cv = Results.assemble_path(self.CV_params[i], self.base_result_path, 
                                            self.num_selected_samples, self.ds_name, self.layer, class_label,  
                                            self.select_samples_by_class, self.reversed_classes, self.select_correct,  
                                            False, self.use_which_data, self.testing)

                self.paths_to_CV_iteration[class_label][i] = path_for_cv

    # ---------------------------------------------------------------------
                
    def _calc_cluster_coherence(self, class_label, cv_idx, cluster_idx):
        
        embedding = self.get_data_class_cv("embedding", class_label, cv_idx)
        labels = self.get_data_class_cv("labels", class_label, cv_idx)
        indices = self.get_data_class_cv("indices", class_label, cv_idx)
        unique_labels = copy(self.get_data_class_cv("unique_labels", class_label, cv_idx))
        index = unique_labels.index(cluster_idx)
        cluster_points = embedding[indices[index]]
        pairwise_distances = pdist(cluster_points, metric='euclidean')
        average_distance = np.mean(pairwise_distances)
            
        if np.isnan(average_distance):
            average_distance = -1

        inverse = 1.0/average_distance

        return inverse

    # ---------------------------------------------------------------------

    def _calc_cluster_relevance(self, class_label, cv_idx, cluster_idx):
        R_time, _ = self.get_samples_of_cluster(class_label, cv_idx, cluster_idx, time_domain=True, relevance=True)
        return np.mean(R_time)
    
    # ---------------------------------------------------------------------

    def _calc_correlations(self, class_label, CV_params, plot=True, discard_negative=False, take_abs=False, use_relevance=True): 

        correlations_over_elements_dict = {
            "Fold_x_Fold_matrices": [],  "CV_x_CV_matrices": [], 
            "Fold_x_Fold_corr_list": [], "CV_x_CV_corr_list": []
        }

        for_CV_x_CV, for_fold_x_fold = self._gather_data_for_correlation(CV_params, class_label, use_relevance=use_relevance)

        for over_folds in [False, True]:

            if over_folds: 
                corr_dict = for_fold_x_fold
                label = "Fold"
                key_list = "Fold_x_Fold_corr_list"
                key_mat = "Fold_x_Fold_matrices"
            else: 
                corr_dict = for_CV_x_CV
                label = "CV"
                key_list = "CV_x_CV_corr_list"
                key_mat = "CV_x_CV_matrices"

            for key in corr_dict.keys():

                ticklabels = [f"{params['nr']}" for params in CV_params]
                R_sim_list = corr_dict[key]


                # get max value (R_sim_list is ragged, so np.max doesn't work)
                max_val = -np.inf

                for R_sim in R_sim_list:
                    max_ = np.max(R_sim)
                    if max_ > max_val:
                        max_val = max_

                R_sim_list_ = []

                for R_sim in R_sim_list:
                    
                    R_sim_ = R_sim

                    if discard_negative:
                        R_sim_[R_sim_ < 0] = 0
                    elif take_abs:
                        R_sim_ = np.abs(R_sim_)

                    R_sim_ = np.sum(R_sim_, axis=0)
                    R_sim_ = R_sim_.flatten()

                    R_sim_list_.append(R_sim_)

                R_sim_list = np.array(R_sim_list_)  
                corr_mat = np.corrcoef(R_sim_list)

                if plot:
                    _, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))

                    if np.min(corr_mat) < 0:
                        vmin = -1
                    else:
                        vmin = 0    

                    sns.set(font_scale=0.5)

                    ax.set_title(f"Correlation on {label}s - R - {key}")
                    hm1 = sns.heatmap(corr_mat, annot=False, cmap="coolwarm", linewidths=0.5, xticklabels=ticklabels, 
                                    yticklabels=ticklabels, ax=ax, vmin=vmin, vmax=1)
                    hm1.set_xticklabels(hm1.get_xticklabels(), rotation=90, ha="right")
                    ax.set_xlabel(label)
                    ax.set_ylabel(label)
                    plt.show()

                lower_triangle_indices = np.tril_indices(corr_mat.shape[0], k=-1)
                lower_triangle_elements = corr_mat[lower_triangle_indices]

                correlations_over_elements_dict[key_list] += list(lower_triangle_elements)
                correlations_over_elements_dict[key_mat].append(corr_mat)

                if plot:
                    plt.tight_layout()
                    plt.show()        

        return correlations_over_elements_dict

    # -------------------------------------------------------------------------------------

    def _gather_data_for_correlation(self, CV_params, class_label, use_relevance=True):
        
        for_CV_x_CV = {}

        folds = [i for i in range(self.num_folds)]

        for fold_idx, fold in enumerate(folds):
            for_CV_x_CV[fold_idx] = []

            for cv_idx, cv in enumerate(CV_params):

                if use_relevance:
                    R_freq_fns = self.get_data_class_cv_fold("R_freq_files", class_label, cv_idx, fold_idx)
                else:
                    R_freq_fns = self.get_data_class_cv_fold("X_freq_files", class_label, cv_idx, fold_idx)

                R_freq = []
                for fn in R_freq_fns:
                    R_freq.append(joblib.load(fn))
                R_freq = np.concatenate(R_freq, axis=0)

                for_CV_x_CV[fold_idx].append(R_freq)


        for_fold_x_fold = {}

        for cv_idx, cv in enumerate(CV_params):
            for_fold_x_fold[cv_idx] = []
            
            for fold_idx, fold in enumerate(folds):

                if use_relevance:
                    R_freq_fns = self.get_data_class_cv_fold("R_freq_files", class_label, cv_idx, fold_idx)
                else:
                    R_freq_fns = self.get_data_class_cv_fold("X_freq_files", class_label, cv_idx, fold_idx)

                R_freq = []
                for fn in R_freq_fns:
                    R_freq.append(joblib.load(fn))
                R_freq = np.concatenate(R_freq, axis=0)

                for_fold_x_fold[cv_idx].append(R_freq)

        return for_CV_x_CV, for_fold_x_fold
    
    # -------------------------------------------------------------------------------------
