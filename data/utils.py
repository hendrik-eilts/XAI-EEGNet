# from mne_icalabel import label_components
import matplotlib.pyplot as plt
import numpy as np
import mne
import os


# ----------------------------------------------------------------------------------

def apply_automatic_ICA_artifact_removal(raw_list, icas, verbose=0):
    filtered_ica = []

    for (raw, idx), ica in zip(raw_list, icas):
        raw.load_data()

        print("!!! apply_automatic_ICA_artifact_removal is broken !!!")
        # ic_labels = label_components(raw, ica, method="iclabel")

        # labels = ic_labels["labels"]

       # exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]

        #if verbose:
        #    print("Excluding these ICA components:", [labels[i] for i in exclude_idx])

       # ica.apply(raw, exclude=exclude_idx)

        filtered_ica.append((raw, idx))

    return filtered_ica

# ------------------------------------------------------------------------------------------

def compute_ICAs(raw_list, result_path, plot=False):
    print("create icas")

    icas = []

    for i, (raw, idx) in enumerate(raw_list):

        ica = mne.preprocessing.ICA(
            # n_components=15,
            max_iter="auto",
            method="infomax",
            random_state=0,
            fit_params=dict(extended=True),
        )

        ica.fit(raw)
        icas.append(ica)

        fig = plt.figure(figsize=(8,8))
        if plot:
            ica.plot_components(inst=raw, picks=range(22), show=False)
        plt.savefig(os.path.join(result_path, f"ica_comps_raw_gme_{i}_{idx}.png"))
        plt.show()

    return icas

# ----------------------------------------------------------------------------------

def cut_data_into_windows(X, y, window_length, step_size):
    X_new = []
    y_new = []
    step = int(step_size*window_length)

    for sample, label in zip(X,y):
        position = 0

        while((position+window_length) < X.shape[2]):
            sample_new = sample[:,position:(position+window_length)]
            position += step
            X_new.append(sample_new)
            y_new.append(label)

    return np.array(X_new), y_new
    
# ----------------------------------------------------------------------------------
