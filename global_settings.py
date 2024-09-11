import os
import mne
import random
import warnings
import numpy as np
from data import data_handler_gme, data_handler_cho, data_handler_kul
from zennit.composites import EpsilonAlpha2Beta1
# ---------------------------------------------------------------------------------------------

# This flag only changes which datasets are added to the list of datasets
# which are used by the other files. By setting this to true, only the internal/external attention data are used. 

test = True

# ---------------------------------------------------------------------------------------------
# Paths

# CHANGE TO YOUR LOCATION
base_path = "/share/temp/students/heilts/Masterthesis"


gme_data_path = os.path.join(base_path, "datasets/GME_Data")
iea_data_path = os.path.join(base_path, "datasets/BA_Data/iea_experiment")
ceh_data_path = os.path.join(base_path, "datasets/CEH_Data/EEG raw data/header_marker_eeg")
kul_data_path = os.path.join(base_path, "datasets/kul-dataset/data_raw")
locs_path = os.path.join(base_path, "kul-dataset/locs2d.npy")

processed_data_path = os.path.join(base_path, "processed_data")

# ---------------------------------------------------------------------------------------------
# Randomness & Misc

seed = 42
file_format = "png"
np.random.seed(seed)
mne.set_log_level("ERROR")
random.seed(a=seed, version=2)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------------------------

sampling_rate = 128

dh_cho = data_handler_cho.DataHandlerCho(use_ica=False, base_result_path=base_path, sampling_rate=sampling_rate, test=False,
                                        resample=True, use_eegnet_torch=True, ds_name="CHO", processed_data_path=processed_data_path)

dh_gme = data_handler_gme.DataHandlerGME(window_length=3, step_size=1, use_ica=False, base_result_path=base_path,
                                         resample=True, sampling_rate=sampling_rate, test=False, data_path=gme_data_path, use_eegnet_torch=True, 
                                         ds_name="GME", processed_data_path=processed_data_path)

dh_kul = data_handler_kul.DataHandlerKUL(window_length=3, step_size=1, use_ica=False, base_result_path=base_path,
                                         resample=True, sampling_rate=sampling_rate, test=False, data_path=kul_data_path,
                                         locs_path=locs_path, use_eegnet_torch=True, ds_name="KUL", processed_data_path=processed_data_path)

data_handlers = []

if test:
    data_handlers.append(dh_gme)
else:
    data_handlers.append(dh_gme)
    data_handlers.append(dh_cho)
    data_handlers.append(dh_kul)
    
# ---------------------------------------------------------------------------------------------

composite = EpsilonAlpha2Beta1() 

CV_params = []
n_repetitions = 10
batch_size = 128

for n in range(n_repetitions):

    # Seed per fold, CAR per sample
    CV_params.append({"bs": batch_size, "val_size": 0, "do":0.25, "seed": "_per_fold", "nr":n, "CAR": "_per_sample"})
    
    # Layer Ablation
    # CV_params.append({"bs": 1024, "val_size": 0, "do":0.25, "seed": n, "nr":n, "conv_layers": 1})

num_samples_to_select = 0.2

select_samples_by_class = True
reversed_classes = False
select_correct = None  # None: all, True: correct, False: incorrect

# None: all
# True: correct
# False: incorrect
select_correct = None

select_samples_by_class = True
reversed_classes = False

class_labels = [0, 1]

epochs = 500

use_what_for_similarity = "activation_map"

use_which_data = "test"
