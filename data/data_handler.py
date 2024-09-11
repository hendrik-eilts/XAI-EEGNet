import os
import math
import joblib
import data.utils as utils

from tqdm import tqdm
from abc import abstractmethod
from training import utils as train_utils


class DataHandler:

    def __init__(self, use_ica, base_result_path, sampling_rate, test, resample, ds_name, step_size,
                 window_length, num_classes, processed_data_path):

        self.channel_names = None
        self.test = test
        self.num_classes = num_classes
        self.ds_name = ds_name
        self.window_length = window_length
        self.step_size = step_size
        self.use_ica = use_ica
        self.resample = resample
        self.baseline_correction = False
        self.num_data_points = None
        self.use_ica = use_ica
        self.base_result_path = base_result_path
        self.processed_data_path = processed_data_path
        self.sampling_rate = sampling_rate
        self.locs_path = ""
        self.channel_names = []
        self.num_subjects = None
        self._prepare_paths()

    # ---------------------------------------------------------------------------------------

    def __str__(self):
        output = "............\n"
        output += f"{self.ds_name} Data Handler\n"
        output += "\tchannels: " + str(self.channel_names) + "\n"
        output += "\ttest_mode: " + str(self.test) + "\n"
        output += "\twindow_length: " + str(self.window_length) + "\n"
        output += "\tstep_size: " + str(self.step_size) + "\n"
        output += "\tuse_ica: " + str(self.use_ica) + "\n"
        output += "\tbaseline correction: " + str(self.baseline_correction) + "\n"
        output += "\tresampled: " + str(self.resample) + "\n"
        output += "\tsampling_rate: " + str(self.sampling_rate) + "\n"

        return output

    # ---------------------------------------------------------------------------------------

    def _create_dh_path(self):
        ica_str = '_ica' if self.use_ica else ""
        wl_str = str(int(1000*self.window_length))
        test_str = "_test" if self.test else ""
        blc_str = "_blc" if self.baseline_correction else ""
        return f"resample_{self.resample}_sr_{self.sampling_rate}_wl_{wl_str}_step_" + f"{int(self.step_size*100)}{ica_str}{blc_str}" + f"{test_str}"

    # ---------------------------------------------------------------------------------------

    def _prepare_paths(self):
    
        self.result_path = os.path.join(self.base_result_path, f"classification_results/{self.ds_name}_data")
        self.result_path = os.path.join(self.result_path, self._create_dh_path())

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)

        self.epochs_path = os.path.join(self.processed_data_path, f"epochs_list_{self.ds_name}")
        self.icas_path = os.path.join(self.result_path, "icas")
        self.model_path = os.path.join(self.result_path, "model")

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    # ---------------------------------------------------------------------------------------

    def create_X_y_separate(self, epochs_list, start_subj=0, file_name_suffix=""):
        X_separate = []
        y_separate = []
        subjects = []

        folder = f"separate_data_list_{self.ds_name}{file_name_suffix}"
        folder_path = os.path.join(self.processed_data_path, folder)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # for k, (ep, y, idx) in tqdm(enumerate(epochs_list)):
        for k in tqdm(range(start_subj, len(epochs_list))):
            ep, y, idx = epochs_list[k]

            # ch_names = ep.info['ch_names']
            X = ep.get_data()

            if self.window_length is not None:

                window_length = int(self.window_length * self.sampling_rate)

                # only cut if the new window length is difference from the existing (with some tolerance)
                if math.fabs(window_length-X.shape[-1]) > 10:
                    X, y = utils.cut_data_into_windows(X, y, window_length, self.step_size)

            #X = self._transform_features(X, y, feature_transform_method, self.locs_path, ch_names,
            #                             excluded=excluded)

            X_separate.append(X)
            y_separate.append(y)
            subjects.append(idx)

            path = os.path.join(folder_path, f"subject_data_{k}")
            joblib.dump((X, y, idx), path)

            if idx == 0:
                self.num_data_points = X_separate[0].shape[-1]

    # ---------------------------------------------------------------------------------------

    def get_epochs(self):
        try:
            return joblib.load(self.epochs_path)
        except FileNotFoundError:
            print(f"{self.ds_name}, get_epochs(), file not found in path {self.epochs_path}")
            return None

    # ---------------------------------------------------------------------------------------

    def get_X_y_separate(self, subject_idx):
        folder = f"separate_data_list_{self.ds_name}"
        folder_path = os.path.join(self.processed_data_path, folder)
        path = os.path.join(folder_path, f"subject_data_{subject_idx}")

        try:
            return joblib.load(path)
        except FileNotFoundError:
            print(f"{self.ds_name}, get_X_y_separate(...), file not found in path {path}")
            return None

    # ---------------------------------------------------------------------------------------

    def get_transformed_data(self, subject_idx):

        data = self.get_X_y_separate(subject_idx)

        if data is None:
            print("data is none")
            return None, None, None
        
        X_subj, y_subj, subj = data

        # apply rereferencing
        X_subj = train_utils.apply_common_average_rereferencing(X_subj, numpy=True)

        # apply standardization
        X_subj = train_utils.standardize_data(X_subj, numpy=True)

        return X_subj, y_subj, subj

    # ---------------------------------------------------------------------------------------

    def get_transformed_data_shape(self):

        data = self.get_X_y_separate(0)

        if data is not None:
            X_subj, _, _ = data
            X_subj = X_subj[:1, :, :]
            return X_subj.shape
        
        print("data is none")
        return None

    # ---------------------------------------------------------------------------------------

    def load_and_preprocess_data(self, create_new_if_exists=False):
        print("Raw & Epochs")

        epochs = self.get_epochs()

        if epochs and not create_new_if_exists:
            print("epochs already exists")
        else:
            raw_list = self.load_raws()

            raws_pp = self.preprocess_raws(raw_list)

            if self.use_ica:
                icas = utils.compute_ICAs(raws_pp, self.result_path, plot=True)
                joblib.dump(icas, self.icas_path)
                icas = joblib.load(self.icas_path)
                raws_pp = utils.apply_automatic_ICA_artifact_removal(raws_pp, icas, verbose=1)

            self.create_epochs(raws_pp)

        # ------------

        print("X, y separate for subjects")

        epochs_list = self.get_epochs()

        start_subj = -1

        # skip the subjects already done
        if not create_new_if_exists:
            for idx in range(self.num_subjects+1):
                start_subj = idx
                data = self.get_X_y_separate(idx)
                if data is None:
                    break

        self.create_X_y_separate(epochs_list, start_subj=start_subj)

    # ---------------------------------------------------------------------------------------

    def get_num_classes(self):
        return self.num_classes

    # ---------------------------------------------------------------------------------------

    def get_num_subjects(self):
        return self.num_subjects

    # ---------------------------------------------------------------------------------------

    def get_channel_names(self):
        return self.channel_names

    # ---------------------------------------------------------------------------------------

    def get_num_channels(self):
        return len(self.get_channel_names())

    # ---------------------------------------------------------------------------------------

    @abstractmethod
    def load_raws(self, verbose=0):
        pass

    # ---------------------------------------------------------------------------------------

    @abstractmethod
    def preprocess_raws(self, raw_list, verbose=0, plot=False):
        pass

    # ---------------------------------------------------------------------------------------

    @abstractmethod
    def create_epochs(self, raw_list, verbose=0, plot=False):
        pass

    # -----------------------------------------------------------------------------------------------
