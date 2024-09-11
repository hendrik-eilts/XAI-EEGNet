import mne
import joblib
import torch 
import numpy as np
from scipy.io import loadmat
from data.data_handler import DataHandler



class DataHandlerKUL(DataHandler):

    def __init__(self, window_length, step_size, use_ica, base_result_path, data_path, processed_data_path, locs_path, sampling_rate=128,
                 test=False, resample=True, scale=False, num_classes=2, use_eegnet_torch=False, ds_name="KUL"):

        super(DataHandlerKUL, self).__init__(use_ica, base_result_path, sampling_rate, test, resample, ds_name,
                                             step_size, window_length, num_classes, processed_data_path=processed_data_path)

        self.channel_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3',
                              'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3',
                              'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2',
                              'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8',
                              'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

        self.data_path = data_path
        self.filenames = None
        self.locs_path = locs_path
        self._prepare_paths()

        self.lp_freq = 0.1
        self.hp_freq = 60
        self.srate = 128
        self.overlap = 1 - self.step_size
        self.num_subjects = 16
        self.use_eegnet_torch = use_eegnet_torch

    # ---------------------------------------------------------------------------------------

    def load_raws(self, verbose=None):
        # Code übernommen von Gabriel Ivucic und leicht modifiziert

        raw_list = []
        subjects = self.num_subjects

        # 6 minutes one label per trial
        trials = 8
        channels = 64
        max_trial_dur = self.srate*60*6  # sr * seconds * minutes
        labels = np.zeros((subjects,trials))

        data = np.zeros((subjects, trials, max_trial_dur, channels))

        # organize data and labels
        for subject in range(subjects):

            subject_data = loadmat(f"{self.data_path}/S{subject+1}", mat_dtype=False, squeeze_me=False)
            relevant_trials = subject_data["trials"][:, :trials][0]

            for trial in range(trials):
                trial_data = relevant_trials[trial][0][0][0][0][0][1][:max_trial_dur]

                # common average rereferencing has been moved into the training and test loop
                car_trial_data = trial_data.T # - np.mean(trial_data,1)
                data[subject, trial] = car_trial_data.T
                label = relevant_trials[trial][0][0][3][0] # 1 == eeg data
                float_label = 0 if label == "L" else 1
                labels[subject, trial] = float_label

        # create mne info object to filter data
        chan_names = [str(x) for x in np.arange(channels)+1]
        info = mne.create_info(ch_names=chan_names, ch_types=['eeg']*channels, sfreq=self.srate)

        # read segments into mne and apply filters
        for p, participant_data in enumerate(data):

            for t_idx, trial in enumerate(participant_data):
                # read single segment into mne
                # reshaping
                # create mne "raw" object
                trial = trial.T
                raw = mne.io.RawArray(trial, info=info, first_samp=0, copy='auto', verbose=verbose)
                raw_list.append((raw, p, t_idx))

        self.labels = labels

        return raw_list

    # ---------------------------------------------------------------------------------------

    def preprocess_raws(self, raw_list, verbose=0, plot=False):

        for idx, (raw, subj, trial) in enumerate(raw_list):

            if verbose:
                print(raw.info)

            # raw.notch_filter(np.arange(50, 50, 50), fir_design='firwin')
            raw.notch_filter(freqs=50, fir_design='firwin')

            # eeg_notch = eeg_filtered.copy().notch_filter(freqs=60)

            raw.filter(1, 60)

            # raw = raw.set_eeg_reference(ref_channels="average")
            # if self.scale:
            #     raw = raw.apply_function(utils.standardize_data, channel_wise=False)

            if verbose:
                print(raw.info)
            if plot:
                raw.plot()

        return raw_list

    # ------------------------------------------------------------------------------------------

    def create_epochs(self, raw_list, verbose=0, plot=False):
        subj_epoch_dict = {}
        subj_label_dict = {}

        for raw, subj, trial in raw_list:
            if subj not in subj_epoch_dict:
                subj_epoch_dict[subj] = []
                subj_label_dict[subj] = []

            label = self.labels[subj][trial]

            epochs = mne.make_fixed_length_epochs(raw, duration=self.window_length, overlap=self.overlap, verbose=False)

            labels_ = [label]*epochs.get_data().shape[0]
            subj_epoch_dict[subj].append(epochs)
            subj_label_dict[subj].append(labels_)

        epochs_list = []

        for ((subj, subj_epochs_list), (_, subj_label_list)) in zip(subj_epoch_dict.items(), subj_label_dict.items()):
            epochs = mne.concatenate_epochs(subj_epochs_list)
            epochs_labels = np.concatenate(subj_label_list)


            if self.resample:
                # print(self.sampling_rate)
                epochs = epochs.resample(sfreq=self.sampling_rate)

            # print(epochs.info)

            epochs_list.append((epochs, epochs_labels, subj))

            if verbose:
                print(epochs.info)
                print(epochs)
                print("These events were dropped:")
                print(epochs.drop_log)
            if plot:
                epochs.plot()

        joblib.dump(epochs_list, self.epochs_path)

        return epochs_list

    # -----------------------------------------------------------------------------------------------
