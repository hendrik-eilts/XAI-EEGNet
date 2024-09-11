import numpy as np
from tqdm import tqdm
import mne
import joblib
from moabb.datasets import Cho2017
from data.data_handler import DataHandler


class DataHandlerCho(DataHandler):

    def __init__(self, use_ica, base_result_path, processed_data_path, sampling_rate, test=False, resample=True, locs_path="", num_classes=2, use_eegnet_torch=False, ds_name="CHO"):

        super(DataHandlerCho, self).__init__(use_ica, base_result_path, sampling_rate, test, resample, ds_name,
                                             step_size=1, window_length=3, num_classes=num_classes, processed_data_path=processed_data_path)

        self.num_subjects = 49
        self.event_identity = {'left_hand': 0, 'right_hand': 1}
        self.event_id = None
        self.events_list = None
        self.locs_path = locs_path
        self.channel_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3',
                              'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3',
                              'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2',
                              'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8',
                              'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
        
        self.use_eegnet_torch = use_eegnet_torch

    # ---------------------------------------------------------------------------------------

    def load_raws(self, verbose=0):
        dataset = Cho2017()
        self.event_id = dataset.event_id

        data = dataset.get_data()
        raw_list = []

        for subj_idx, d in data.items():
            raw = d['session_0']['run_0']

            raw_list.append((raw, str(subj_idx)))

        return raw_list



    def load_raws1(self, verbose=0):
        dataset = Cho2017()
        self.event_id = dataset.event_id

        data = dataset.get_data()
        raw_list = []

        for subj_idx, d in data.items():
            raw = d['session_0']['run_0']

            raw_list.append((raw, str(subj_idx)))

        return raw_list, dataset
    
    # ---------------------------------------------------------------------------------------

    def preprocess_raws(self, raw_list, verbose=0, plot=False):

        events_list = []

        for idx, (raw, fn) in tqdm(enumerate(raw_list)):
            if verbose:
                print(raw.info)

            raw.notch_filter(np.arange(50, 201, 50), fir_design='firwin')
            raw.filter(1, 60)

            events = mne.find_events(raw, shortest_event=0, verbose=False)
            events = mne.pick_events(events, include=list(self.event_id.values()))

            if self.resample:
                if verbose:
                    print("resample to ", self.sampling_rate)
                raw, events = raw.resample(self.sampling_rate, events=events)

            events_list.append(events)

            # raw = raw.set_eeg_reference(ref_channels="average")
            # if self.scale:
            #     raw = raw.apply_function(utils.standardize_data, channel_wise=False)

            raw_list[idx] = (raw, fn)

            if verbose:
                print(raw.info)
            if plot:
                raw.plot()

        self.events_list = events_list

        return raw_list

    # ---------------------------------------------------------------------------------------

    def create_epochs(self, raw_list, verbose=0, plot=False):

        epochs_list = []
        tmin = 0
        tmax = 3

        for (raw, fn), events in tqdm(zip(raw_list, self.events_list)):
            picks = mne.pick_types(raw.info, eeg=True, stim=False)

            epochs = mne.Epochs(
                raw,
                events,
                event_id=self.event_id,
                tmin=tmin,
                tmax=tmax,
                proj=False,
                baseline=None,
                preload=True,
                verbose=False,
                picks=picks,
                event_repeated="drop",
                on_missing="ignore",
            )

            if self.channel_names is None:
                self.channel_names = epochs.info['ch_names']
                print(self.channel_names)

            if verbose:
                print(epochs.info)
                print(epochs)
                print("These events were dropped:")
                print(epochs.drop_log)
            if plot:
                epochs.plot()

            y = (epochs.events[:, 2]).astype(np.int64)
            y = [val-1 for val in y]

            epochs_list.append((epochs, y, fn))

        joblib.dump(epochs_list, self.epochs_path)

        return epochs_list

    # -----------------------------------------------------------------------------------------------
