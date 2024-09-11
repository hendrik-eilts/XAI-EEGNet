import os
import mne
import numpy as np
import joblib
from data.data_handler import DataHandler
from tqdm import tqdm

from sys import platform
import pyxdf

class DataHandlerGME(DataHandler):
     
    def __init__(self, window_length, step_size, use_ica, base_result_path, data_path, processed_data_path, test=False,
                 resample=False, sampling_rate=500, filename_suffix="", offset=3, locs_path="",
                 num_classes=2, use_eegnet_torch=False, ds_name="GME"):

        super(DataHandlerGME, self).__init__(use_ica, base_result_path, sampling_rate, test, resample, ds_name,
                                             step_size, window_length, num_classes, processed_data_path=processed_data_path)

        self.channel_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 
                              'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 
                              'FC2', 'F4', 'F8', 'Fp2']

        self.label_to_string_map = {0: "external", 1: "internal"}
        
        self.sample_length = 6
        self.offset = offset
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.events_list = None
        self.locs_path = locs_path
        self.filename_suffix = filename_suffix
        self.num_subjects = 10
        self.use_eegnet_torch = use_eegnet_torch
        
# ---------------------------------------------------------------------------------------

    def load_raws(self, verbose=0):

        file_list = [
            # ("proband_003_haupt1.xdf", 3), #  -> Fehler: Kein Marker Stream
            # ("proband_003_haupt2.xdf", 3),
            ("proband_004_haupt1.xdf", 4),
            ("proband_004_haupt2.xdf", 4),
            ("proband_005_haupt1.xdf", 5),
            ("proband_005_haupt2.xdf", 5),
            ("proband_006_haupt1.xdf", 6),
            ("proband_006_haupt2.xdf", 6),
            ("proband_007_haupt1.xdf", 7),
            ("proband_007_haupt2.xdf", 7),
            ("proband_008_haupt1.xdf", 8),
            ("proband_008_haupt2.xdf", 8),
            ("proband_009_haupt1.xdf", 9),
            ("proband_009_haupt2.xdf", 9),
            ("proband_010_haupt1.xdf", 10),
            ("proband_010_haupt2.xdf", 10),
            ("proband_011_haupt1.xdf", 11),
            ("proband_011_haupt2.xdf", 11),
            ("sub-daniel-haupt_ses-S001_task-Default_run-001_eeg.xdf", 12),
            ("sub-daniel-haupt-2_ses-S001_task-Default_run-001_eeg.xdf", 12),
            ("Haupt1_sub-P001_ses-S001_task-Default_run-001_eeg.xdf", 13),
            ("Haupt2_sub-P001_ses-S001_task-Default_run-001_eeg.xdf", 13),
        ]

        if self.test:
            file_list = file_list[:6]

        raw_list = self._load_xdf_into_raw(file_list, self.data_path)
        return raw_list

    # ------------------------------------------------------------------------------------------

    def preprocess_raws(self, raw_list, verbose=0, plot=False):
        filtered = []
        events_list = []

        for raw, idx in tqdm(raw_list):

            picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
            raw.filter(1, 60)
            raw.notch_filter(np.arange(50, 201, 50), picks=picks, fir_design='firwin')

            events = mne.find_events(raw, stim_channel="STI", initial_event=True, output="onset")

            if self.resample:
                if verbose:
                    print("resampling to ", self.sampling_rate)
                raw, events = raw.resample(self.sampling_rate, events=events)

            filtered.append((raw, idx))
            events_list.append(events)

        self.events_list = events_list

        return filtered

    # ------------------------------------------------------------------------------------------

    def create_epochs(self, raw_list, verbose=0, plot=0):

        tmin = self.offset
        tmax = tmin + self.sample_length
        epochs_list = []

        # go through each Raw and create Epochs
        for (raw, subj_nr), events in zip(raw_list, self.events_list):

            epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None)

            epochs.load_data()
            try:
                epochs = epochs.drop_channels("STI")
            except:
                print("drop_channels STI did not work")

            epochs_list.append((epochs, subj_nr))

        epochs_list = epochs_list

        epochs_list_ = []

        # merge epochs from same subjects, also extracts and formats the labels
        for i in range(0, len(epochs_list)-1):
            subj = epochs_list[i][1]
            subj_next = epochs_list[i+1][1]

            epochs = epochs_list[i][0]
            epochs_next = epochs_list[i+1][0]

            if subj == subj_next:
                epochs_new = mne.concatenate_epochs([epochs, epochs_next])
                y = self._adjust_epochs_labels(epochs_new)
                epochs_list_.append((epochs_new, y, subj))

        joblib.dump(epochs_list_, self.epochs_path)

        return epochs_list_

    # ------------------------------------------------------------------------------------------

    def _load_xdf_into_raw(self, file_list, data_path):
        """
        Given a list of filenames, this method loads the data of the files into Raw-Objects
        and returns them

        """

        raw_list = []

        for filename, idx in file_list:
            path_xdf = os.path.join(data_path, filename)

            streams, header = pyxdf.load_xdf(path_xdf, verbose=0)
            data_matrix, data_timestamps, channel_labels, stream_to_pos_mapping = self._load_stream_data(streams)

            # stream_info = streams[0]['info']
            # fs = float(stream_info['nominal_srate'][0])
            fs = 500
            info = mne.create_info(channel_labels, fs, ch_types='eeg')

            data_reshaped = data_matrix.transpose()
            data_reshaped = data_reshaped[:32, :]

            # get the markers and the ground truths
            marker_timestamps, ground_truths = self._get_marker_and_labels_from_stream(streams, False,
                                                                                    stream_to_pos_mapping)

            # create the stim channel
            stim_data = self._create_stim_channel(data_timestamps, marker_timestamps, ground_truths)

            raw = mne.io.RawArray(data=data_reshaped, info=info)

            # add the stim channel
            self._add_stim_to_raw(raw, stim_data)

            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage)

            raw._filenames = [filename]
            raw_list.append((raw, idx))

        return raw_list

    # ------------------------------------------------------------------------------------------

    def _adjust_epochs_labels(self, epochs):

        # extract the labels from the events
        y = (epochs.events[:, 2] - 2).astype(np.int64)

        # adjust the labels
        for idx, j in enumerate(y):
            if j == 0:
                y[idx] = 0
            else:
                y[idx] = 1

        return y

    # -----------------------------------------------------------------------------------------------
    @staticmethod
    def _load_stream_data(streams, stream_name='ActiChamp-0'):
        """
        Loads the recorded data, timestamps and channel information of a specific
        stream in a xdf file.

        Args:
            streams :
                Streams which should be used, has to be loaded before with _load_stream()
            stream_name (str):
                Name of the stream in the xdf file to be loaded

        Returns:
            data_matrix, data_timestamps, channel_labels

        """

        # find all stream names in the file
        stream_to_pos_mapping = {}
        for pos in range(0, len(streams)):
            stream = streams[pos]['info']['name']
            stream_to_pos_mapping[stream[0]] = pos

        # raise an error if the searched stream_name is not existing
        if stream_name not in stream_to_pos_mapping.keys():
            raise ValueError(
                'Stream ' + str(stream_name) + ' not found in xdf file. '
                                               'Found streams are: '
                                               '\n' + str(
                    stream_to_pos_mapping.keys()))

        # Read the channel labels of the stream
        channel_labels = []
        try:
            for channel in streams[stream_to_pos_mapping[stream_name]]['info']['desc'][0]['channels'][0]['channel']:
                channel_labels.append(channel['label'][0])
        except TypeError:
            # no channel information could be found!
            pass

        # Read the data and timestamps
        data_matrix = streams[stream_to_pos_mapping[stream_name]]['time_series']
        data_timestamps = streams[stream_to_pos_mapping[stream_name]]['time_stamps']

        return data_matrix, data_timestamps, channel_labels, stream_to_pos_mapping

    # ---------------------------------------------------------------------------------------

    @staticmethod
    def _get_marker_and_labels_from_stream(streams, exclude_trainings_trials, stream_to_pos_mapping=None):
        """
        Gets the markers and ground truths from the stream

        Args:
            streams: The streamfs from the xdf-files
            exclude_trainings_trials: True to exclude the training trials
        """

        marker_timestamps = []
        marker_names = []
        ground_truths = []
        durations = []
        start = None
        end = None
        go = False

        idx = stream_to_pos_mapping['ssvepMarkerStream']
        stream = streams[idx]

        # list of strings, draw one vertical line for each marker
        for timestamp, marker in zip(stream['time_stamps'], stream['time_series']):

            if marker[0] == 'phase_start' and marker[1] == 'Phase: Run Classification':
                go = True

            if go or (not exclude_trainings_trials):
                if marker[0] == "ground_truth":
                    ground_truths.append(marker[1])    

                elif marker[0] == "classification_start":      
                    if start is None:
                        start = timestamp
                    # marker_timestamps.append(timestamp)
                    marker_names.append(marker[0])

                elif marker[0] == "classification_end":
                    if end is None:
                        end = timestamp

                    marker_timestamps.append((start, end))
                    durations.append(end-start)
                    start = None
                    end = None

        return marker_timestamps, ground_truths

    # -----------------------------------------------------------------------------------------------

    @staticmethod
    def _create_stim_channel(data_timestamps, marker_timestamps, ground_truths):

        mts = [m[0] for m in marker_timestamps]
        stim = []

        idx = 0
        for t in data_timestamps:   

            if len(mts) > 0 and t >= mts[0]:
                ground_truth = ground_truths[idx] 
                stim.append(int(ground_truth)+1)
                mts.pop(0)
                idx += 1
            else:
                stim.append(0)

        return np.array([stim])

    # -----------------------------------------------------------------------------------------------

    @staticmethod
    def _add_stim_to_raw(raw, stim_data):
        """
        Adds the stim channel to the Raw-Object
        """

        # remove it if theres already a stim channel present (to rerun sections in notebook)
        if "STI" in raw.info["ch_names"]: 
            raw = raw.drop_channels("STI")

        info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(stim_data, info)
        raw.add_channels([stim_raw], force_update_info=True) 

    # -----------------------------------------------------------------------------------------------
