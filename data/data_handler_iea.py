import numpy as np
import mne
import os
import joblib
from tqdm import tqdm
from data.data_handler import DataHandler


class DataHandlerIEA(DataHandler):
         
    def __init__(self, window_length, step_size, use_ica, base_result_path, data_path, processed_data_path, test=False,
                 resample=False, sampling_rate=None, locs_path="", num_classes=2, use_eegnet_torch=False, ds_name="IEA"):

        super(DataHandlerIEA, self).__init__(use_ica, base_result_path, sampling_rate, test, resample, ds_name,
                                             step_size, window_length, num_classes, processed_data_path=processed_data_path)

        self.channel_names = ['Cz', 'Fp2', 'F3', 'FT7', 'C3', 'C4', 'FT8', 'P3', 'P4', 'PO7', 'PO8', 'Oz']
        self.data_path = data_path
        self.events_list = None
        self.locs_path = locs_path
        self.num_subjects = 13
        self.use_eegnet_torch = use_eegnet_torch

    # ---------------------------------------------------------------------------------------

    def load_raws(self, verbose=0):
        
        filenames = ["01", "16", "22", "34", "35", "36", "51", "63", "72", "77", "79", "87", "94"]

        if self.test:
            filenames = filenames[:2]
        
        for idx, fn in enumerate(filenames):

            filenames[idx] = \
                (os.path.join(self.data_path, f"iea_{fn}", f"iea_{fn}-raw.fif"),
                 os.path.join(self.data_path, f"iea_{fn}", f"iea_{fn}-eve.fif"))

        raw_list = []
        events_list = []
        
        for idx, (fn_raw, fn_eve) in enumerate(filenames):
            raw = mne.io.read_raw_fif(fn_raw, preload=True, verbose=verbose)
            events = mne.read_events(fn_eve)
            raw_list.append((raw, idx))
            events_list.append(events)

        self.events_list = events_list
        return raw_list

    # ---------------------------------------------------------------------------------------    

    def preprocess_raws(self, raw_list, verbose=0, plot=False):

        events_resampled = []

        for (raw, idx), events in tqdm(zip(raw_list, self.events_list)):
            if verbose: 
                print("Before preprocessing\n")
                print(raw.info)
                print(raw.info.ch_names)
            
            if plot and idx == 0:
                raw.plot()
            
            # select channels
            if "P4-0" in raw.info['ch_names']:
                mne.rename_channels(raw.info, {'P4-0': 'P4'})
            if "CZ" in raw.info['ch_names']:
                mne.rename_channels(raw.info, {'CZ': 'Cz'})
            if "FP2" in raw.info['ch_names']:
                mne.rename_channels(raw.info, {'FP2': 'Fp2'})
            if "FP1" in raw.info['ch_names']:
                mne.rename_channels(raw.info, {'FP1': 'Fp1'})
            if "OZ" in raw.info['ch_names']:
                mne.rename_channels(raw.info, {'OZ': 'Oz'})

            raw.pick_channels(self.channel_names)
                        
            # notch filter
            raw.notch_filter(np.arange(50, 201, 50), fir_design='firwin')

            raw.filter(1, 60)

            if self.resample:
                if verbose:
                    print("resampling to ", self.sampling_rate)
                raw, events = raw.resample(self.sampling_rate, events=events)
                events_resampled.append(events)
            else:
                events_resampled.append(events)

            # average ref
            #if self.avg_ref:
            #    raw = raw.set_eeg_reference(ref_channels="average")

            #if self.scale:
            #    raw = raw.apply_function(utils.standardize_data, channel_wise=False)

            if verbose:
                print("After preprocessing\n")
                print(raw.info)
                print(".......................")
            
            if plot and idx == 0:
                raw.plot()

        self.events_list = events_resampled

        return raw_list

    # ---------------------------------------------------------------------------------------    
    
    def create_epochs(self, raw_list, verbose=0, plot=False):
        epochs_list = []
        event_id = {'internal': 7, 'external': 5}

        # Define the time window to extract for each epoch
        tmin, tmax = 1.0, 13.0
        
        for (raw, idx), events in tqdm(zip(raw_list, self.events_list)):

            epochs = mne.Epochs(raw, events=events, event_id=event_id,
                                tmin=tmin, tmax=tmax, 
                                baseline=None, preload=True)
            
            y = (epochs.events[:, 2]).astype(np.int64)

            # 5 to 1, 7 to 0
            y = [0 if val == 7 else 1 for val in y]
            
            if verbose: 
                print(epochs.info)
                print(epochs)
                print("These events were dropped:")
                print(epochs.drop_log)
            if plot:
                epochs.plot()
                
            epochs_list.append((epochs, y, idx))
                        
        epochs_list = epochs_list
        joblib.dump(epochs_list, self.epochs_path)

        return epochs_list

    # -----------------------------------------------------------------------------------------------

    def _load_files_into_epochs(self, iea_data_path):
        
        # load epochs
        subjects = ["01", "16", "22", "34", "35", "36", "51", "63", "72", "77", "79", "87", "94"]

        if self.test: 
            subjects = subjects[:6]
            
        self.epochs = []
        verbose = 0
        file = "epo"

        self.file_names = []
        
        for nr in subjects:
            subj_path = os.path.join(iea_data_path, f"iea_{nr}")
            path = os.path.join(subj_path, f"iea_{nr}-{file}.fif")
            epo = mne.read_epochs(path, verbose=verbose)
            self.file_names.append(f"{nr}_{file}")
            self.epochs.append(epo)

        chns_count = dict([('Cz', 0), ('Fp2', 0), ('F3', 0), ('F4', 0), ('FT7', 0), ('C3', 0), ('C4', 0), ('Fp1', 0),
                           ('FT8', 0), ('P3', 0), ('PZ', 0), ('P4', 0), ('PO7', 0), ('PO8', 0), ('Oz', 0), ('FZ', 0)])

        for epo in self.epochs:
            if "P4-0" in epo.info['ch_names']:
                mne.rename_channels(epo.info, {'P4-0': 'P4'})
            if "CZ" in epo.info['ch_names']:
                mne.rename_channels(epo.info, {'CZ': 'Cz'})
            if "FP2" in epo.info['ch_names']:
                mne.rename_channels(epo.info, {'FP2': 'Fp2'})
            if "FP1" in epo.info['ch_names']:
                mne.rename_channels(epo.info, {'FP1': 'Fp1'})
            if "OZ" in epo.info['ch_names']:
                mne.rename_channels(epo.info, {'OZ': 'Oz'})

            for ch in epo.info['ch_names']:
                try:
                    chns_count[ch] += 1
                except:
                    print(ch, "not in dict")

        select = []
        for i, ch in enumerate(chns_count):
            if chns_count[ch]/len(self.epochs) == 1:
                select.append(ch)    
        
        for epo in self.epochs:
            epo = epo.pick_channels(select)
            epo.apply_baseline((1, 1.5))
