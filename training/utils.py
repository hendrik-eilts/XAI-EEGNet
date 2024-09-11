import os
import torch
import numpy as np
import training

from models import eeg_net

# -----------------------------------------------------------------------

def load_model(dh, params, num_chn, num_data_points, subject=None, return_dict=False, show_errors=False, 
               use_min_val_loss=False, epochs=None, num_conv_layers=None):

    file_name = "raw" + training.utils.dict_to_str(params)
    load_path = os.path.join(dh.model_path, file_name)

    # ----------------

    if "do" in params:
        do = params['do']
    else:
        do = 0.25

    F1, F2 = 8, 16

    if num_conv_layers is None:
        model = eeg_net.EEGNet_Torch(chunk_size=num_data_points, num_electrodes=num_chn, F1=F1, F2=F2, D=2, num_classes=2, dropout=do)

    else:
        print(">>> NUM CONV LAYERS == 1 <<<")
        model = eeg_net.EEGNet_Torch_1_ConvLayer(chunk_size=num_data_points, num_electrodes=num_chn, F1=F1, F2=F2, D=2, num_classes=2, dropout=do)

    if use_min_val_loss:
        model_name = f"checkpoint_{subject}_min_val_loss"
    else:
        if epochs:
            model_name = f"checkpoint_{subject}_{epochs}"
        else:
            model_name = f"checkpoint_{subject}"

    path = os.path.join(load_path, model_name)
    
    try:
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()

    except Exception as e:
        if show_errors:
            print(f"load_model(...) file not found error, path: {path}, error message: {e}")
        model = checkpoint = None

    if not return_dict:
        return model
    else:
        return model, checkpoint

# -----------------------------------------------------------------------

def standardize_data(data, numpy=False):
    # Reshape the data by concatenating the samples along the time steps dimension
    reshaped_data = data.reshape(data.shape[0], -1)

    if numpy:
        mean = np.mean(reshaped_data, axis=1, keepdims=True)
        sd = np.std(reshaped_data, axis=1, keepdims=True)
    else:
        mean = torch.mean(reshaped_data, dim=1, keepdim=True)
        sd = torch.std(reshaped_data, dim=1, keepdim=True)

    standardized_data = (data - mean[:, None, :]) / sd[:, None, :]
    return standardized_data

# -----------------------------------------------------------------------

def apply_common_average_rereferencing(eeg_data, numpy=False):

    if numpy:
        # Calculate the average across all samples and channels
        # average_reference = np.mean(eeg_data, axis=(0, 1), keepdims=True)
        average_reference = np.mean(eeg_data, axis=(1), keepdims=True)
    else:
        # average_reference = torch.mean(eeg_data, dim=(0, 1), keepdim=True)
        average_reference = torch.mean(eeg_data, dim=(1), keepdim=True)

    # Subtract the average reference from each channel
    rereferenced_data = eeg_data - average_reference

    return rereferenced_data

# -----------------------------------------------------------------------

def dict_to_str(d):
    output = ""
    for k, v in d.items():
        output += f"_{k}{str(v)}"
    return output

# -----------------------------------------------------------------------

def save_model(model, optimizer, train_losses, train_accs, test_losses, test_accs, val_losses, val_accs, num_epochs, save_path, val_subjs=None, subj_test_data_dict=None):

    if save_path is not None:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),

                    'train_accs': train_accs,
                    'train_loss': train_losses,

                    'test_accs': test_accs,
                    'test_loss': test_losses,

                    'val_accs': val_accs,
                    'val_loss': val_losses,

                    'num_epochs': num_epochs, 
                    'val_subjs': val_subjs,

                    'subj_test_data_dict': subj_test_data_dict

                    }, save_path)

# -----------------------------------------------------------------------
