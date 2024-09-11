import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from models import eeg_net
from training import utils
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.data_handler import DataHandler
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------

if torch.cuda.is_available():
    dtype_float = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

# -----------------------------------------------------------------------

def train_models(data_handler, CV_params, num_epochs=100, retrain_if_exist=False):

    for cv_index, params in enumerate(CV_params):

        batch_size = 32
        do = 0.25
        F1, F2 = 8, 16

        cv_index_old = cv_index

        if "nr" in params:
            cv_index = params['nr']

        print(f"({cv_index_old+1}/{len(CV_params)}), Nr. {cv_index}")

        if "bs" in params:
            batch_size = params['bs']

        if "conv_layers" in params:
            num_conv_layers = True #  params['conv_layers']
        else:
            num_conv_layers = None

        if "do" in params:
            do = params['do']

        if "seed" in params:
            seed = params['seed']
        else:
            seed = 42

        if seed != "_per_fold":
            torch.manual_seed(seed)
            seed_per_fold = False
        else:
            seed_per_fold = True

        # -------------

        file_name = "raw" + utils.dict_to_str(params)
        save_path = os.path.join(data_handler.model_path, file_name)

        data_shape = data_handler.get_transformed_data_shape()

        num_chn = data_shape[-2]
        num_time_points = data_shape[-1]

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cross_validation(data_handler, save_path=save_path, num_epochs=num_epochs, batch_size=batch_size,
                            params=params, chunk_size=num_time_points, num_chn=num_chn, F1=F1, F2=F2, 
                            dropout=do, num_conv_layers=num_conv_layers, retrain_if_exist=retrain_if_exist, 
                            seed_per_fold=seed_per_fold, cv_index=cv_index)
            
# -----------------------------------------------------------------------

def cross_validation(dh: DataHandler, params: dict, save_path=None, num_epochs=100, batch_size=32,
                     chunk_size=None, num_chn=None, F1=None, F2=None, dropout=None, num_conv_layers=None, retrain_if_exist=False, seed_per_fold=False, cv_index=0):

    file_name = utils.dict_to_str(params)
    tb_path = os.path.join(dh.ds_name, dh._create_dh_path(), file_name)

    writer = SummaryWriter(log_dir=os.path.join(".run", tb_path))

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    criterion = nn.CrossEntropyLoss()

    num_folds = dh.num_subjects

    for test_subj in tqdm(range(0, num_folds)):
        # print(f"{test_subj+1}/{dh.num_subjects}")

        individual_seed = (cv_index * num_folds) + test_subj

        if seed_per_fold:
            torch.manual_seed(individual_seed)
            # print("seed per fold:", individual_seed, "(", cv_index, "*", num_folds, "+", test_subj, ")")
            
        # --------------------------

        if not retrain_if_exist:

            # def load_model(dh, params, num_chn, num_data_points, subject=None, return_dict=False, show_errors=True, 
            #        use_min_val_loss=False, epochs=None, num_conv_layers=None):
            
            model = utils.load_model(dh, params, num_chn, chunk_size, test_subj, epochs=num_epochs, num_conv_layers=num_conv_layers)            

            if model is not None:
                continue

        # --------------------------

        train_ids = [i for i in range(0, dh.num_subjects)]
        train_ids.remove(test_subj)
        
        if num_conv_layers is None:
            model = eeg_net.EEGNet_Torch(chunk_size=chunk_size, num_electrodes=num_chn, F1=F1, F2=F2, D=2, num_classes=2, dropout=dropout)
        else:
            model = eeg_net.EEGNet_Torch_1_ConvLayer(chunk_size=chunk_size, num_electrodes=num_chn, F1=F1, F2=F2, D=2, num_classes=2, dropout=dropout)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model.to(device)
        optimizer = optim.Adam(model.parameters())

        X_train = []
        y_train = []

        X_test = None
        y_test = None

        for subj in range(dh.num_subjects):
            data = dh.get_X_y_separate(subj)

            if data is None:
                break

            X_subj, y_subj, _ = data

            if type(y_subj) != np.ndarray:
                y_subj = np.array(y_subj)

            X_subj = torch.from_numpy(X_subj).type(dtype_float)
            y_subj = torch.from_numpy(y_subj).type(dtype_long)

            if subj in train_ids:
                X_train.append(X_subj)
                y_train.append(y_subj)
            else:
                X_test = X_subj
                y_test = y_subj

        X_train = torch.cat(X_train)
        y_train = torch.cat(y_train)
        train_ds = torch.utils.data.TensorDataset(X_train, y_train.long())
        train_loader = DataLoader(train_ds, batch_size=batch_size)

        test_ds = torch.utils.data.TensorDataset(X_test, y_test.long())
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        train_losses = []
        test_losses = []
        val_losses = []

        train_accs = []
        test_accs = []
        val_accs = []

        stop_training = False

        for epoch in range(num_epochs):

            torch.cuda.empty_cache()

            model, train_acc, train_loss = train_loop(model, train_loader, optimizer, criterion, rereference=True, standardize=True)
            
            test_acc, test_loss = eval_loop(model, test_loader, criterion, rereference=True, standardize=True)

            train_losses.append(train_loss)
            train_accs.append(train_acc)

            test_losses.append(test_loss)
            test_accs.append(test_acc)

            writer.add_scalars(f"subj_{test_subj}/acc", {f"train":train_acc}, epoch)
            writer.add_scalars(f"subj_{test_subj}/acc", {f"test":test_acc}, epoch)

            writer.add_scalars(f"subj_{test_subj}/loss", {f"train":train_loss}, epoch)
            writer.add_scalars(f"subj_{test_subj}/loss", {f"test":test_loss}, epoch)
    
            if (epoch+1) % 100 == 0 or stop_training or (epoch+1) == num_epochs:
                utils.save_model(model, optimizer, train_losses, train_accs, test_losses, test_accs, val_losses, val_accs, num_epochs=epoch+1, 
                                    save_path=os.path.join(save_path, f"checkpoint_{test_subj}_{epoch+1}"))
                
                utils.save_model(model, optimizer, train_losses, train_accs, test_losses, test_accs, val_losses, val_accs, num_epochs=epoch+1, 
                                    save_path=os.path.join(save_path, f"checkpoint_{test_subj}"))

    writer.flush()
    writer.close()

# -----------------------------------------------------------------------

def train_loop(model, train_loader, optimizer, criterion, rereference=True, standardize=True):

    model.train()
    epoch_loss = []

    if torch.cuda.is_available():
        model.cuda()

    for batch_idx, (data, targets) in enumerate(train_loader):
        correct = 0
        total = 0

        # apply rereferencing
        if rereference:
            data = utils.apply_common_average_rereferencing(data)
 
        if standardize:
            data = utils.standardize_data(data)

        if type(data) == np.ndarray:
            data = torch.from_numpy(data).type(dtype_float) 

        # data, targets = data.to(device), targets.to(device)
        data, targets = data.type(dtype_float), targets.type(dtype_long)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    accuracy = correct / total

    epoch_loss = np.array(epoch_loss)

    return model, accuracy, np.sum(epoch_loss)/len(epoch_loss)

# -----------------------------------------------------------------------

def eval_loop(model, test_loader, criterion, rereference=True, standardize=True, return_preds_and_targets=False):
    # Evaluate the model

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    epoch_loss = []

    if return_preds_and_targets:
        predictions_list, targets_list = [], []

    with torch.no_grad():
        correct = 0
        total = 0

        # for data, targets in test_loader:
        for batch_idx, (data, targets) in enumerate(test_loader):

            # apply rereferencing
            if rereference:
                data = utils.apply_common_average_rereferencing(data)

            # apply standardization
            if standardize:
                data = utils.standardize_data(data)

            data_np = data.detach().cpu().numpy()
            data_np = data_np.astype('float64')
       
            if type(data) == np.ndarray:
                data = torch.from_numpy(data).type(dtype_float) 

            # data, targets = data.to(device), targets.to(device)
            data, targets = data.type(dtype_float), targets.type(dtype_long)

            outputs = model(data)

            loss = criterion(outputs, targets)

            loss = loss.detach().cpu().numpy()

            epoch_loss.append(loss)

            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if return_preds_and_targets:
                # print(type(predicted), type(targets))
                predictions_list.append(predicted.tolist())
                targets_list.append(targets.tolist())

        accuracy = correct / total

    # numpy_array = tensor.detach().cpu().numpy()
    epoch_loss = np.array(epoch_loss)

    if return_preds_and_targets:
        return accuracy, np.sum(epoch_loss)/len(epoch_loss), predictions_list, targets_list
    else:
        return accuracy, np.sum(epoch_loss)/len(epoch_loss)

# -----------------------------------------------------------------------
