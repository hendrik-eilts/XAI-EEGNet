import os
import gc
import torch
import relevance
import global_settings as gs
import relevance.utils as rel_utils
from tqdm import tqdm

# ----------------------------------------------------------------------------------------

if __name__ == "__main__":

    create_new = False

    layer_num_filter_dict = {
        "b2_conv2": 16,
        # "b1_conv2": 16, 
        # "b2_conv1": 16,
        # "b1_conv1": 8,  
    }

    gs.use_which_data

    for dh in gs.data_handlers:

        if dh.ds_name == "CHO":
            batch_size = 256
        elif dh.ds_name == "GME":
            batch_size = 512
        else:
            batch_size = 32

        print("\n")
        print(">>", dh.ds_name)

        for params in tqdm(gs.CV_params):

            for layer, num_filters in layer_num_filter_dict.items():

                for class_label in gs.class_labels:

                    torch.cuda.empty_cache()
                    gc.collect()

                    path_base, path_attributions = \
                        relevance.results.Results.assemble_path(params, gs.base_path, gs.num_samples_to_select, dh.ds_name, layer, class_label,  
                                                                gs.select_samples_by_class, gs.reversed_classes, gs.select_correct,  
                                                                False, gs.use_which_data, testing=False)
                    paths = [path_base, path_attributions]
                    for path in paths:
                        if not os.path.exists(path):
                            os.makedirs(path)

                    rel_utils.prepare_data_for_filter(batch_size=batch_size, layer=layer, path_store_data=path_attributions, params=params, class_label=class_label,
                                                    dh=dh, num_filters=num_filters, composite=gs.composite, create_new=create_new,
                                                    use_which_data=gs.use_which_data, num_samples_to_select=gs.num_samples_to_select, reversed_classes=gs.reversed_classes,
                                                    select_correct=gs.select_correct, select_samples_by_class=gs.select_samples_by_class)

# ----------------------------------------------------------------------------------------
