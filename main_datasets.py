import global_settings as gs


if __name__ == "__main__":

    for dh in gs.data_handlers:
       print("\n", "Process data for:", dh.ds_name)
       
       dh.load_and_preprocess_data(create_new_if_exists=False)

