import global_settings as gs
from training import train

# --------------------------------------------------------------------

if __name__ == "__main__":

    print("\n"*10)

    create_new = False

    for k, dh in enumerate(gs.data_handlers):
        print(f"DHs: {k+1}/{len(gs.data_handlers)}")

        train.train_models(dh, gs.CV_params, num_epochs=gs.epochs, retrain_if_exist=create_new)

# --------------------------------------------------------------------