import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------

def plot_loss(train_losses=None, train_accs=None, test_losses=None, test_accs=None, val_losses=None, val_accs=None,
              save_path=None, epochs_offset=0):

    x = list(np.arange(1, len(train_losses)+1))

    fig, ax = plt.subplots(1,2, figsize=(8, 3))

    if train_losses:
        ax[1].plot(x, train_losses, label='Train loss', linestyle="--", color=(0.5, 0, 0, 0.2))

    if train_accs:
        ax[0].plot(x, train_accs, label='Train Acc', color=(0, 0.5, 0, 0.2))

    if test_losses:
        ax[1].plot(x, test_losses, label='Test loss', linestyle="--", color=(0, 0, 0.5, 0.5))

    if test_accs:
        ax[0].plot(x, test_accs, label='Test Acc', color=(0, 0, 0.5, 0.5))

    if val_losses:
        ax[1].plot(x, val_losses, label='Val loss', linestyle="--", color=(1, 0, 0, 1))

    if val_accs:
        ax[0].plot(x, val_accs, label='Val Acc', color=(0, 1, 0, 1))

    ax[0].grid(color='0.95')
    ax[1].grid(color='0.95')

    ax[0].set_yticks([0.1*i for i in range(epochs_offset, 11)])
    ax[1].set_yticks([0.1*i for i in range(epochs_offset, 11)])

    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')

    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')

    ax[0].set_ylim(0, 1.05)
    ax[1].set_ylim(0, 1.05)

    # ax.legend()
    ax[0].legend() # loc='center left', bbox_to_anchor=(1, 0.8))
    ax[1].legend() # loc='center left', bbox_to_anchor=(1, 0.8))

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

# -----------------------------------------------------------------------
