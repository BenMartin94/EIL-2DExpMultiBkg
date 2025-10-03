import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def training_histories():
    base_path = "lightning_logs/mbg_train_synth_test_cal_exp_"
    num_bkgs = [1, 5, 25, 50]
    paths = [base_path + str(n) + "bkgs" for n in num_bkgs]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for path, n_bkg in zip(paths, num_bkgs):
        # Find the latest version directory
        version_dirs = glob.glob(os.path.join(path, "version_*"))
        if not version_dirs:
            print(f"No versions found for {path}")
            continue
        
        # Sort by version number and get the latest
        version_dirs.sort(key=lambda x: int(x.split("_")[-1]))
        latest_version = version_dirs[-1]
        print(f"Loading {latest_version} for {n_bkg} backgrounds")
        
        # Find the tensorboard event file
        event_files = glob.glob(os.path.join(latest_version, "events.out.tfevents.*"))
        if not event_files:
            print(f"No event file found in {latest_version}")
            continue
        
        # Load the tensorboard data
        ea = event_accumulator.EventAccumulator(latest_version)
        ea.Reload()
        
        # Extract training and validation loss at epoch level
        try:
            train_loss = ea.Scalars('train_loss_epoch')
            val_loss = ea.Scalars('val_loss')
            
            # Use the actual epoch numbers (0-indexed, so we start from 0)
            train_epochs = list(range(len(train_loss)))
            train_values = [s.value for s in train_loss]
            val_epochs = list(range(len(val_loss)))
            val_values = [s.value for s in val_loss]
            
            # Plot training loss in log scale
            axes[0].plot(train_epochs, train_values, label=f'{n_bkg} bkgs', marker='o', markersize=4)
            
            # Plot validation loss in log scale
            axes[1].plot(val_epochs, val_values, label=f'{n_bkg} bkgs', marker='o', markersize=4)

        except KeyError as e:
            print(f"Could not find loss metrics in {latest_version}: {e}")
            print(f"Available scalars: {ea.Tags()['scalars']}")
    
    # Format the plots
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_yscale('log')  # Set log scale for y-axis
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss')
    axes[1].set_yscale('log')  # Set log scale for y-axis
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/training_histories.pdf', dpi=300, bbox_inches='tight')
    
    print("Training histories plot saved to figures/training_histories.pdf")


if __name__ == "__main__":
    training_histories()

