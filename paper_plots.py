import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def training_histories():
    base_path = "lightning_logs/mbg_train_synth_test_cal_exp_"
    num_bkgs = [1, 2, 3, 4, 5, 6, 25, 50]
    paths = [base_path + str(n) + "bkgs" for n in num_bkgs]
    
    # Create output directory for individual plots
    os.makedirs('figures/histories', exist_ok=True)
    
    # Create two figures: one for epochs, one for steps (original combined plots)
    fig_epochs, axes_epochs = plt.subplots(1, 2, figsize=(14, 5))
    fig_steps, axes_steps = plt.subplots(1, 2, figsize=(14, 5))

    min_loss = np.inf
    max_loss = -np.inf
    
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
        
        # Load the tensorboard data with unlimited size
        size_guidance = {
            event_accumulator.SCALARS: 0,  # 0 means load all scalars
        }
        ea = event_accumulator.EventAccumulator(latest_version, size_guidance=size_guidance)
        ea.Reload()

        # Extract training and validation loss at epoch level
        try:
            train_loss_step = ea.Scalars('train_loss_step')
            train_loss = ea.Scalars('train_loss_epoch')
            val_loss = ea.Scalars('val_loss')
            
            # Get the actual training steps from the last logged step
            if train_loss_step:
                actual_training_steps = train_loss_step[-1].step + 1  # +1 because steps are 0-indexed
                logged_steps = len(train_loss_step)
                print(f"  Logged training steps: {logged_steps}")
                print(f"  Actual training steps: {actual_training_steps}")
            
            # Data for epoch-based plots
            train_epochs = list(range(len(train_loss)))
            train_values = [s.value for s in train_loss]
            val_epochs = list(range(len(val_loss)))
            val_values = [s.value for s in val_loss]
            
            # Data for step-based plots
            train_steps = [s.step for s in train_loss_step]
            train_step_values = [s.value for s in train_loss_step]
            val_steps = [s.step for s in val_loss]
            val_step_values = [s.value for s in val_loss]

            # Update min/max loss for y-axis scaling
            min_loss = min(min_loss, min(train_values + val_values))
            max_loss = max(max_loss, max(train_values + val_values))
            
            # Plot on original combined epoch-based figure
            axes_epochs[0].plot(train_epochs, train_values, label=f'{n_bkg} bkgs', marker='o', markersize=4)
            axes_epochs[1].plot(val_epochs, val_values, label=f'{n_bkg} bkgs', marker='o', markersize=4)
            
            # Plot on original combined step-based figure
            axes_steps[0].plot(train_steps, train_step_values, label=f'{n_bkg} bkgs', alpha=0.7, linewidth=1)
            axes_steps[1].plot(val_steps, val_step_values, label=f'{n_bkg} bkgs', marker='o', markersize=4)

            # Create individual plot for this experiment (epoch-based only)
            fig_individual, ax_individual = plt.subplots(1, 1, figsize=(8, 6))
            ax_individual.plot(train_epochs, train_values, label='Training Loss', marker='o', markersize=4)
            ax_individual.plot(val_epochs, val_values, label='Validation Loss', marker='s', markersize=4)
            ax_individual.set_xlabel('Epoch')
            ax_individual.set_ylabel('Loss')
            ax_individual.set_title(f'Training & Validation Loss ({n_bkg} backgrounds)')
            ax_individual.set_yscale('log')
            ax_individual.legend()
            ax_individual.grid(True, alpha=0.3)
            fig_individual.tight_layout()
            fig_individual.savefig(f'figures/histories/training_history_{n_bkg}bkgs.pdf', dpi=300, bbox_inches='tight')
            plt.close(fig_individual)
            
            print(f"  Saved individual plot for {n_bkg} backgrounds")

        except KeyError as e:
            print(f"Could not find loss metrics in {latest_version}: {e}")
            print(f"Available scalars: {ea.Tags()['scalars']}")

    
    # Set consistent y-limits based on min/max loss for original combined plots
    y_min = min_loss * 0.9
    y_max = max_loss * 1.1
    for ax in axes_epochs:
        ax.set_ylim(y_min, y_max)
    for ax in axes_steps:
        ax.set_ylim(y_min, y_max)
    
    # Format the original epoch-based plots
    axes_epochs[0].set_xlabel('Epoch')
    axes_epochs[0].set_ylabel('Loss')
    axes_epochs[0].set_title('Training Loss')
    axes_epochs[0].set_yscale('log')
    axes_epochs[0].legend()
    axes_epochs[0].grid(True, alpha=0.3)
    
    axes_epochs[1].set_xlabel('Epoch')
    axes_epochs[1].set_ylabel('Loss')
    axes_epochs[1].set_title('Validation Loss')
    axes_epochs[1].set_yscale('log')
    axes_epochs[1].legend()
    axes_epochs[1].grid(True, alpha=0.3)

    fig_epochs.tight_layout()
    fig_epochs.savefig('figures/histories/training_histories.pdf', dpi=300, bbox_inches='tight')
    
    # Format the original step-based plots
    axes_steps[0].set_xlabel('Training Step')
    axes_steps[0].set_ylabel('Loss')
    axes_steps[0].set_title('Training Loss vs Steps')
    axes_steps[0].set_yscale('log')
    axes_steps[0].legend()
    axes_steps[0].grid(True, alpha=0.3)
    
    axes_steps[1].set_xlabel('Training Step')
    axes_steps[1].set_ylabel('Loss')
    axes_steps[1].set_title('Validation Loss vs Steps')
    axes_steps[1].set_yscale('log')
    axes_steps[1].legend()
    axes_steps[1].grid(True, alpha=0.3)

    fig_steps.tight_layout()
    fig_steps.savefig('figures/histories/training_histories_by_steps.pdf', dpi=300, bbox_inches='tight')

    print("\nOriginal training histories plot saved to figures/histories/training_histories.pdf")
    print("Original training histories by steps plot saved to figures/histories/training_histories_by_steps.pdf")
    print("Individual plots saved to figures/histories/")




if __name__ == "__main__":
    training_histories()

