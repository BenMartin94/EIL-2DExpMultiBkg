import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tensorboard.backend.event_processing import event_accumulator


# Global font size configuration
AXIS_LABEL_FONTSIZE = 14
AXIS_TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 16
TITLE_FONTSIZE = 18


def training_histories():
    base_path = "lightning_logs/mbg_train_synth_test_cal_exp_"
    num_bkgs = [1, 2, 4, 5, 6, 25, 50]
    paths = [base_path + str(n) + "bkgs" for n in num_bkgs]
    
    # Create output directory
    os.makedirs('figures/histories', exist_ok=True)
    
    # Create three figures: one for epochs, one for steps, one for clipped steps (validation loss only)
    fig_epochs, ax_epochs = plt.subplots(1, 1, figsize=(8, 6))
    fig_steps, ax_steps = plt.subplots(1, 1, figsize=(8, 6))
    fig_steps_clipped, ax_steps_clipped = plt.subplots(1, 1, figsize=(8, 6))

    min_loss = np.inf
    max_loss = -np.inf
    
    # Store data for clipped steps plot
    all_step_data = []
    min_steps = np.inf
    
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
            min_loss = min(min_loss, min(val_values))
            max_loss = max(max_loss, max(val_values))
            
            # Plot on validation loss figures only
            ax_epochs.plot(val_epochs, val_values, label=f'{n_bkg} bkgs', marker='o', markersize=4)
            ax_steps.plot(val_steps, val_step_values, label=f'{n_bkg} bkgs', marker='o', markersize=4)
            
            # Store data for clipped steps plot
            all_step_data.append({
                'n_bkg': n_bkg,
                'val_steps': val_steps,
                'val_step_values': val_step_values
            })
            min_steps = min(min_steps, len(val_steps))

        except KeyError as e:
            print(f"Could not find loss metrics in {latest_version}: {e}")
            print(f"Available scalars: {ea.Tags()['scalars']}")

    
    # Plot clipped steps data
    print(f"\nClipping all experiments to {min_steps} steps")
    for data in all_step_data:
        clipped_steps = data['val_steps'][:min_steps]
        clipped_values = data['val_step_values'][:min_steps]
        ax_steps_clipped.plot(clipped_steps, clipped_values, label=f'{data["n_bkg"]} bkgs', marker='o', markersize=4)
    
    # Set consistent y-limits based on min/max loss
    y_min = min_loss * 0.9
    y_max = max_loss * 1.1
    ax_epochs.set_ylim(y_min, y_max)
    ax_steps.set_ylim(y_min, y_max)
    ax_steps_clipped.set_ylim(y_min, y_max)
    
    # Format the epoch-based plot (validation loss only)
    ax_epochs.set_xlabel('Epoch', fontsize=AXIS_LABEL_FONTSIZE)
    ax_epochs.set_ylabel('Loss', fontsize=AXIS_LABEL_FONTSIZE)
    ax_epochs.set_title('Validation Loss', fontsize=TITLE_FONTSIZE)
    ax_epochs.set_yscale('log')
    ax_epochs.legend(fontsize=LEGEND_FONTSIZE)
    ax_epochs.grid(True, alpha=0.3)
    ax_epochs.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONTSIZE)

    fig_epochs.tight_layout()
    fig_epochs.savefig('figures/histories/training_histories.pdf', dpi=300, bbox_inches='tight')
    
    # Format the step-based plot (validation loss only)
    ax_steps.set_xlabel('Training Step', fontsize=AXIS_LABEL_FONTSIZE)
    ax_steps.set_ylabel('Loss', fontsize=AXIS_LABEL_FONTSIZE)
    ax_steps.set_title('Validation Loss vs Steps', fontsize=TITLE_FONTSIZE)
    ax_steps.set_yscale('log')
    ax_steps.legend(fontsize=LEGEND_FONTSIZE)
    ax_steps.grid(True, alpha=0.3)
    ax_steps.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONTSIZE)
    ax_steps.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

    fig_steps.tight_layout()
    fig_steps.savefig('figures/histories/training_histories_by_steps.pdf', dpi=300, bbox_inches='tight')
    
    # Format the clipped step-based plot (validation loss only)
    ax_steps_clipped.set_xlabel('Training Step', fontsize=AXIS_LABEL_FONTSIZE)
    ax_steps_clipped.set_ylabel('Loss', fontsize=AXIS_LABEL_FONTSIZE)
    ax_steps_clipped.set_title('Validation Loss vs Training Steps', fontsize=TITLE_FONTSIZE)
    ax_steps_clipped.set_yscale('log')
    ax_steps_clipped.legend(fontsize=LEGEND_FONTSIZE)
    ax_steps_clipped.grid(True, alpha=0.3)
    ax_steps_clipped.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONTSIZE)
    ax_steps_clipped.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    ax_steps_clipped.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONTSIZE)

    fig_steps_clipped.tight_layout()
    fig_steps_clipped.savefig('figures/histories/training_histories_by_steps_clipped.pdf', dpi=300, bbox_inches='tight')

    print("\nValidation loss plot saved to figures/histories/training_histories.pdf")
    print("Validation loss by steps plot saved to figures/histories/training_histories_by_steps.pdf")
    print("Validation loss by steps (clipped) plot saved to figures/histories/training_histories_by_steps_clipped.pdf")




def training_histories_2x2():
    """Create a 4x1 plot showing training and validation loss per step for selected backgrounds."""
    base_path = "lightning_logs/mbg_train_synth_test_cal_exp_"
    selected_bkgs = [1, 2, 5, 25]
    
    # Create output directory
    os.makedirs('figures/histories', exist_ok=True)
    
    # Create 4x1 subplot
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))
    
    # Track min/max loss across all plots for consistent y-axis
    min_loss = np.inf
    max_loss = -np.inf
    
    # Store data for all subplots first
    plot_data = []
    
    for idx, n_bkg in enumerate(selected_bkgs):
        path = base_path + str(n_bkg) + "bkgs"
        
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
        
        # Extract training and validation loss at step level
        try:
            train_loss_step = ea.Scalars('train_loss_step')
            val_loss = ea.Scalars('val_loss')
            
            # Data for step-based plots
            train_steps = [s.step for s in train_loss_step]
            train_step_values = [s.value for s in train_loss_step]
            val_steps = [s.step for s in val_loss]
            val_step_values = [s.value for s in val_loss]
            
            # Update min/max for y-axis scaling
            all_values = train_step_values + val_step_values
            min_loss = min(min_loss, min(all_values))
            max_loss = max(max_loss, max(all_values))
            
            # Store data for plotting later
            plot_data.append({
                'idx': idx,
                'n_bkg': n_bkg,
                'train_steps': train_steps,
                'train_values': train_step_values,
                'val_steps': val_steps,
                'val_values': val_step_values
            })
            
            print(f"  Loaded data for {n_bkg} backgrounds")
            
        except KeyError as e:
            print(f"Could not find loss metrics in {latest_version}: {e}")
            print(f"Available scalars: {ea.Tags()['scalars']}")
    
    # Set consistent y-limits for all subplots
    y_min = min_loss * 0.9
    y_max = max_loss * 1.1
    
    # Now plot all data with consistent y-axis
    for data in plot_data:
        ax = axes[data['idx']]
        ax.plot(data['train_steps'], data['train_values'], label='Training Loss', alpha=0.7, linewidth=1.5)
        ax.plot(data['val_steps'], data['val_values'], label='Validation Loss', marker='o', markersize=3, linewidth=1.5)
        
        ax.set_xlabel('Training Step', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel('Loss', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(f'{data["n_bkg"]} Background{"s" if data["n_bkg"] > 1 else ""}', fontsize=TITLE_FONTSIZE, fontweight='bold')
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONTSIZE)
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        print(f"  Plotted {data['n_bkg']} backgrounds on subplot {data['idx'] + 1}")
    
    fig.tight_layout()
    fig.savefig('figures/histories/training_validation_loss_4x1.pdf', dpi=300, bbox_inches='tight')
    print("\n4x1 training and validation loss plot saved to figures/histories/training_validation_loss_4x1.pdf")


if __name__ == "__main__":
    training_histories()
    training_histories_2x2()

