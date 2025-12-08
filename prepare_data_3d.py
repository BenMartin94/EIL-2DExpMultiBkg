import numpy as np
import os
import h5py
import torchvision  # Replaces keras
from skimage.transform import resize

class Dataset:
    def __init__(self, fields, targets, num_targets, num_freqs, num_recvers, num_sources):
        self.fields = fields
        self.targets = targets
        self.num_targets = num_targets
        self.num_freqs = num_freqs
        self.num_recvers = num_recvers
        self.num_sources = num_sources


def change_resolution(targets, new_dimensions):
    # targets shape: (height, width, num_images)
    rescaled_targets = np.empty((targets.shape[0], new_dimensions[0], new_dimensions[1]), dtype=np.float64)
    for i in range(targets.shape[0]):
        image = targets[i, :, :]
        rescaled_image = resize(image, new_dimensions, anti_aliasing=True)
        rescaled_targets[i, :, :] = rescaled_image
    return rescaled_targets

def create_3d_mnist_data(target_file):
    ### extract target information from target_file to create 3d mnist targets

    ### load 2d targets
    with h5py.File(target_file, 'r') as target_data:
        targets = np.array(target_data['target_info'])      # (num_targets, 5)
        # targets = targets.T             # remove this --- just for when called from test.py !!!!!
        print(target_data.keys())
        num_targets = targets.shape[0]
        targets_permittivity = targets[:, 4]

    target_indices = targets[:, 0]
    target_indices = target_indices.astype(int)
    target_indices = target_indices - 1    # to align with indices used in julia

    # (x_train, _), (_, _) = keras.datasets.mnist.load_data()
    # Load MNIST data using torchvision, ensuring it's downloaded to a local './data' directory
    mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    x_train = mnist_trainset.data.numpy()  # Convert tensor to numpy array

    targets = x_train[target_indices]

    ### change targets to 3d by expanding along one axis, and increase resolution
    final_targets = np.ones((num_targets, 56, 56, 56), dtype=np.float32)
    padding_size = 3        # same as in forward solve (maybe should have saves number ?)
    targets = change_resolution(targets, (56, 56))

    for target in range(num_targets):
        current_target = targets[target, :, :]

        # set permittivity to match forward solve
        permittivity = targets_permittivity[target] - 2 # (already 1 where the target is, and add 1 everywhere later)
        mask = current_target > 0.0
        current_target[mask] += permittivity
        
        # extrude along dimension
        depth = 56 - 2 * padding_size
        extruded = np.repeat(current_target[:, :, np.newaxis], depth, axis=2)
        
        # zero pad along extruded dim to get cube 
        padded = np.pad(
            extruded,
            pad_width=((0, 0), (0, 0), (padding_size, padding_size)),
            mode='constant',
            constant_values=0
        )  # shape=(112, 112, 112)
        
        final_targets[target, :, :, :] = padded
        final_targets[target] += 1

    return final_targets



# for multifrequency data generated from julia solver, saved as a .mat file
#    returns train_data and test_data as Dataset objects
#    Field data format: (N, 2, S, R) where 2 = real and imag channels
def process_multifreq_data(fields_file, targets_file, targetless_fields_file, num_test, max_samples=None):
    print("Processing data...")
    # load field data
    with h5py.File(fields_file, 'r') as field_data:
        fields = np.array(field_data['fields'])     # (num_targets, num_freqs, num_sources, num_recvers)  
        freqs = np.array(field_data['freqs'])
        num_freqs = np.size(freqs)
        num_targets = np.array(field_data['num_targets'])
        num_recvers = len(fields[0, 0, 0, :])
        num_sources = len(fields[0, 0, :, 0])
        print("recvers: ", num_recvers)
        print("sources: ", num_sources)

    # Convert to single frequency data (2.5GHz) - take frequency index 7 (14:16 are real/imag)
    # fields is (num_targets, num_freqs, num_sources, num_recvers)
    # We want (num_targets, 2, num_sources, num_recvers) where 2 = [real, imag]
    freq_idx = 7  # 2.5GHz
    
    # HDF5 stores complex as structured array with 'real' and 'imag' fields
    fields_real = fields[:, freq_idx, :, :]['real']  # (num_targets, num_sources, num_recvers)
    fields_imag = fields[:, freq_idx, :, :]['imag']  # (num_targets, num_sources, num_recvers)
    
    # Stack real and imag as channels: (num_targets, 2, num_sources, num_recvers)
    field_data = np.stack([fields_real, fields_imag], axis=1)
    num_freqs = 1

    # load targetless field data
    with h5py.File(targetless_fields_file, 'r') as targetless_field_data:
        targetless_fields = np.array(targetless_field_data['fields'])     # (1, num_freqs, num_sources, num_recvers) 
    
    # Extract same frequency for targetless - access structured array fields
    targetless_real = targetless_fields[0, freq_idx, :, :]['real']  # (num_sources, num_recvers)
    targetless_imag = targetless_fields[0, freq_idx, :, :]['imag']  # (num_sources, num_recvers)
    targetless_field_data = np.stack([targetless_real, targetless_imag], axis=0)  # (2, num_sources, num_recvers)

    # convert total fields into scattered fields by subtracting fields with no targets
    # Broadcasting: (num_targets, 2, num_sources, num_recvers) - (2, num_sources, num_recvers)
    field_data = field_data - targetless_field_data[np.newaxis, :, :, :]

    print("Field data shape: ", field_data.shape)  # (num_targets, 2, num_sources, num_recvers)

    # num_sources = 24 and recvers = 72. Make it square by duplicating sources 3x
    # Concatenate along source axis: (num_targets, 2, num_sources*3, num_recvers)
    field_data = np.concatenate([field_data, field_data, field_data], axis=2)
    num_sources = num_sources * 3

    print("Final field data shape: ", field_data.shape)  # (num_targets, 2, 72, 72)

    # load targets
    ### for providing physical targets (not just target info)
    # with h5py.File(targets_file, 'r') as target_data:
    #     targets = np.array(target_data['targets'])      # (num_targets, 112, 112)
    
    # Cache 3D MNIST targets to avoid rebuilding every time
    cache_path = "./data/3d/cached_3d_mnist_targets.npy"
    if os.path.exists(cache_path):
        print("Loading cached 3D MNIST targets...")
        targets = np.load(cache_path)
    else:
        print("Creating 3D MNIST targets (this may take a moment)...")
        targets = create_3d_mnist_data(targets_file)
        os.makedirs("./data/3d", exist_ok=True)
        np.save(cache_path, targets)
        print(f"Cached 3D MNIST targets saved to {cache_path}")

    # Limit dataset size if max_samples is specified
    if max_samples is not None and max_samples < num_targets:
        print(f"Limiting dataset from {num_targets} to {max_samples} samples")
        field_data = field_data[:max_samples]
        targets = targets[:max_samples]
        num_targets = max_samples

    # split into training and test sets
    field_data_train = field_data[num_test:, :, :, :]  # (N_train, 2, S, R)
    targets_train = targets[num_test:, :, :]
    num_targets_train = num_targets - num_test

    field_data_test = field_data[0:num_test, :, :, :]  # (N_test, 2, S, R)
    targets_test = targets[0:num_test, :, :]

    # z-score normalize data pixel-wise across samples only (axis=0)
    data_mean = np.mean(field_data_train, axis=0, keepdims=True)  # Shape: (1, 2, S, R)
    data_std = np.std(field_data_train, axis=0, keepdims=True)    # Shape: (1, 2, S, R)
    
    # Normalize both train and test with training stats
    field_data_train = (field_data_train - data_mean) / (data_std + 1e-6)
    field_data_test = (field_data_test - data_mean) / (data_std + 1e-6)
    
    # Save normalization stats
    os.makedirs("./data/3d", exist_ok=True)
    np.save("./data/3d/2000_mean_std.npy", {'mean': data_mean, 'std': data_std})
    
    print(f"Train field data shape: {field_data_train.shape}")  # (N_train, 2, 72, 72)
    print(f"Test field data shape: {field_data_test.shape}")    # (N_test, 2, 72, 72)
    
    # Create Dataset objects
    train_data = Dataset(field_data_train, targets_train, num_targets_train, num_freqs, num_recvers, num_sources)
    test_data = Dataset(field_data_test, targets_test, num_test, num_freqs, num_recvers, num_sources)
    
    return train_data, test_data

