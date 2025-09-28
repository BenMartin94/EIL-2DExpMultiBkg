import numpy as np
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
#    returns field_data as (num_targets, num_recvers, 2*num_sources, num_freqs), target data as (num_targets, 28, 28)
def process_multifreq_data(fields_file, targets_file, targetless_fields_file, num_test, trainset=True):
    print("Processing data...")
    # load field data
    with h5py.File(fields_file, 'r') as field_data:
        # print(list(field_data.keys()))
        fields = np.array(field_data['fields'])     # (num_targets, num_freqs, num_sources, num_recvers)  
        freqs = np.array(field_data['freqs'])
        num_freqs = np.size(freqs)
        num_targets = np.array(field_data['num_targets'])
        num_recvers = len(fields[0, 0, 0, :])
        num_sources = len(fields[0, 0, :, 0])
        print("recvers: ", num_recvers)
        print("sources: ", num_sources)


    fields = np.transpose(fields, [0, 2, 3, 1]) # freqs last

    # seperate complex number data for field data
    field_data = np.zeros((num_targets, num_sources, num_recvers, 2*num_freqs))
    for i in range(0, num_targets):
        for j in range(0, num_sources):
            for k in range(0, num_recvers):
                for l in range(0, num_freqs):
                    field = fields[i, j, k, l]
                    field_data[i, j, k, 2*l] = field[0]
                    field_data[i, j, k, 2*l+1] = field[1]

    # load targetless field data
    with h5py.File(targetless_fields_file, 'r') as targetless_field_data:
        # print(list(field_data.keys()))
        targetless_fields = np.array(targetless_field_data['fields'])     # (num_targets, num_freqs, num_recvers, num_sources) 
        targetless_fields = np.transpose(targetless_fields, [0, 2, 3, 1])

    # seperate complex number data for targetless field data
    targetless_field_data = np.zeros((num_targets, num_sources, num_recvers, 2*num_freqs))
    for j in range(0, num_sources):
        for k in range(0, num_recvers):
            for l in range(0, num_freqs):
                field = targetless_fields[0, j, k, l]
                targetless_field_data[0, j, k, 2*l] = field[0]
                targetless_field_data[0, j, k, 2*l+1] = field[1]

    # convert total fields into scattered fields by subtracting fields with no targets
    for target in range(0, num_targets):
        field_data[target, :, :, :] = field_data[target, :, :, :] - targetless_field_data[0, :, :, :]

    # convert to single frequency data (1GHz) (take both real and im channels)
    field_data = field_data[:, :, :, 14:16]
    num_freqs = 1

    # load targets
    ### for providing physical targets (not just target info)
    # with h5py.File(targets_file, 'r') as target_data:
    #     targets = np.array(target_data['targets'])      # (num_targets, 112, 112)
    
    targets = create_3d_mnist_data(targets_file)

    # split into training and test sets
    field_data_train = field_data[num_test:, :, :, :]
    targets_train = targets[num_test:, :, :]
    num_targets = num_targets - num_test

    field_data_test = field_data[0:num_test, :, :, :]
    targets_test = targets[0:num_test, :, :]

    # z-score normalize data (on trainset data only)
    if (trainset):
        data_mean = np.mean(field_data_train)
        data_std = np.std(field_data_train)
        field_data_train = (field_data_train - data_mean) / (data_std+1e-6)
        train_data = Dataset(field_data_train, targets_train, num_targets, num_freqs, num_recvers, num_sources)

        return train_data
    
    else: # testset
        stats = np.load("./Data/3d/2000_mean_std.npy")
        data_mean = stats[0]
        data_std = stats[1]
        field_data_test = (field_data_test - data_mean) / data_std
        test_data = Dataset(field_data_test, targets_test, num_test, num_freqs, num_recvers, num_sources)

        return test_data

    # normalize to [0,1] (on trainset data only)
    # if (trainset):
    #     fields_max = np.max(field_data_train)
    #     fields_min = np.min(field_data_train)
    #     field_data_train = (field_data_train - fields_min) / (fields_max - fields_min)
    #     train_data = Dataset(field_data_train, targets_train, num_targets, num_freqs, num_recvers, num_sources)

    #     np.save("./Data/3d/2000_max_min", np.array([fields_max, fields_min]))
    #     return train_data
    
    # else: # testset
    #     stats = np.load("./Data/3d/2000_max_min.npy")
    #     fields_max = stats[0]
    #     fields_min = stats[1]
    #     field_data_test = (field_data_test - fields_min) / (fields_max - fields_min)
    #     test_data = Dataset(field_data_test, targets_test, num_test, num_freqs, num_recvers, num_sources)

    #     return test_data

