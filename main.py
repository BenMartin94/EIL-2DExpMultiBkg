import scipy.io as sio
import numpy as np

def split_complex_to_real_imag(arr):
    """
    Converts an n-dimensional complex array to an (n+1)-dimensional array
    with the last dimension of size 2: [real, imag].
    """
    if not np.iscomplexobj(arr):
        raise ValueError("Input array must be of complex type.")
    return np.stack((arr.real, arr.imag), axis=-1)

def read_mat_file(file_path):
    """Reads a .mat file and returns the contained matrix."""
    mat_data = sio.loadmat(file_path)
    num_samples = mat_data['data'].shape[1]
    grid = mat_data['data'][0, 0][0]
    uncal_s_par = mat_data['data'][0, 0][1]
    cal_e_field = mat_data['data'][0, 0][2]
    synth_field = mat_data['data'][0, 0][3]
    grids = np.zeros((num_samples, *grid.shape), dtype=np.complex128)
    uncal_s_pars = np.zeros((num_samples, *uncal_s_par.shape), dtype=np.complex128)
    cal_e_fields = np.zeros((num_samples, *cal_e_field.shape), dtype=np.complex128)
    synth_fields = np.zeros((num_samples, *synth_field.shape), dtype=np.complex128)
    for i in range(num_samples):
        grids[i] = mat_data['data'][0, i][0]
        uncal_s_pars[i] = mat_data['data'][0, i][1]
        cal_e_fields[i] = mat_data['data'][0, i][2]
        synth_fields[i] = mat_data['data'][0, i][3]

    return grids, uncal_s_pars, cal_e_fields, synth_fields

if __name__ == "__main__":

    file_path = 'all_data.mat'  # Replace with your .mat file path
    grids, uncal_s_pars, cal_e_fields, synth_fields = read_mat_file(file_path)

    grids = split_complex_to_real_imag(grids)
    uncal_s_pars = split_complex_to_real_imag(uncal_s_pars)
    cal_e_fields = split_complex_to_real_imag(cal_e_fields)
    synth_fields = split_complex_to_real_imag(synth_fields)
    print("Grid shape:", grids.shape)
    print("Uncalibrated S-parameters shape:", uncal_s_pars.shape)
    print("Calibrated E-fields shape:", cal_e_fields.shape)
    print("Synthesized fields shape:", synth_fields.shape)