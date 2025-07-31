import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure
import warnings

warnings.filterwarnings("ignore")


def get_correlation_matrices(time_series):
    """Calculate Pearson correlation for the given time series."""
    return np.corrcoef(time_series)


def read_singal_fmri_ts(data_file):
    """Read fMRI time series data from a CSV file."""
    try:
        csv_data = pd.read_csv(data_file, header=None, encoding='latin1')  # Use a lenient encoding
        data = np.array(csv_data)

        # Check if splitting is needed
        if isinstance(data[0, 0], str):  # If the first cell is a string
            roi_ts = []
            for d in data:
                ts = d[0].split(' ')  # Split the string
                roi_ts.append(ts)
            return np.array(roi_ts).astype('float64')
        else:
            # Data is already numeric, no splitting required
            return data.astype('float64')

    except Exception as e:
        print(f"Error reading file {data_file}: {e}")
        return None


def get_fmri_data_by_atlas(file_dir):
    """Classify data by atlas_name and save results."""
    atlas_data = {}  # Dictionary to group data by atlas

    name_list = os.listdir(file_dir)
    for name in tqdm(name_list):
        dir_path = os.path.join(file_dir, name)
        if name == ".DS_Store" or name.startswith("._"):
            continue

        files = os.listdir(dir_path)
        for file_name in files:
            # Extract metadata from file name
            id_indiv = file_name.split('sub-')[-1].split('_task')[0]
            atlas_name = file_name.split('_')[-1].split('.')[0]  # Extract atlas name
            csv_path = os.path.join(dir_path, file_name)

            # Read time series and calculate connectivity matrix
            time_series = read_singal_fmri_ts(csv_path)
            if time_series is None:
                continue

            time_series = time_series.T  # Transpose to match the expected shape for correlation

            try:
                conn = get_correlation_matrices(time_series)
                if np.any(np.isnan(conn)) or np.any(np.isinf(conn)):
                    print(f"Skipping {id_indiv} due to NaN or Inf values in connectivity matrix.")
                    continue

                # Add data to the corresponding atlas group
                if atlas_name not in atlas_data:
                    atlas_data[atlas_name] = []
                atlas_data[atlas_name].append(conn)

            except Exception as e:
                print(f"Error processing {id_indiv}: {e}")
                continue

    # Ensure all matrices in each atlas group have the same shape
    for atlas_name, matrices in atlas_data.items():
        # Check for shape consistency
        shapes = [mat.shape for mat in matrices]
        print(f"Shapes for {atlas_name}: {shapes}")

        # Assuming all matrices are square, check for shape consistency
        target_shape = shapes[0] if shapes else None
        if target_shape is not None:
            matrices = [mat for mat in matrices if mat.shape == target_shape]

        if matrices:
            conn_array = np.array(matrices).astype(np.float32)  # Convert list to array
            save_path = f'/home/xinxu/Lehigh/Codes/BICLab_data/data/conn_abide2_{atlas_name}.npy'
            np.save(save_path, conn_array)
            print(conn_array.shape)
            print(f"Saved {atlas_name} connectivity matrices to: {save_path}")
        else:
            print(f"No valid matrices for {atlas_name}.")

    print("All data has been successfully processed and saved.")


if __name__ == '__main__':
    file_dir = '/home/xinxu/Lehigh/Codes/BICLab_data/ABIDE/ABIDE_II_ROISignals'
    get_fmri_data_by_atlas(file_dir)
