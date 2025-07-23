import pickle
import numpy as np

dataset_path = 'C:\\Data\\tinyRNN\\ibl_behavior_data.pkl'

if __name__ == '__main__':
    ibl_data = pickle.load(open(dataset_path, 'rb'))

    n_blocks=749
    n_trials=25

    # Go through all the keys in ibl_data and turn the list of values into a list of 1D numpy arrays.
    # The arrays should have n_trials elements and the length of the list should be n_blocks.
    for key in ibl_data.keys():
        new_list = []
        for i in range(n_blocks):
            new_list.append(np.array(ibl_data[key][i * n_trials:(i + 1) * n_trials]))
        ibl_data[key] = new_list

    from helper_functions import detect_trial_type_ibl
    ibl_data = detect_trial_type_ibl(ibl_data)

    for i in range(len(ibl_data['Choice_prev'])):
        ibl_data['Choice_prev'][i] = np.array(ibl_data['Choice_prev'][i], dtype=int)
        ibl_data['Reward_prev'][i] = np.array(ibl_data['Reward_prev'][i], dtype=int)
        ibl_data['Choice'][i] = np.array(ibl_data['Choice'][i], dtype=int)

    # Save the modified data back to the file
    new_dataset_path = 'C:\\Data\\tinyRNN\\ibl_behavior_data_converted.pkl'
    with open(new_dataset_path, 'wb') as f:
        pickle.dump(ibl_data, f)

    # np.unique(np.concatenate(data['trial_type']))
