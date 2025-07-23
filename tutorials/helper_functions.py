import numpy as np

def detect_trial_type(behav_data, task='reversal_learning'):
    """Determine trial type from behavioral data."""
    behav = behav_data
    behav['trial_type'] = []
    for i in range(len(behav['action'])):
        if task == 'reversal_learning':
            behav['trial_type'].append(behav['stage2'][i] * 2 + behav['reward'][i])
        else:
            behav['trial_type'].append(behav['action'][i] * 4 + behav['stage2'][i] * 2 + behav['reward'][i])

    return behav



def detect_trial_type_ibl(behav_data):
    """Determine trial type from behavioral data."""
    behav = behav_data
    behav['trial_type'] = []
    for i in range(len(behav['Choice'])):
        behav['trial_type'].append(behav['Choice_prev'][i] * 4 + (np.sign(behav['Reward_prev'][i]) > 0).astype(int) * 2 + (np.sign(behav['Stimulus'][i]) > 0).astype(int))
    # TODO: this conversion doesnt care about 0 contrast stimuli it groups them with something else
    return behav
