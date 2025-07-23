
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