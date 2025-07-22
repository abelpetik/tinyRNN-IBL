import numpy as np
import joblib
# Run in parent directory
import os
if os.getcwd().split('/')[-1] != 'tinyRNN':
    os.chdir('..')
assert os.getcwd().split('\\')[-1] == 'tinyRNN'


class MultiArmedBandit:
    '''5-armed Bernoulli bandit'''
    def __init__(self, n_actions=5):
        self.n_actions = n_actions

    def reset(self, n_trials=None):
        self.probs = np.random.uniform(0, 1, size=self.n_actions)

    def trial(self, choice):
        outcome = np.random.binomial(1, self.probs[choice])
        return (choice, outcome)




from agents.MABCogAgent import MABCogAgent

mf = MABCogAgent(dict(cog_type='MF', n_actions=5))
mf.model.params[mf.model.param_names.index('iTemp')] = 2.


from simulating_experiments.simulate_experiment import simulate_exp
config = dict(
    n_blocks=500, n_trials=30, sim_seed=42, sim_exp_name='test', additional_name='',
    task='BartoloMonkey',
)
data = simulate_exp(mf, MultiArmedBandit(), config, save=False)

print(data['action'][0])
print(data['reward'][0])

# Save the data to a file
fn = 'C:\\Data\\tinyRNN\\custom_dataset_data.pkl'
joblib.dump({
    'action': data['action'],
    'reward': data['reward'],
}, fn)



x = np.zeros((3, 4, 5))

items = np.array([0, 2, 3], dtype=int)
x[0, items, 0] = 1
print(x)



from datasets.dataset_utils import Dataset
behav_data_spec = {
    'data': fn,
    'input_format': [
        dict(name='action', one_hot_classes=5),
        dict(name='reward'),
    ],
    'output_dim': 5, # number of output dimensions
    'target_name': 'action',
}
dd = Dataset('Simple', behav_data_spec=behav_data_spec)
dd.behav_to(dict(behav_format='tensor', output_h0=True))

print('actions are encoded properly')
print(data['action'][0][:10])
print(dd.torch_beahv_input[:, 0, :5][:10].max(1).indices)

print('rewards are encoded properly')
print(data['reward'][0][:10])
print(dd.torch_beahv_input[:, 0, 5][:10])


from training_experiments.training import behavior_cv_training_config_combination

base_config = {
      ### dataset info
      'dataset': 'Simple',
      'behav_data_spec': behav_data_spec,

      ### model info
      'behav_format': 'tensor',
      'agent_type': 'RNN',
      'rnn_type': 'GRU', # which rnn layer to use
      'include_embedding': False,
      'input_dim': 6,
      'hidden_dim': 2, # dimension of this rnn layer
      'output_dim': 6, # dimension of action
      'output_h0': True, # whether initial hidden state included in loss
      'trainable_h0': False, # the agent's initial hidden state trainable or not
      'readout_FC': True, # whether the readout layer is full connected or not
      'one_hot': False, # whether the data input is one-hot or not

      'device': 'cpu',
      ### training info for one model
      'lr': 0.005,
      'l1_weight': 1e-5,
      'weight_decay': 0,
      'penalized_weight': 'rec',
      'max_epoch_num': 10000,
      'early_stop_counter': 200,
      'batch_size': 0, # no mini-batch
      ### training info for many models on dataset
      'outer_splits': 3,
      'inner_splits': 2,
      'single_inner_fold': True,
      'seed_num': 1,

      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': [], # can be a list of diagnose function strings

      ### current training exp path
      'exp_folder': '5ab_mf',
}

config_ranges = { # keys are used to generate model names
  'hidden_dim': [2,5],
}

behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)




ll = np.mean(np.concatenate([
    row['trial_log_likelihood']
    for row in data['mid_vars']
]))
print('average log likelihood of generated data', ll)



exp_folder = '5ab_mf'
# from first tutorial ipynb
from analyzing_experiments.analyzing_perf import run_scores_exp

run_scores_exp(
    exp_folder, demask=False,
    pointwise_loss=False,
    model_filter={'distill': 'none', 'cog_type': 'MF'},
    overwrite_config={
        # 'behav_data_spec': {'augment': True},
        'device': 'cpu',
    },
    include_data='all',
    has_cog=False,
    has_rnn=False
)

from plotting_experiments.plotting_dynamics import plot_all_models_value_change

dynamics_plot_pipeline = [
    '2d_logit_change', # logit vs logit change
]
plot_all_models_value_change(exp_folder, plots=dynamics_plot_pipeline, save_pdf=True)




from analyzing_experiments.analyzing_perf import find_best_models_for_exp

exp_folder = '5ab_mf'
find_best_models_for_exp(
    exp_folder, 'MABCog',
    additional_rnn_keys={'model_identifier_keys': ['block','distill','pretrained', 'distill_temp','teacher_prop',],},
    rnn_sort_keys=['block', 'hidden_dim'],
    has_rnn=True,
    has_cog=False,
    return_dim_est=True,
    include_acc=True,
    check_missing=False,
)

