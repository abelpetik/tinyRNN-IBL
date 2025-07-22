# Run in parent directory
import os
print(os.getcwd())
if os.getcwd().split('/')[-1] != 'tinyRNN':
    os.chdir('..')
    print(os.getcwd())
assert os.getcwd().split('\\')[-1] == 'tinyRNN'

import numpy as np

import sys
sys.path.append(os.getcwd() + '/simulating_experiments')
sys.path.append(os.getcwd() + '/plotting_experiments')


class MultiArmedBandit:
    '''2-armed Bernoulli bandit'''
    def __init__(self, n_actions=2):
        self.n_actions = n_actions

    def reset(self, n_trials=None):
        self.probs = np.random.uniform(0, 1, size=self.n_actions)

    def trial(self, choice):
        outcome = np.random.binomial(1, self.probs[choice])
        return (choice, outcome)


from agents.MABCogAgent import MABCogAgent

mf = MABCogAgent(dict(cog_type='MF', n_actions=2))
mf.model.params[mf.model.param_names.index('iTemp')] = 2.


from agents.RNNAgent import RNNAgent
mf = RNNAgent(dict(rnn_type='GRU', input_dim=2, hidden_dim=2, output_dim=2, device='cpu', seed=2))


print(mf.simulate(MultiArmedBandit(), 5, get_DVs=False))

from simulating_experiments.simulate_experiment import simulate_exp
config = dict(
    n_blocks=500, n_trials=20, sim_seed=42, sim_exp_name='test', additional_name='',
    task='BartoloMonkey',
)
_ = simulate_exp(mf, MultiArmedBandit(), config)


from  training_experiments.training import behavior_cv_training_config_combination

base_config = {
      ### dataset info
      'dataset': 'SimAgent',
      'behav_data_spec': {
        'agent_path': ['test'],
        'agent_name': 'MF_seed42',
      },

      ### model info
      'behav_format': 'tensor',
      'agent_type': 'RNN',
      'rnn_type': 'GRU', # which rnn layer to use
      'include_embedding': False,
      'input_dim': 3,
      'hidden_dim': 2, # dimension of this rnn layer
      'output_dim': 2, # dimension of action
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
      # 'single_inner_fold': True,
      'seed_num': 2,

      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': [], # can be a list of diagnose function strings

      ### current training exp path
      'exp_folder': 'sim_mf',
}

config_ranges = { # keys are used to generate model names
    #   'rnn_type': ['GRU'],
      'hidden_dim': [1,2],
    #   'l1_weight': [1e-5],
}

behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)

