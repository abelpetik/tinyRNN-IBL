import sys

from matplotlib.pyplot import legend

sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *
from analyzing_experiments.analyzing_perf import run_scores_exp
from plotting_experiments.plotting_dynamics import plot_all_models_value_change
from analyzing_experiments.analyzing_perf import find_best_models_for_exp

behavior_data_path = 'C:\\Data\\tinyRNN\\ibl_behavior_data_converted.pkl'

# Creating a Dataset object with the behavior data loaded from the file
behav_data_spec = {
    'data': behavior_data_path,
    'input_format': [
        dict(name='Choice_prev'),
        dict(name='Reward_prev'),
        dict(name='Stimulus'),
    ],
    'output_dim': 2, # number of output dimensions
    'target_name': 'Choice',
}

dd = Dataset('Simple', behav_data_spec=behav_data_spec)
dd.behav_to(dict(behav_format='tensor', output_h0=True))

base_config = {
      ### dataset info
      'dataset': 'Simple',
      'behav_data_spec': behav_data_spec,
      'behav_format': 'tensor',

      ### model info
      'agent_type': 'RNN',
      'rnn_type': 'GRU', # which rnn layer to use
      'input_dim': 3,
      'hidden_dim': 1, # dimension of this rnn layer
      'output_dim': 2, # dimension of action
      'device': 'cpu',
      'output_h0': True, # whether initial hidden state included in loss
      'trainable_h0': False, # the agent's initial hidden state trainable or not
      'readout_FC': True, # whether the readout layer is full connected or not
      'one_hot': False, # whether the data input is one-hot or not
      ### training info for one model
      'lr':0.005,
      'l1_weight': 1e-5,
      'weight_decay': 0,
      'penalized_weight': 'rec',
      'max_epoch_num': 2000,
      'early_stop_counter': 200,
      ### training info for many models on dataset
      'outer_splits': 5,
      'inner_splits': 3,
      'seed_num': 2,
      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
}

config_ranges = { # keys are used to generate model names
      'rnn_type': ['GRU'],
      'hidden_dim': [1],
      'readout_FC': [True],
      'l1_weight': [1e-5],
}

if __name__ == '__main__':
    behavior_cv_training_config_combination(base_config, config_ranges)

    exp_folder = get_training_exp_folder_name(__file__)
    find_best_models_for_exp(
        exp_folder, 'MABCog',
        additional_rnn_keys={
            'model_identifier_keys': ['block', 'distill', 'pretrained', 'distill_temp', 'teacher_prop', ], },
        rnn_sort_keys=['block', 'hidden_dim'],
        has_rnn=True,
        has_cog=False,
        return_dim_est=True,
        include_acc=True,
        check_missing=False,
    )

    run_scores_exp(
        exp_folder, demask=False,
        pointwise_loss=False,
        model_filter={'distill': 'none', 'rnn_type': 'GRU'},
        overwrite_config={
            # 'behav_data_spec': {'augment': True},
            'device': 'cpu',
        },
        include_data='all',
        has_cog=False,
    )

    dynamics_plot_pipeline = [
        '2d_logit_change',  # logit vs logit change
        # '2d_logit_next', # a different representation of logit change
        # '2d_pr_nextpr', # a different representation of probability change
        # '2d_pr_change', # ~same as logit change but worse
    ]
    plot_all_models_value_change(exp_folder, plots=dynamics_plot_pipeline, save_pdf=True, plot_max_logit=3)
