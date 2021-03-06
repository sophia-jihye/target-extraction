import os, json
from datetime import datetime
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_pick_count', type=int, default=0)
parser.add_argument('--config_option', type=str, default='') # 'f1mi', 'premi'
parser.add_argument('--min_pattern_count', type=int, default=0)
parser.add_argument('--min_pattern_f1', type=float, default=0.0)

parser.add_argument('--allow_f1_decrease_count', type=int, default=0)
args = parser.parse_args()

domains = ['Digital camera']   
pattern_types = ['ot']   # , 'tt'
#domains = ['Digital camera1', 'Digital camera2', 'Cell phone', 'MP3 player', 'DVD player', 'Computer', 'Wireless router', 'Speaker']

# system configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_output_dir = os.path.join(base_dir, 'output')
output_dir = os.path.join(base_dir, 'output', '20200805-13-24-52')   # datetime.now().strftime("%Y%m%d-%H-%M-%S")
output_err_dir = os.path.join(output_dir, 'err')
output_training_dir = os.path.join(output_dir, 'training')
output_test_dir = os.path.join(output_dir, 'test_%s_%s_pick%d_mpc=%d_f=%.1f'% ('-'.join(domains), args.config_option, args.max_pick_count, args.min_pattern_count, args.min_pattern_f1))
output_save_dir = os.path.join(output_dir, 'save')
output_targets_dir = os.path.join(output_dir, 'targets')

parameters_json_filepath = os.path.join(output_test_dir, 'parameters.json')

# create dirs
dirs = [output_dir, output_err_dir, output_training_dir, output_test_dir, output_save_dir, output_targets_dir]
for directory in dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

class Parameters:
    def __init__(self):
        self.allow_f1_decrease_count = args.allow_f1_decrease_count
        self.config_option = args.config_option
        self.min_pattern_count = args.min_pattern_count
        self.max_pick_count = args.max_pick_count
        self.min_pattern_f1 = args.min_pattern_f1
        self.domains = domains
        self.pattern_types = pattern_types
        self.base_dir = base_dir
        self.data_filepath = os.path.join(base_dir, 'data', 'parsed', 'five_3903.json') 
        self.lexicon_filepath = os.path.join(base_dir, 'data', 'parsed', 'lexicon_6788.json')
        self.output_dir = output_dir
        self.output_targets_dir = output_targets_dir
        self.output_test_dir = output_test_dir
        self.parameters_json_filepath = parameters_json_filepath
        self.output_time_txt_filepath = os.path.join(output_test_dir, 'elapsed_time.txt')
        self.errlog_filepath = os.path.join(output_err_dir, '%s.log' % (datetime.now().strftime("%Y%m%d-%H-%M-%S")))
        self.output_raw_df_pkl_filepath = os.path.join(output_save_dir, '[%s]raw_df.pkl')
        self.output_training_test_dfs_pkl_filepath = os.path.join(output_save_dir, '[%s]training_test_dfs.pkl')
        self.output_pattern_quality_estimation_csv_filepath = os.path.join(output_training_dir, '[%s][k=%d]pattern_quality_estimation.csv')
        self.output_target_log_csv_filepath = os.path.join(output_targets_dir, '%s_k=%d_%s_%02d_%s.csv')
        self.output_pattern_counter_pkl_filepath = os.path.join(output_save_dir, '[%s][k=%d][%s]pattern_counter.pkl')
        self.output_pattern_csv_filepath = os.path.join(output_training_dir, '[%s][k=%d][%s]patterns_%d.csv')
        self.output_error_csv_filepath = os.path.join(output_training_dir, '[%s][k=%d][%s]error_%d_%d.csv')
        self.output_targets_concat_csv_filepath = os.path.join(output_save_dir, '[%s][k=%d][%s]targets.csv')
        self.output_subset_selection_log_filepath = os.path.join(output_training_dir, '[%s][k=%d][%s]subset_selection.log')
        self.output_subset_pkl_filepath = os.path.join(output_save_dir, '[%s][k=%d][%s]subset.pkl')
        self.output_target_extraction_report_csv_filepath = os.path.join(output_test_dir, '[%s][k=%d]target_extraction_report.csv')
        self.output_final_report_csv_filepath = os.path.join(output_test_dir, 'final_report.csv')
        self.num_cpus = 8

    def __str__(self):
        item_strf = ['{} = {}'.format(attribute, value) for attribute, value in self.__dict__.items()]
        strf = 'Parameters(\n  {}\n)'.format('\n  '.join(item_strf))
        return strf

    def save(self):
        with open(self.parameters_json_filepath, 'w+') as js:
            json.dump(self.__dict__, js)
        print('Created file:', self.parameters_json_filepath)

parameters = Parameters()
parameters.save()
print(parameters)