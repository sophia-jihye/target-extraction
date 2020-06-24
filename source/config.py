import os, json
from datetime import datetime

# system configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_output_dir = os.path.join(base_dir, 'output')
#output_dir = os.path.join(base_dir, 'output', datetime.now().strftime("%Y%m%d-%H-%M-%S"))
output_dir = os.path.join(base_dir, 'output', 'preliminary')

parameters_json_filepath = os.path.join(output_dir, 'parameters.json')

# create dirs
dirs = [base_output_dir, output_dir]
for directory in dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)  
        
class Parameters:
    def __init__(self):
        self.base_dir = base_dir
        self.data_filepath = os.path.join(base_dir, 'data', 'parsed', 'five-three_5995.json')
        self.lexicon_filepath = os.path.join(base_dir, 'data', 'parsed', 'lexicon_6788.json')
        self.output_dir = output_dir
        self.parameters_json_filepath = parameters_json_filepath
        self.output_time_txt_filepath = os.path.join(output_dir, 'elapsed_time.txt')
        
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