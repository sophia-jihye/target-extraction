import os, time
from config import parameters
import numpy as np
import pandas as pd
from Rules import Rules
from tqdm import tqdm
tqdm.pandas()

data_filepath = parameters.data_filepath
output_dir = parameters.output_dir
output_time_txt_filepath = parameters.output_time_txt_filepath

def accuracy(extracted_targets, targets):
    if extracted_targets!='-' and len(targets)!=0:
        return len([item for item in extracted_targets if item in targets]) / len(targets) 
    else:
        return '-'

def apply_R11(rules, df):
    df['extracted_targets'] = df.progress_apply(lambda x: rules.extract_targets_R11(*rules.parse_document(x['content'])), axis=1)
    df['R11_applied'] = df.progress_apply(lambda x: 1 if x['extracted_targets']!='-' else 0, axis=1)
    
    df['accuracy'] = df.progress_apply(lambda x: accuracy( x['extracted_targets'],  x['target']), axis=1)
    num_ = len(df[df['R11_applied']==1])
    acc_ = np.mean(df[df['accuracy']!='-']['accuracy'])
    print('Number of sentences which R11 applies: %d [%.2f]' % (num_, num_/len(df)))
    print('Average accuracy: %.2f' % acc_)

    # Save
    filepath = os.path.join(output_dir, 'R11_Cov%d_Acc%.2f.csv' % (num_, acc_))
    df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)

def main():
    rules = Rules()
    df = pd.read_json(data_filepath)
    
    print('Apply R11..')
    apply_R11(rules, df)
    
    
if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    elapsed_time = end - start
    elapsed_time_txt = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    text_file = open(output_time_txt_filepath, "w", encoding='utf-8')
    content = 'Start: %s, End: %s => Elapsed time: %s\nCreated %s' % (time.strftime("%H:%M:%S", time.gmtime(start)), time.strftime("%H:%M:%S", time.gmtime(end)), elapsed_time_txt, output_time_txt_filepath)
    text_file.write(content)
    text_file.close()
    print(content)