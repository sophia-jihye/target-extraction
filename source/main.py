import os, time, re
from config import parameters
import pandas as pd
from Rules import Rules
from tqdm import tqdm
tqdm.pandas()

data_filepath = parameters.data_filepath
output_dir = parameters.output_dir
output_time_txt_filepath = parameters.output_time_txt_filepath

def extract_targets_from_document(rid, rules, document, extracted_targets):
    doc = rules.nlp(document)
    for sentence in doc.sentences:
        if rid == 'R11':
            extracted_targets.update(rules.extract_targets_R11(*rules.parse_sentence(sentence)))
        elif rid == 'R12':
            extracted_targets.update(rules.extract_targets_R12(*rules.parse_sentence(sentence)))
    return extracted_targets
    
def apply_rule(rid, rules, df):
    df['extracted_targets'] = df.progress_apply(lambda x: extract_targets_from_document(rid, rules, x['content'], x['extracted_targets']), axis=1)

def calculate_coverage(df):
    df['rules_applied'] = df.apply(lambda x: 1 if len(x['extracted_targets'])>0 else 0, axis=1)
    num_ = len(df[df['rules_applied']==1])
    coverage = num_/len(df)
    print('Coverage: %.2f [%d]' % (coverage, num_))
    return coverage
    
number_pattern = re.compile('\d+')
def calculate_accuracy(df):
    # To complement the defect that the stanfordnlp dependency parsing does not identify "mp3" but identify "mp" only.
    correct_targets_mul = list([number_pattern.sub(' ', item).strip() for sublist in df['target'].values for item in sublist if item != ''])
    predicted_targets_mul = list([number_pattern.sub(' ', item).strip() for sublist in df['extracted_targets'].values for item in sublist if item != ''])
    cnt = 0
    for item in correct_targets_mul:
        try: 
            predicted_targets_mul.remove(item)
            cnt += 1
        except: pass
    mul_acc = cnt / len(correct_targets_mul)
    print('Average accuracy (based on multiple occurences): %.2f' % mul_acc)
    
    correct_targets_dis = set([number_pattern.sub(' ', item).strip() for sublist in df['target'].values for item in sublist if item != ''])
    predicted_targets_dis = set([number_pattern.sub(' ', item).strip() for sublist in df['extracted_targets'].values for item in sublist if item != ''])
    dis_acc = len([item for item in predicted_targets_dis if item in correct_targets_dis]) / len(correct_targets_dis)
    print('Average accuracy (based on distinct occurences): %.2f' % dis_acc)
    return mul_acc, dis_acc

def check_result(title, df):
    coverage = calculate_coverage(df) 
    mul_acc, dis_acc = calculate_accuracy(df)

    # Save
    filepath = os.path.join(output_dir, '[%s]Cov%.2f_MulAcc%.2f_DisAcc%.2f.csv' % (title, coverage, mul_acc, dis_acc))
    df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)

def main():
    rules = Rules()
    
    # Check each rule of Type1 rules
    for rid_set in ['R11+R12', 'R12+R11']:
        raw_df = pd.read_json(data_filepath)
        
        for filename in raw_df['filename'].unique():
            df = raw_df[raw_df['filename']==filename]
        
            df['extracted_targets'] = df.apply(lambda x: set(), axis=1)
            for idx, rid in enumerate(rid_set.split('+')):
                print('[%s %s] Applying %s..' % (filename, rid_set, rid))
                apply_rule(rid, rules, df)

                if idx == 0: check_result('_'.join([rid_set, str(idx)]), df)   # Temp: rid_set이 2개의 rid를 가지고있을 때
            check_result(' '.join([filename, rid_set]), df)
    
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