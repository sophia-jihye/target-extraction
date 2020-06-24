import os, time, re
from config import parameters
import pandas as pd
from Rules import Rules
from tqdm import tqdm
tqdm.pandas()

data_filepath = parameters.data_filepath
output_dir = parameters.output_dir
output_time_txt_filepath = parameters.output_time_txt_filepath

convert_carefully = {'mp':'mp3'}
def extract_targets_from_document(rid, rules, document, extracted_targets):
    doc = rules.nlp(document)
    for sentence in doc.sentences:
        if rid == 'R11':
            extracted_targets.update(rules.extract_targets_R11(*rules.parse_sentence(sentence)))
        elif rid == 'R12':
            extracted_targets.update(rules.extract_targets_R12(*rules.parse_sentence(sentence)))
    
    if len(extracted_targets) == 0: return '-'
    return [convert_carefully[item] if item in convert_carefully.keys() else item for item in list(extracted_targets)]
    
def apply_rule(rid, rules, df):
    df['extracted_targets'] = df.progress_apply(lambda x: extract_targets_from_document(rid, rules, x['content'], x['extracted_targets']), axis=1)
    df['%s_applied'%rid] = df.progress_apply(lambda x: 1 if x['extracted_targets']!='-' else 0, axis=1)
    
    num_ = len(df[df['%s_applied'%rid]==1])
    print('Number of sentences which %s applies: %d [%.2f]' % (rid, num_, num_/len(df)))
    return num_

number_pattern = re.compile('\d+')
def calculate_accuracy(df):
    # To complement the defect that the stanfordnlp dependency parsing does not identify "mp3" but identify "mp" only.
    correct_targets_dup = list([number_pattern.sub(' ', item).strip() for sublist in df['target'].values for item in sublist if item != ''])
    predicted_targets_dup = list([number_pattern.sub(' ', item).strip() for sublist in df['extracted_targets'].values for item in sublist if item != ''])
    cnt = 0
    for item in correct_targets_dup:
        try: 
            predicted_targets_dup.remove(item)
            cnt += 1
        except: pass
    dup_acc = cnt / len(correct_targets_dup)
    print('Average accuracy (duplicated targets): %.2f' % dup_acc)
    
    correct_targets_unique = set(correct_targets_dup)
    predicted_targets_unique = set(predicted_targets_dup)
    uniq_acc = len([item for item in predicted_targets_unique if item in correct_targets_unique]) / len(correct_targets_unique)
    print('Average accuracy (unique targets): %.2f' % uniq_acc)
    return dup_acc, uniq_acc
    
def main():
    rules = Rules()
    
    # Check each rule of Type1 rules
    for rid in ['R11', 'R12']:
        df = pd.read_json(data_filepath)
        df['extracted_targets'] = df.progress_apply(lambda x: set(), axis=1)

        print('Apply %s..' % rid)
        num_ = apply_rule(rid, rules, df)
        
        dup_acc, uniq_acc = calculate_accuracy(df)
        
        # Save
        filepath = os.path.join(output_dir, '%s_Cov%d_DupAcc%.2f_UniqAcc%.2f.csv' % (rid, num_, dup_acc, uniq_acc))
        df.to_csv(filepath, index = False, encoding='utf-8-sig')
        print('Created %s' % filepath)

    
    
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