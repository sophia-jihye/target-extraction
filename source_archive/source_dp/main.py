import os, time, re, csv
from config import parameters
import pandas as pd
from collections import namedtuple
from Rules import Rules
from tqdm import tqdm
tqdm.pandas()

data_filepath = parameters.data_filepath
output_dir = parameters.output_dir
output_sub_dir = parameters.output_sub_dir
output_time_txt_filepath = parameters.output_time_txt_filepath
output_report_csv_filepath = parameters.output_report_csv_filepath

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
def calculate_precision_recall(df):
    
    # number_pattern, strip()
    # To complement the defect that the stanfordnlp dependency parsing does not identify "mp3" but identify "mp" only.
    correct_targets_mul = list([number_pattern.sub(' ', item).strip() for sublist in df['target'].values for item in sublist if item != ''])
    predicted_targets_mul = list([number_pattern.sub(' ', item).strip() for sublist in df['extracted_targets'].values for item in sublist if item != ''])
    
    cnt = 0
    for item in correct_targets_mul:
        try: 
            predicted_targets_mul.remove(item)
            cnt += 1
        except: pass
    pre_mul = cnt / len(predicted_targets_mul)
    print('Precision (based on multiple occurences): %.2f' % pre_mul)
    
    cnt = 0
    for item in predicted_targets_mul:
        try: 
            correct_targets_mul.remove(item)
            cnt += 1
        except: pass
    rec_mul = cnt / len(correct_targets_mul)
    print('Recall (based on multiple occurences): %.2f' % rec_mul)
    
    # number_pattern, strip()
    # To complement the defect that the stanfordnlp dependency parsing does not identify "mp3" but identify "mp" only.
    correct_targets_dis = set([number_pattern.sub(' ', item).strip() for sublist in df['target'].values for item in sublist if item != ''])
    predicted_targets_dis = set([number_pattern.sub(' ', item).strip() for sublist in df['extracted_targets'].values for item in sublist if item != ''])
    
    pre_dis = len([item for item in predicted_targets_dis if item in correct_targets_dis]) / len(predicted_targets_dis)
    print('Precision (based on distinct occurences): %.2f' % pre_dis)
    
    rec_dis = len([item for item in correct_targets_dis if item in predicted_targets_dis]) / len(correct_targets_dis)
    print('Recall (based on distinct occurences): %.2f' % rec_dis)
    
    return pre_mul, rec_mul, pre_dis, rec_dis

def check_result(filename, rule_name, df):
    coverage = calculate_coverage(df) 
    pre_mul, rec_mul, pre_dis, rec_dis = calculate_precision_recall(df)

    f_mul = (2*pre_mul*rec_mul) / (pre_mul + rec_mul)
    f_dis = (2*pre_dis*rec_dis) / (pre_dis + rec_dis)
    
    # Save
    filepath = os.path.join(output_sub_dir, '[%s][%s]Cov%.2f_F1Mul%.2f_F1Dis%.2f.csv' % (filename, rule_name, coverage, f_mul, f_dis))
    df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)
    
    Val = namedtuple('Val', ['measure', 'measure_type', 'value'])
    return filename, rule_name, [Val('Coverage', 'Multiple', coverage), Val('Precision', 'Multiple', pre_mul), Val('Precision', 'Distinct', pre_dis), Val('Recall', 'Multiple', rec_mul), Val('Recall', 'Distinct', rec_dis), Val('F-measure', 'Multiple', f_mul), Val('F-measure', 'Distinct', f_dis)]

def create_report(rule_names, filenames, measures, measure_types, val_dict):
    rule_names, filenames, measures, measure_types = list(rule_names), list(filenames), list(measures), list(measure_types)
    rule_names.sort()
    filenames.sort()
    measures.sort()
    measure_types.sort(reverse=True)
    f = open(output_report_csv_filepath, 'w', encoding='utf-8-sig', newline='')
    wr = csv.writer(f)  
    columns = ['domain', 'measure', 'type']
    columns.extend(rule_names)
    wr.writerow(columns)
    for filename in filenames:
        for measure in measures:
            for measure_type in measure_types:
                vals = [filename, measure, measure_type]
                for rule_name in rule_names:
                    try: vals.append('%.4f' % val_dict[(rule_name, filename, measure, measure_type)])
                    except: pass
                wr.writerow(vals)
    f.close()
    print('Created %s' % output_report_csv_filepath)

def main():
    rules = Rules()
    
    # To create report.csv
    rule_names, filenames, measures, measure_types, val_dict = set(), set(), set(), set(), dict()
    
    # Check each rule of Type1 rules
    for rid_set in ['R11+R12', 'R12']:
        raw_df = pd.read_json(data_filepath)
        
        for filename in raw_df['filename'].unique():
            df = raw_df[raw_df['filename']==filename]
        
            df['extracted_targets'] = df.apply(lambda x: set(), axis=1)
            for idx, rid in enumerate(rid_set.split('+')):
                print('[%s %s] Applying %s..' % (filename, rid_set, rid))
                apply_rule(rid, rules, df)

                if len(rid_set.split('+')) > 1 and idx == 0: check_result(filename, '_'.join([rid_set, str(idx)]), df)   # Temp: rid_set이 2개의 rid를 가지고있을 때
            
            # To create report.csv
            filename, rule_name, val_tup_list = check_result(filename, rid_set, df)
            rule_names.add(rule_name)
            filenames.add(filename)
            for val_tup in val_tup_list:
                measures.add(val_tup.measure)
                measure_types.add(val_tup.measure_type)
                val_dict[(rule_name, filename, val_tup.measure, val_tup.measure_type)] = val_tup.value
            
    # report.csv
    create_report(rule_names, filenames, measures, measure_types, val_dict)

def elapsed_time(start):
    end = time.time()
    elapsed_time = end - start
    elapsed_time_txt = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    text_file = open(output_time_txt_filepath, "w", encoding='utf-8')
    content = 'Start: %s, End: %s => Elapsed time: %s\nCreated %s' % (time.strftime("%H:%M:%S", time.gmtime(start)), time.strftime("%H:%M:%S", time.gmtime(end)), elapsed_time_txt, output_time_txt_filepath)
    text_file.write(content)
    text_file.close()
    print('Created %s' % output_time_txt_filepath)
    
if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time(start)