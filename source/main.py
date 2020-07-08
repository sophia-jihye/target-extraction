import time, copy, os, pickle, glob, csv, ast
from config import parameters
from PatternHandler import PatternHandler
from DependencyGraphHandler import DependencyGraphHandler
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
tqdm.pandas()
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=parameters.num_cpus, progress_bar=True)

data_filepath = parameters.data_filepath
lexicon_filepath = parameters.lexicon_filepath
output_time_txt_filepath = parameters.output_time_txt_filepath
output_pattern_csv_filepath = parameters.output_pattern_csv_filepath
output_error_csv_filepath = parameters.output_error_csv_filepath
output_target_log_csv_filepath = parameters.output_target_log_csv_filepath
output_raw_df_pkl_filepath = parameters.output_raw_df_pkl_filepath
output_pattern_counter_pkl_filepath = parameters.output_pattern_counter_pkl_filepath
output_targets_dir = parameters.output_targets_dir
output_targets_concat_csv_filepath = parameters.output_targets_concat_csv_filepath
output_pattern_evaluation_csv_filepath = parameters.output_pattern_evaluation_csv_filepath

def match_opinion_words(content, opinion_word_lexicon):
    opinion_words = []
    for opinion in opinion_word_lexicon:
        for token in content.split():
            if token == opinion: opinion_words.append(token)
    return list(set(opinion_words))

def save_extracted_pattern_results(domain, pattern_counter, err_list):
    pattern_list = [tup for tup in pattern_counter.items()]
    pattern_df = pd.DataFrame(pattern_list, columns =['pattern', 'count'])  
    filepath = output_pattern_csv_filepath % (domain, len(pattern_df))
    pattern_df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)
    
    err_df = pd.DataFrame(err_list, columns =['content', 'current_opinion_word', 'current_target_word', 'parse_error', 'opinion_words', 'targets', 'raw_targets'])  
    filepath = output_error_csv_filepath % (domain, len(err_df[err_df['parse_error']==True]), len(err_df))
    err_df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)

def pattern_extraction(domain, df, pattern_handler, dependency_handler):
    pattern_counter, err_list = defaultdict(int), list()
    pattern_handler.extract_patterns(df, pattern_counter, err_list, dependency_handler)
    
    save_extracted_pattern_results(domain, pattern_counter, err_list)
    return pattern_counter

def merge_dfs(data_filepaths):
    dfs = []
    for data_filepath in data_filepaths:
        df = pd.read_csv(data_filepath)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def calculate_precision_recall(df):
    correct_targets_mul = list([item for sublist in df['targets'].values for item in sublist if item != ''])
    predicted_targets_mul = list([item for sublist in df['predicted_targets'].values for item in sublist if item != ''])
    
    tp_mul = 0
    for item in predicted_targets_mul:
        try: 
            correct_targets_mul.remove(item)
            tp_mul += 1
        except: pass
    
    if len(predicted_targets_mul) != 0: pre_mul = tp_mul / len(predicted_targets_mul)
    else: pre_mul = 0
    
    if len(correct_targets_mul) != 0: rec_mul = tp_mul / len(correct_targets_mul)
    else: rec_mul = 0
    
    correct_targets_dis = set([item for sublist in df['targets'].values for item in sublist if item != ''])
    predicted_targets_dis = set([item for sublist in df['predicted_targets'].values for item in sublist if item != ''])
    
    tp_dis = len([item for item in predicted_targets_dis if item in correct_targets_dis])
    if len(predicted_targets_dis) != 0: pre_dis = tp_dis / len(predicted_targets_dis)
    else: pre_dis = 0
    
    if len(correct_targets_dis) != 0: rec_dis = tp_dis / len(correct_targets_dis)
    else: rec_dis = 0
    
    return pre_mul, rec_mul, pre_dis, rec_dis

def calculate_f1(precision, recall):
    denominator = precision + recall
    if denominator == 0: return 0
    return (2*precision*recall)/denominator

def pattern_quality_estimation(domain, original_df, pattern_counter, pattern_handler, dependency_handler):
    idx = 0
    for one_flattened_dep_rels, pattern_count in sorted(pattern_counter.items(), key=lambda x: x[-1], reverse=True):
        idx += 1
        filepath = output_target_log_csv_filepath % (domain, pattern_count, one_flattened_dep_rels)
        if os.path.exists(filepath): continue
        print('[%d/%d]Extracting targets by pattern %s' % (idx, len(pattern_counter.keys()), one_flattened_dep_rels))
        dep_rels = one_flattened_dep_rels.split('-')
        df = copy.deepcopy(original_df)
        df['predicted_targets'] = df.parallel_apply(lambda x: pattern_handler.extract_targets(x['doc'], x['opinion_words'], dep_rels, dependency_handler), axis=1)
        df['pattern'] = one_flattened_dep_rels
        df['pattern_count'] = pattern_count

        df.drop(['filename', 'doc'], axis=1, inplace=True)
        df.to_csv(filepath, index = False, encoding='utf-8-sig')
        print('Created %s' % filepath)
        
    print('Merging csv files in %s for [%s]..' % (output_targets_dir, domain))
    concat_df = merge_dfs(glob.glob(os.path.join(output_targets_dir, '*.csv')))
    filepath = output_targets_concat_csv_filepath % domain
    concat_df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)

    concat_df['targets'] = concat_df.apply(lambda x: ast.literal_eval(x['targets']), axis=1)
    concat_df['predicted_targets'] = concat_df.apply(lambda x: ast.literal_eval(x['predicted_targets']), axis=1)
    
    print('Evaluating rules for [%s]..' % domain)
    filepath = output_pattern_evaluation_csv_filepath % domain
    f = open(filepath, 'w', encoding='utf-8-sig')
    wr = csv.writer(f)  
    wr.writerow(['domain', 'pattern', 'count', 'precision_multiple', 'recall_multiple', 'f1_multiple', 'precision_distinct', 'recall_distinct', 'f1_distinct'])
    for pattern in concat_df['pattern'].unique():
        current_df = concat_df[concat_df['pattern']==pattern]
        pre_mul, rec_mul, pre_dis, rec_dis = calculate_precision_recall(current_df)
        wr.writerow([domain, pattern, '%d'%current_df['pattern_count'].iloc[0], '%.2f'%pre_mul, '%.2f'%rec_mul, '%.2f'%calculate_f1(pre_mul,rec_mul), '%.2f'%pre_dis, '%.2f'%rec_dis, '%.2f'%calculate_f1(pre_dis,rec_dis)])
    f.close()
    print('Created %s' % filepath)

def save_pkl(item_to_save, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(item_to_save, f)
    print('Created %s' % filepath)

def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        loaded_item = pickle.load(f)
    print('Loaded %s' % filepath)
    return loaded_item

def main():
    pattern_handler = PatternHandler()
    dependency_handler = DependencyGraphHandler()
    
    if os.path.exists(output_raw_df_pkl_filepath): raw_df = load_pkl(output_raw_df_pkl_filepath)
    else:
        raw_df = pd.read_json(data_filepath)
        print('Matching opinion words..')
        opinion_word_lexicon = [item for sublist in pd.read_json(lexicon_filepath).values for item in sublist]
        raw_df['opinion_words'] = raw_df.parallel_apply(lambda x: match_opinion_words(x['content'], opinion_word_lexicon), axis=1)
        print('Converting document into nlp(doc)..')
        raw_df['doc'] = raw_df.progress_apply(lambda x: pattern_handler.nlp(x['content']), axis=1)
        raw_df['targets'] = raw_df.apply(lambda x: pattern_handler.process_targets(x['content'], x['raw_targets']), axis=1) 
        save_pkl(raw_df, output_raw_df_pkl_filepath)
    
    for domain in raw_df['domain'].unique():
        print('Processing [%s]..' % domain)
        df = raw_df[raw_df['domain']==domain]
        
        filepath = output_pattern_counter_pkl_filepath % domain
        if os.path.exists(filepath): pattern_counter = load_pkl(filepath)
        else: 
            pattern_counter = pattern_extraction(domain, df, pattern_handler, dependency_handler)
            save_pkl(pattern_counter, filepath)
        
        pattern_quality_estimation(domain, df, pattern_counter, pattern_handler, dependency_handler)
    
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