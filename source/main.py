from utils import *
import time, copy, os, glob, csv, ast
import pandas as pd
import numpy as np
from collections import defaultdict
from config import parameters
from PatternHandler import PatternHandler
from DependencyGraphHandler import DependencyGraphHandler
from SubsetHandler import SubsetHandler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
from tqdm import tqdm
tqdm.pandas()
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=parameters.num_cpus, progress_bar=True)

domains = parameters.domains
min_pattern_count = parameters.min_pattern_count
min_pattern_f1 = parameters.min_pattern_f1

data_filepath = parameters.data_filepath
lexicon_filepath = parameters.lexicon_filepath
output_time_txt_filepath = parameters.output_time_txt_filepath
output_pattern_csv_filepath = parameters.output_pattern_csv_filepath
output_error_csv_filepath = parameters.output_error_csv_filepath
output_target_log_csv_filepath = parameters.output_target_log_csv_filepath
output_raw_df_pkl_filepath_ = parameters.output_raw_df_pkl_filepath
output_training_test_dfs_pkl_filepath = parameters.output_training_test_dfs_pkl_filepath
output_pattern_counter_pkl_filepath = parameters.output_pattern_counter_pkl_filepath
output_targets_dir = parameters.output_targets_dir
output_targets_concat_csv_filepath = parameters.output_targets_concat_csv_filepath
output_pattern_quality_estimation_csv_filepath = parameters.output_pattern_quality_estimation_csv_filepath
output_subset_selection_log_filepath = parameters.output_subset_selection_log_filepath
output_subset_pkl_filepath = parameters.output_subset_pkl_filepath
output_target_extraction_report_csv_filepath = parameters.output_target_extraction_report_csv_filepath
output_final_report_csv_filepath = parameters.output_final_report_csv_filepath

def match_opinion_words(content, opinion_word_lexicon):
    opinion_words = []
    for opinion in opinion_word_lexicon:
        for token in content.split():
            if token == opinion: opinion_words.append(token)
    return list(set(opinion_words))

def save_extracted_pattern_results(domain, k, pattern_type, pattern_counter, err_list):
    pattern_list = [tup for tup in pattern_counter.items()]
    pattern_df = pd.DataFrame(pattern_list, columns =['pattern', 'count'])  
    filepath = output_pattern_csv_filepath % (domain, k, pattern_type, len(pattern_df))
    pattern_df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)
    
    err_df = pd.DataFrame(err_list, columns =['content', 'current_opinion_word', 'current_target_word', 'parse_error', 'opinion_words', 'targets', 'raw_targets'])  
    filepath = output_error_csv_filepath % (domain, k, pattern_type, len(err_df[err_df['parse_error']==True]), len(err_df))
    err_df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)

def pattern_extraction(domain, k, pattern_type, df, pattern_handler, dependency_handler):
    print('[%s k=%d %s]' % (domain, k, pattern_type))
    pattern_counter, err_list = defaultdict(int), list()
    if pattern_type == 'ot': pattern_handler.extract_patterns_ot(df, pattern_counter, err_list, dependency_handler)
    elif pattern_type == 'tt': pattern_handler.extract_patterns_tt(df, pattern_counter, err_list, dependency_handler)
    
    save_extracted_pattern_results(domain, k, pattern_type, pattern_counter, err_list)
    return pattern_counter

def merge_dfs(data_filepaths):
    dfs = []
    for data_filepath in data_filepaths:
        df = pd.read_csv(data_filepath)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def pattern_quality_estimation(domain, k, pattern_type, original_df, pattern_counter, pattern_handler, dependency_handler):
    print('Processing pattern_quality_estimation for [%s] (# of patterns = %d)..' % (domain, len(pattern_counter)))
    idx = 0
    for one_flattened_dep_rels, pattern_count in sorted(pattern_counter.items(), key=lambda x: x[-1], reverse=True):
        idx += 1
        filepath = output_target_log_csv_filepath % (domain, k, pattern_type, pattern_count, one_flattened_dep_rels)
        if os.path.exists(filepath): continue
        print('[%d/%d]Evaluating quality of pattern %s' % (idx, len(pattern_counter.keys()), one_flattened_dep_rels))
        dep_rels = one_flattened_dep_rels.split('-')
        df = copy.deepcopy(original_df)
        
        if pattern_type == 'ot': df['predicted_targets'] = df.parallel_apply(lambda x: pattern_handler.extract_targets(x['doc'], x['opinion_words'], dep_rels, dependency_handler), axis=1)
        elif pattern_type == 'tt': df['predicted_targets'] = df.parallel_apply(lambda x: pattern_handler.extract_targets(x['doc'], x['targets'], dep_rels, dependency_handler), axis=1)
        
        df['pattern'] = one_flattened_dep_rels
        df['pattern_count'] = pattern_count

        df.drop(['filename', 'doc'], axis=1, inplace=True)
        df.to_csv(filepath, index = False, encoding='utf-8-sig')
        print('Created %s' % filepath)
        
    filepath = output_targets_concat_csv_filepath % (domain, k, pattern_type)
    if os.path.exists(filepath): concat_df = pd.read_csv(filepath)
    else:
        print('Merging csv files in %s for [%s]..' % (output_targets_dir, domain))
        concat_df = merge_dfs(glob.glob(os.path.join(output_targets_dir, '%s_k=%d_%s*.csv' % (domain, k, pattern_type))))
        concat_df.to_csv(filepath, index = False, encoding='utf-8-sig')
        print('Created %s' % filepath)

    concat_df['targets'] = concat_df.apply(lambda x: ast.literal_eval(x['targets']), axis=1)
    concat_df['predicted_targets'] = concat_df.apply(lambda x: ast.literal_eval(x['predicted_targets']), axis=1)
    
    print('Evaluating rules for [%s k=%d]..' % (domain, k))
    filepath = output_pattern_quality_estimation_csv_filepath % (domain, k)
    if not os.path.exists(filepath): 
        f = open(filepath, 'w', encoding='utf-8-sig')
        wr = csv.writer(f)  
        wr.writerow(['domain', 'pattern', 'count', 'precision_multiple', 'recall_multiple', 'f1_multiple', 'precision_distinct', 'recall_distinct', 'f1_distinct'])
        for pattern in concat_df['pattern'].unique():
            current_df = concat_df[concat_df['pattern']==pattern]
            pre_mul, rec_mul, pre_dis, rec_dis = calculate_precision_recall(current_df)
            wr.writerow([domain, pattern, '%d'%current_df['pattern_count'].iloc[0], '%.2f'%pre_mul, '%.2f'%rec_mul, '%.2f'%calculate_f1(pre_mul,rec_mul), '%.2f'%pre_dis, '%.2f'%rec_dis, '%.2f'%calculate_f1(pre_dis,rec_dis)])
        f.close()
        print('Created %s' % filepath)
    pattern_evaluation_df = pd.read_csv(filepath)
    print('Loaded %s' % filepath)
    return concat_df, pattern_evaluation_df

def evaluate_rule_set(original_df, selected_pattern_list, pattern_handler, dependency_handler):
    df = copy.deepcopy(original_df)
    df['predicted_targets'] = df.apply(lambda x: list(), axis=1)
    for one_flattened_dep_rels in selected_pattern_list:
        dep_rels = one_flattened_dep_rels.split('-')
        df['predicted_targets'] = df.apply(lambda x: pattern_handler.extract_targets(x['doc'], x['opinion_words'], dep_rels, dependency_handler, x['predicted_targets']), axis=1)

    pre_mul, rec_mul, pre_dis, rec_dis = calculate_precision_recall(df)
    f1_mul = calculate_f1(pre_mul,rec_mul)
    f1_dis = calculate_f1(pre_dis,rec_dis)
    return pre_mul, rec_mul, f1_mul, f1_dis

def pick_least_redundant_one_pattern(selected_pattern_list, subset_handler):
    print('Picking out the least redundant one pattern against %s.. ' % str(selected_pattern_list))
    x1 = subset_handler.evaluate_patterns_tp(selected_pattern_list)['tp'].values.reshape(-1,1)
    
    redundancy_degree_score_df = pd.DataFrame([candidate_pattern for candidate_pattern in subset_handler.pattern_list if candidate_pattern not in selected_pattern_list], columns=['candidate_pattern'])
    redundancy_degree_score_df['redundancy_score'] = redundancy_degree_score_df.parallel_apply(lambda row: mutual_info_classif(x1, subset_handler.evaluate_patterns_tp([*selected_pattern_list, row['candidate_pattern']])['tp'].values.reshape(-1,1), discrete_features=[0]), axis=1)
    
    min_row = redundancy_degree_score_df.sort_values(by='redundancy_score').iloc[0]
    min_redundant_pattern = min_row['candidate_pattern']
    min_redundancy_degree_score = min_row['redundancy_score']
    
    return min_redundant_pattern, min_redundancy_degree_score

def pattern_subset_selection(domain, k, pattern_type, selected_pattern_list, original_df, subset_handler, pattern_handler, dependency_handler):
    print('Processing subset selection for [%s k=%d]..' % (domain, k))
    
    pkl_filepath = output_subset_pkl_filepath % (domain, k, pattern_type)
    if os.path.exists(pkl_filepath): best_subset = load_pkl(pkl_filepath)
    else:
        best_f1_mul, best_f1_dis, best_subset = 0, 0, []
        _, _, f1_mul, f1_dis = evaluate_rule_set(original_df, selected_pattern_list, pattern_handler, dependency_handler)
        content = "Selected pattern list = %s \n\tF1 (multiple): %.4f\tF1 (distinct): %.4f" % (str(selected_pattern_list), f1_mul, f1_dis)
        print("[%s]Selected pattern list = %s \n\tF1 (multiple): %.4f\tF1 (distinct): %.4f" % (domain, str(selected_pattern_list), f1_mul, f1_dis))
        while True:
            best_f1_mul, best_f1_dis, best_subset = f1_mul, f1_dis, copy.deepcopy(selected_pattern_list)

            min_redundant_pattern, mi_score = pick_least_redundant_one_pattern(selected_pattern_list, subset_handler)
            content += "\nLeast redundant pattern = %s [Redundancy score (MI score) = %.4f]" % (min_redundant_pattern, mi_score)
            print("[%s]Least redundant pattern = %s [Redundancy score (MI score) = %.4f]" % (domain, min_redundant_pattern, mi_score))
            selected_pattern_list.append(min_redundant_pattern)
            _, _, f1_mul, f1_dis = evaluate_rule_set(original_df, selected_pattern_list, pattern_handler, dependency_handler)
            content += "\n\nSelected pattern list = %s \n\tF1 (multiple): %.4f\tF1 (distinct): %.4f" % (str(selected_pattern_list), f1_mul, f1_dis)
            print("[%s]Selected pattern list = %s \n\tF1 (multiple): %.4f\tF1 (distinct): %.4f" % (domain, str(selected_pattern_list), f1_mul, f1_dis))

            if len(selected_pattern_list) == len(subset_handler.pattern_list): break
            if f1_mul < best_f1_mul: break

        content += "\n\n<Best> Selected pattern list = %s \n\tF1 (multiple): %.4f\tF1 (distinct): %.4f" % (str(best_subset), best_f1_mul, best_f1_dis)
        print("[%s]<Best> Selected pattern list = %s \n\tF1 (multiple): %.4f\tF1 (distinct): %.4f" % (domain, str(best_subset), best_f1_mul, best_f1_dis))
        
        filepath = output_subset_selection_log_filepath % (domain, k, pattern_type)
        text_file = open(filepath, "w", encoding='utf-8')
        text_file.write(content)
        text_file.close()
        print('Created %s' % filepath)
        save_pkl(best_subset, pkl_filepath)
    return best_subset

def main():
    kf, kfold_results = KFold(n_splits=10), defaultdict(lambda: [])
    pattern_handler, dependency_handler = PatternHandler(), DependencyGraphHandler()
    
    for domain in domains:   # raw_df['domain'].unique()
        output_raw_df_pkl_filepath = output_raw_df_pkl_filepath_ % domain
        if os.path.exists(output_raw_df_pkl_filepath): raw_df = load_pkl(output_raw_df_pkl_filepath)
        else:
            raw_df_ = pd.read_json(data_filepath)
            raw_df = raw_df_[raw_df_['domain']==domain]
            print('Matching opinion words..')
            opinion_word_lexicon = [item for sublist in pd.read_json(lexicon_filepath).values for item in sublist]
            raw_df['opinion_words'] = raw_df.parallel_apply(lambda x: match_opinion_words(x['content'], opinion_word_lexicon), axis=1)
            print('\nConverting document into nlp(doc)..')
            raw_df['doc'] = raw_df.progress_apply(lambda x: pattern_handler.nlp(x['content']), axis=1)

            print('Filtering targets using nlp(doc)..')
            raw_df['targets'] = raw_df.progress_apply(lambda x: pattern_handler.process_targets(x['content'], x['raw_targets']), axis=1) 
            save_pkl(raw_df, output_raw_df_pkl_filepath)
    
        print('Processing [%s]..' % domain)
        df = raw_df[raw_df['domain']==domain]
        
        filepath = output_training_test_dfs_pkl_filepath % domain
        if os.path.exists(filepath): 
            training_test_dfs = load_pkl(filepath)
            training_dfs, test_dfs = training_test_dfs[0], training_test_dfs[1]
        else:
            training_dfs, test_dfs = [], []
            for train_indices, test_indices in kf.split(df):
                training_df, test_df = df.iloc[train_indices], df.iloc[test_indices]
                training_dfs.append(training_df)
                test_dfs.append(test_df)
            training_test_dfs = [training_dfs, test_dfs]
            save_pkl(training_test_dfs, filepath)
        
        for k in range(len(training_dfs)):
            training_df, test_df = training_dfs[k], test_dfs[k]
        
            # Training
            whole_patterns, top_3_patterns, top_5_patterns, top_10_patterns = [], [], [], []
            for pattern_type in ['ot']:   # , 'tt'
                filepath = output_pattern_counter_pkl_filepath % (domain, k, pattern_type)
                if os.path.exists(filepath): pattern_counter = load_pkl(filepath)
                else: 
                    pattern_counter = pattern_extraction(domain, k, pattern_type, training_df, pattern_handler, dependency_handler)
                    save_pkl(pattern_counter, filepath)

                pattern_counter = {k:v for k,v in pattern_counter.items() if v > min_pattern_count}
                predicted_targets_df, pattern_evaluation_df = pattern_quality_estimation(domain, k, pattern_type, training_df, pattern_counter, pattern_handler, dependency_handler)
                pattern_evaluation_df = pattern_evaluation_df[pattern_evaluation_df['f1_multiple']>min_pattern_f1]

                print('Calculating true positives for each sentence..')
                predicted_targets_df['tp'] = predicted_targets_df.progress_apply(lambda row: calculate_true_positive(row['predicted_targets'], row['targets']), axis=1)

                subset_handler = SubsetHandler(domain, predicted_targets_df, pattern_evaluation_df)
                whole_patterns.extend(subset_handler.pattern_list)
                top_3_patterns.extend(subset_handler.pattern_list[:3])
                top_5_patterns.extend(subset_handler.pattern_list[:5])
                top_10_patterns.extend(subset_handler.pattern_list[:])
                if pattern_type == 'ot': best_subset = [subset_handler.pattern_list[0]]
                best_subset = pattern_subset_selection(domain, k, pattern_type, best_subset, training_df, subset_handler, pattern_handler, dependency_handler)

            # Test
            filepath = output_target_extraction_report_csv_filepath % (domain, k)
            f, wr = start_csv(filepath)
            wr.writerow(['Domain', 'Measure', 'All', 'Best subset (F1)', 'Top3 F1', 'Top5 F1', 'Top10 F1'])
            
            all_pre_mul, all_rec_mul, all_f1_mul, all_f1_dis = evaluate_rule_set(test_df, whole_patterns, pattern_handler, dependency_handler)
            best_pre_mul, best_rec_mul, best_f1_mul, _ = evaluate_rule_set(test_df, best_subset, pattern_handler, dependency_handler)
            top3_pre_mul, top3_rec_mul, top3_f1_mul, top3_f1_dis = evaluate_rule_set(test_df, top_3_patterns, pattern_handler, dependency_handler)
            top5_pre_mul, top5_rec_mul, top5_f1_mul, top5_f1_dis = evaluate_rule_set(test_df, top_5_patterns, pattern_handler, dependency_handler)
            top10_pre_mul, top10_rec_mul, top10_f1_mul, top10_f1_dis = evaluate_rule_set(test_df, top_10_patterns, pattern_handler, dependency_handler)
            
            wr.writerow([domain, 'Precision', '%.4f'%all_pre_mul, '%.4f'%best_pre_mul, '%.4f'%top3_pre_mul, '%.4f'%top5_pre_mul, '%.4f'%top10_pre_mul])
            wr.writerow([domain, 'Recall', '%.4f'%all_rec_mul, '%.4f'%best_rec_mul, '%.4f'%top3_rec_mul, '%.4f'%top5_rec_mul, '%.4f'%top10_rec_mul])
            wr.writerow([domain, 'F1 score', '%.4f'%all_f1_mul, '%.4f'%best_f1_mul, '%.4f'%top3_f1_mul, '%.4f'%top5_f1_mul, '%.4f'%top10_f1_mul])
            wr.writerow([domain, 'Rules', '%d'%len(whole_patterns), '%s'%str(best_subset), '%s'%str(top_3_patterns), '%s'%str(top_5_patterns), '%s'%str(top_10_patterns)])
            
            kfold_results['_'.join([domain, 'Precision', 'All'])].append(all_pre_mul)
            kfold_results['_'.join([domain, 'Precision', 'Best subset (F1)'])].append(best_pre_mul)
            kfold_results['_'.join([domain, 'Precision', 'Top3 F1'])].append(top3_pre_mul)
            kfold_results['_'.join([domain, 'Precision', 'Top5 F1'])].append(top5_pre_mul)
            kfold_results['_'.join([domain, 'Precision', 'Top10 F1'])].append(top10_pre_mul)
            
            kfold_results['_'.join([domain, 'Recall', 'All'])].append(all_rec_mul)
            kfold_results['_'.join([domain, 'Recall', 'Best subset (F1)'])].append(best_rec_mul)
            kfold_results['_'.join([domain, 'Recall', 'Top3 F1'])].append(top3_rec_mul)
            kfold_results['_'.join([domain, 'Recall', 'Top5 F1'])].append(top5_rec_mul)
            kfold_results['_'.join([domain, 'Recall', 'Top10 F1'])].append(top10_rec_mul)
            
            kfold_results['_'.join([domain, 'F1 score', 'All'])].append(all_f1_mul)
            kfold_results['_'.join([domain, 'F1 score', 'Best subset (F1)'])].append(best_f1_mul)
            kfold_results['_'.join([domain, 'F1 score', 'Top3 F1'])].append(top3_f1_mul)
            kfold_results['_'.join([domain, 'F1 score', 'Top5 F1'])].append(top5_f1_mul)
            kfold_results['_'.join([domain, 'F1 score', 'Top10 F1'])].append(top10_f1_mul)
            
            kfold_results['_'.join([domain, 'Rules', 'All'])].append(len(whole_patterns))
            kfold_results['_'.join([domain, 'Rules', 'Best subset (F1)'])].append(len(best_subset))
            kfold_results['_'.join([domain, 'Rules', 'Top3 F1'])].append(len(top_3_patterns))
            kfold_results['_'.join([domain, 'Rules', 'Top5 F1'])].append(len(top_5_patterns))
            kfold_results['_'.join([domain, 'Rules', 'Top10 F1'])].append(len(top_10_patterns))
            end_csv(f, filepath)
    
    f, wr = start_csv(output_final_report_csv_filepath)
    wr.writerow(['Domain', 'Measure', 'All', 'Best subset (F1)', 'Top3 F1', 'Top5 F1', 'Top10 F1'])
    for domain in domains:
        for measure in ['Precision', 'Recall', 'F1 score', 'Rules']:
            wr.writerow([domain, measure, '%.4f'%np.mean(kfold_results['_'.join([domain, measure, 'All'])]), '%.4f'%np.mean(kfold_results['_'.join([domain, measure, 'Best subset (F1)'])]), '%.4f'%np.mean(kfold_results['_'.join([domain, measure, 'Top3 F1'])]), '%.4f'%np.mean(kfold_results['_'.join([domain, measure, 'Top5 F1'])]), '%.4f'%np.mean(kfold_results['_'.join([domain, measure, 'Top10 F1'])]) ])
    end_csv(f, output_final_report_csv_filepath)
    
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