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

data_filepath = parameters.data_filepath
lexicon_filepath = parameters.lexicon_filepath
output_time_txt_filepath = parameters.output_time_txt_filepath
output_pattern_csv_filepath = parameters.output_pattern_csv_filepath
output_error_csv_filepath = parameters.output_error_csv_filepath
output_target_log_csv_filepath = parameters.output_target_log_csv_filepath
output_raw_df_pkl_filepath = parameters.output_raw_df_pkl_filepath
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

def save_extracted_pattern_results(domain, k, pattern_counter, err_list):
    pattern_list = [tup for tup in pattern_counter.items()]
    pattern_df = pd.DataFrame(pattern_list, columns =['pattern', 'count'])  
    filepath = output_pattern_csv_filepath % (domain, k, len(pattern_df))
    pattern_df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)
    
    err_df = pd.DataFrame(err_list, columns =['content', 'current_opinion_word', 'current_target_word', 'parse_error', 'opinion_words', 'targets', 'raw_targets'])  
    filepath = output_error_csv_filepath % (domain, k, len(err_df[err_df['parse_error']==True]), len(err_df))
    err_df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)

def pattern_extraction(domain, k, df, pattern_handler, dependency_handler):
    print('[%s k=%d]' % (domain, k))
    pattern_counter, err_list = defaultdict(int), list()
    pattern_handler.extract_patterns(df, pattern_counter, err_list, dependency_handler)
    
    save_extracted_pattern_results(domain, k, pattern_counter, err_list)
    return pattern_counter

def merge_dfs(data_filepaths):
    dfs = []
    for data_filepath in data_filepaths:
        df = pd.read_csv(data_filepath)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def pattern_quality_estimation(domain, k, original_df, pattern_counter, pattern_handler, dependency_handler):
    print('Processing pattern_quality_estimation for [%s] (# of patterns = %d)..' % (domain, len(pattern_counter)))
    idx = 0
    for one_flattened_dep_rels, pattern_count in sorted(pattern_counter.items(), key=lambda x: x[-1], reverse=True):
        idx += 1
        filepath = output_target_log_csv_filepath % (domain, k, pattern_count, one_flattened_dep_rels)
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
        
    filepath = output_targets_concat_csv_filepath % (domain, k)
    if os.path.exists(filepath): concat_df = pd.read_csv(filepath)
    else:
        print('Merging csv files in %s for [%s]..' % (output_targets_dir, domain))
        concat_df = merge_dfs(glob.glob(os.path.join(output_targets_dir, '%s_k=%d_*.csv' % (domain, k))))
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

def pattern_subset_selection(domain, k, original_df, subset_handler, pattern_handler, dependency_handler, end_condition='F1', end_condition_mi_score=1):
    print('Processing subset selection for [%s k=%d]..' % (domain, k))
    
    pkl_filepath = output_subset_pkl_filepath % (domain, k)
    if os.path.exists(pkl_filepath): best_subset = load_pkl(pkl_filepath)
    else:
        best_f1_mul, best_f1_dis, best_subset = 0, 0, []
        selected_pattern_list = [subset_handler.pattern_list[0]]
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
            if end_condition == 'F1' and f1_mul < best_f1_mul: break
            if end_condition == 'MI' and mi_score > end_condition_mi_score: break

        content += "\n\n<Best> Selected pattern list = %s \n\tF1 (multiple): %.4f\tF1 (distinct): %.4f" % (str(best_subset), best_f1_mul, best_f1_dis)
        print("[%s]<Best> Selected pattern list = %s \n\tF1 (multiple): %.4f\tF1 (distinct): %.4f" % (domain, str(best_subset), best_f1_mul, best_f1_dis))
        
        filepath = output_subset_selection_log_filepath % (domain, k)
        text_file = open(filepath, "w", encoding='utf-8')
        text_file.write(content)
        text_file.close()
        print('Created %s' % filepath)
        save_pkl(best_subset, pkl_filepath)
    return best_subset

def main():
    kf, kfold_results = KFold(n_splits=10), defaultdict(lambda: [])
    pattern_handler, dependency_handler = PatternHandler(), DependencyGraphHandler()
    
    if os.path.exists(output_raw_df_pkl_filepath): raw_df = load_pkl(output_raw_df_pkl_filepath)
    else:
        raw_df = pd.read_json(data_filepath)
        print('Matching opinion words..')
        opinion_word_lexicon = [item for sublist in pd.read_json(lexicon_filepath).values for item in sublist]
        raw_df['opinion_words'] = raw_df.parallel_apply(lambda x: match_opinion_words(x['content'], opinion_word_lexicon), axis=1)
        print('\nConverting document into nlp(doc)..')
        raw_df['doc'] = raw_df.progress_apply(lambda x: pattern_handler.nlp(x['content']), axis=1)
        
        print('Filtering targets using nlp(doc)..')
        raw_df['targets'] = raw_df.progress_apply(lambda x: pattern_handler.process_targets(x['content'], x['raw_targets']), axis=1) 
        save_pkl(raw_df, output_raw_df_pkl_filepath)
    
    for domain in domains:   # raw_df['domain'].unique()
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
            filepath = output_pattern_counter_pkl_filepath % (domain, k)
            if os.path.exists(filepath): pattern_counter = load_pkl(filepath)
            else: 
                pattern_counter = pattern_extraction(domain, k, training_df, pattern_handler, dependency_handler)
                save_pkl(pattern_counter, filepath)
            
            predicted_targets_df, pattern_evaluation_df = pattern_quality_estimation(domain, k, training_df, pattern_counter, pattern_handler, dependency_handler)

            print('Calculating true positives for each sentence..')
            predicted_targets_df['tp'] = predicted_targets_df.progress_apply(lambda row: calculate_true_positive(row['predicted_targets'], row['targets']), axis=1)

            subset_handler = SubsetHandler(domain, predicted_targets_df, pattern_evaluation_df)
            best_subset = pattern_subset_selection(domain, k, training_df, subset_handler, pattern_handler, dependency_handler)
            
            best_subset_mi5 = pattern_subset_selection(domain, k, training_df, subset_handler, pattern_handler, dependency_handler, end_condition='MI', end_condition_mi_score=0.5)
            best_subset_mi6 = pattern_subset_selection(domain, k, training_df, subset_handler, pattern_handler, dependency_handler, end_condition='MI', end_condition_mi_score=0.6)
            best_subset_mi7 = pattern_subset_selection(domain, k, training_df, subset_handler, pattern_handler, dependency_handler, end_condition='MI', end_condition_mi_score=0.7)
            best_subset_mi8 = pattern_subset_selection(domain, k, training_df, subset_handler, pattern_handler, dependency_handler, end_condition='MI', end_condition_mi_score=0.8)

            # Test
            filepath = output_target_extraction_report_csv_filepath % (domain, k)
            f, wr = start_csv(filepath)
            wr.writerow(['Domain', 'Measure', 'All', 'Best subset (F1)', 'Best subset (MI 0.5)', 'Best subset (MI 0.6)', 'Best subset (MI 0.7)', 'Best subset (MI 0.8)'])
            all_pre_mul, all_rec_mul, all_f1_mul, all_f1_dis = evaluate_rule_set(test_df, subset_handler.pattern_list, pattern_handler, dependency_handler)
            best_pre_mul, best_rec_mul, best_f1_mul, _ = evaluate_rule_set(test_df, best_subset, pattern_handler, dependency_handler)
            best_pre_mul_mi5, best_rec_mul_mi5, best_f1_mul_mi5, _ = evaluate_rule_set(test_df, best_subset_mi5, pattern_handler, dependency_handler)
            best_pre_mul_mi6, best_rec_mul_mi6, best_f1_mul_mi6, _ = evaluate_rule_set(test_df, best_subset_mi6, pattern_handler, dependency_handler)
            best_pre_mul_mi7, best_rec_mul_mi7, best_f1_mul_mi7, _ = evaluate_rule_set(test_df, best_subset_mi7, pattern_handler, dependency_handler)
            best_pre_mul_mi8, best_rec_mul_mi8, best_f1_mul_mi8, _ = evaluate_rule_set(test_df, best_subset_mi8, pattern_handler, dependency_handler)
            wr.writerow([domain, 'Precision', '%.4f'%all_pre_mul, '%.4f'%best_pre_mul, '%.4f'%best_pre_mul_mi5, '%.4f'%best_pre_mul_mi6, '%.4f'%best_pre_mul_mi7, '%.4f'%best_pre_mul_mi8])
            wr.writerow([domain, 'Recall', '%.4f'%all_rec_mul, '%.4f'%best_rec_mul, '%.4f'%best_rec_mul_mi5, '%.4f'%best_rec_mul_mi6, '%.4f'%best_rec_mul_mi7, '%.4f'%best_rec_mul_mi8])
            wr.writerow([domain, 'F1 score', '%.4f'%all_f1_mul, '%.4f'%best_f1_mul, '%.4f'%best_f1_mul_mi5, '%.4f'%best_f1_mul_mi6, '%.4f'%best_f1_mul_mi7, '%.4f'%best_f1_mul_mi8])
            kfold_results['_'.join([domain, 'Precision', 'All'])].append(all_pre_mul)
            kfold_results['_'.join([domain, 'Precision', 'Best subset (F1)'])].append(best_pre_mul)
            kfold_results['_'.join([domain, 'Precision', 'Best subset (MI 0.5)'])].append(best_pre_mul_mi5)
            kfold_results['_'.join([domain, 'Precision', 'Best subset (MI 0.6)'])].append(best_pre_mul_mi6)
            kfold_results['_'.join([domain, 'Precision', 'Best subset (MI 0.7)'])].append(best_pre_mul_mi7)
            kfold_results['_'.join([domain, 'Precision', 'Best subset (MI 0.8)'])].append(best_pre_mul_mi8)
            kfold_results['_'.join([domain, 'Recall', 'All'])].append(all_rec_mul)
            kfold_results['_'.join([domain, 'Recall', 'Best subset (F1)'])].append(best_rec_mul)
            kfold_results['_'.join([domain, 'Recall', 'Best subset (MI 0.5)'])].append(best_rec_mul_mi5)
            kfold_results['_'.join([domain, 'Recall', 'Best subset (MI 0.6)'])].append(best_rec_mul_mi6)
            kfold_results['_'.join([domain, 'Recall', 'Best subset (MI 0.7)'])].append(best_rec_mul_mi7)
            kfold_results['_'.join([domain, 'Recall', 'Best subset (MI 0.8)'])].append(best_rec_mul_mi8)
            kfold_results['_'.join([domain, 'F1 score', 'All'])].append(all_f1_mul)
            kfold_results['_'.join([domain, 'F1 score', 'Best subset (F1)'])].append(best_f1_mul)
            kfold_results['_'.join([domain, 'F1 score', 'Best subset (MI 0.5)'])].append(best_f1_mul_mi5)
            kfold_results['_'.join([domain, 'F1 score', 'Best subset (MI 0.6)'])].append(best_f1_mul_mi6)
            kfold_results['_'.join([domain, 'F1 score', 'Best subset (MI 0.7)'])].append(best_f1_mul_mi7)
            kfold_results['_'.join([domain, 'F1 score', 'Best subset (MI 0.8)'])].append(best_f1_mul_mi8)
            end_csv(f, filepath)
    
    f, wr = start_csv(output_final_report_csv_filepath)
    wr.writerow(['Domain', 'Measure', 'All', 'Best subset (F1)', 'Best subset (MI 0.5)', 'Best subset (MI 0.6)', 'Best subset (MI 0.7)', 'Best subset (MI 0.8)'])
    for domain in raw_df['domain'].unique():
        for measure in ['Precision', 'Recall', 'F1 score']:
            wr.writerow([domain, measure, '%.4f'%np.mean(kfold_results['_'.join([domain, measure, 'All'])]), '%.4f'%np.mean(kfold_results['_'.join([domain, measure, 'Best subset (F1)'])]), '%.4f'%np.mean(kfold_results['_'.join([domain, measure, 'Best subset (MI 0.5)'])]), '%.4f'%np.mean(kfold_results['_'.join([domain, measure, 'Best subset (MI 0.6)'])]), '%.4f'%np.mean(kfold_results['_'.join([domain, measure, 'Best subset (MI 0.7)'])]), '%.4f'%np.mean(kfold_results['_'.join([domain, measure, 'Best subset (MI 0.8)'])])])
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