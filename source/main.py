import time, copy, os
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
    df['targets'] = df.apply(lambda x: pattern_handler.process_targets(x['content'], x['raw_targets']), axis=1) 
    
    pattern_counter, err_list = defaultdict(int), list()
    pattern_handler.extract_patterns(df, pattern_counter, err_list, dependency_handler)
    
    save_extracted_pattern_results(domain, pattern_counter, err_list)
    return pattern_counter

def pattern_quality_estimation(domain, original_df, pattern_counter, pattern_handler, dependency_handler):
    dfs = []
    for idx, one_flattened_dep_rels in enumerate(pattern_counter.keys()):
        print('[%d/%d]Extracting targets by pattern %s' % (idx, len(pattern_counter.keys()), one_flattened_dep_rels))
        dep_rels = one_flattened_dep_rels.split('-')
        df = copy.deepcopy(original_df)
        df['doc'] = df.progress_apply(lambda x: pattern_handler.nlp(x['content']), axis=1)
        df['predicted_targets'] = df.parallel_apply(lambda x: pattern_handler.extract_targets(x['doc'], x['opinion_words'], dep_rels, dependency_handler), axis=1)
        df['pattern'] = one_flattened_dep_rels
        dfs.append(df)
    concat_df = pd.concat(dfs, ignore_index=True)
    filepath = output_target_log_csv_filepath % domain
    concat_df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)
    return concat_df

def main():
    raw_df = pd.read_json(data_filepath)
    pattern_handler = PatternHandler()
    dependency_handler = DependencyGraphHandler()
    
    print('Matching opinion words..')
    opinion_word_lexicon = [item for sublist in pd.read_json(lexicon_filepath).values for item in sublist]
    raw_df['opinion_words'] = raw_df.parallel_apply(lambda x: match_opinion_words(x['content'], opinion_word_lexicon), axis=1)
    
    for domain in raw_df['domain'].unique():
        print('Processing %s..' % domain)
        df = raw_df[raw_df['domain']==domain]
        pattern_counter = pattern_extraction(domain, df, pattern_handler, dependency_handler)
        
        concat_df = pattern_quality_estimation(domain, df, pattern_counter, pattern_handler, dependency_handler)
    
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