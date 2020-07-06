import time
from config import parameters
from PatternExtractor import PatternExtractor
from DependencyGraphHandler import DependencyGraphHandler
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
tqdm.pandas()

data_filepath = parameters.data_filepath
output_time_txt_filepath = parameters.output_time_txt_filepath
output_pattern_csv_filepath = parameters.output_pattern_csv_filepath
output_error_csv_filepath = parameters.output_error_csv_filepath
                
def save_extracted_pattern_results(pattern_counter, err_list):
    pattern_list = [tup for tup in pattern_counter.items()]
    pattern_df = pd.DataFrame(pattern_list, columns =['pattern', 'count'])  
    filepath = output_pattern_csv_filepath % len(pattern_df)
    pattern_df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)
    
    err_df = pd.DataFrame(err_list, columns =['content', 'current_opinion_word', 'current_target_word', 'parse_error', 'opinion_words', 'targets', 'raw_targets'])  
    filepath = output_error_csv_filepath % len(err_df[err_df['parse_error']==True])
    err_df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)

def pattern_extraction(df, pattern_extractor, dependency_handler):
    df['targets'] = df.apply(lambda x: pattern_extractor.process_targets(x['content'], x['target']), axis=1) 
    df['opinion_words'] = df.progress_apply(lambda x: pattern_extractor.match_opinion_words(x['content']), axis=1)
    
    pattern_counter, err_list = defaultdict(int), list()
    pattern_extractor.extract_patterns(df, pattern_counter, err_list, dependency_handler)
    
    save_extracted_pattern_results(pattern_counter, err_list)
    return pattern_counter
    
def main():
    df = pd.read_json(data_filepath)
    pattern_extractor = PatternExtractor()
    dependency_handler = DependencyGraphHandler()
    
    pattern_counter = pattern_extraction(df, pattern_extractor, dependency_handler)
    
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