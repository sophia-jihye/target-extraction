import time, stanfordnlp, re
from collections import defaultdict
from config import parameters
import pandas as pd
from DependencyGraphHandler import DependencyGraphHandler
from tqdm import tqdm
tqdm.pandas()

data_filepath = parameters.data_filepath
output_time_txt_filepath = parameters.output_time_txt_filepath
lexicon_filepath = parameters.lexicon_filepath
output_pattern_csv_filepath = parameters.output_pattern_csv_filepath
output_error_csv_filepath = parameters.output_error_csv_filepath

special_char_pattern = re.compile('([,.]+.?\d*)')
def process_targets(content, targets):
    content = special_char_pattern.sub(' ', content)   # dvds -> dvds (o) dvds.(x)
    processed_targets = []
    for target in targets:
        candidate_token = None
        
        if len(target.split()) > 1:   # compound target
            compound_target_with_spaces = ' ' + target + ' '
            content_with_spaces = ' ' + content + ' '
            if compound_target_with_spaces in content_with_spaces:
                candidate_token = target
        else:   # unigram target
            for token in content.split():
                if target == token:   # price -> price (o) lower-priced (x)
                    candidate_token = token
                    break
                if target in token: candidate_token = token   # rip -> ripping
        
        if candidate_token is not None: processed_targets.append(candidate_token) # transfter (MISSPELLED) -> (DROP)
    
    processed_targets = [item for item in processed_targets if item != '']
    return list(set(processed_targets))

opinion_word_lexicon = [item for sublist in pd.read_json(lexicon_filepath).values for item in sublist]
def match_opinion_words(content):
    opinion_words = []
    for opinion in opinion_word_lexicon:
        for token in content.split():
            if token == opinion: opinion_words.append(token)
    return list(set(opinion_words))

def sentence_contains_token(sentence_from_doc, o_word, t_word):
    flattened_string = ''.join([sentence_from_doc.words[i].text for i in range(len(sentence_from_doc.words))])
    if o_word not in flattened_string or t_word not in flattened_string:
        return False
    return True

nlp = stanfordnlp.Pipeline()
def extract_patterns(df, pattern_counter, err_list, dependency_handler):
    cnt = 0
    for _, row in df.iterrows():    
        document = row['content']
        doc = nlp(document)
        for sentence_from_doc in doc.sentences:
            o_t = [(o,t) for o in row['opinion_words'] for t in row['targets']]
            for o_word,t_word in o_t:
                if sentence_contains_token(sentence_from_doc, o_word, t_word) == False:
                    continue

                try: 
                    extracted_patterns = dependency_handler.get_pattern(sentence_from_doc, o_word, t_word)
                    pattern_counter['-'.join([dep_rel for token, pos, dep_rel in extracted_patterns if dep_rel != 'root'])] += 1
                    parse_error = False
                except: 
                    parse_error = True
                err_list.append([row['content'], o_word, t_word, parse_error, row['opinion_words'], row['targets'], row['target']])
                if cnt % 100 == 0: print('[%04dth] Extracting patterns..' % (cnt))
                cnt += 1
                
def save_results(pattern_counter, err_list):
    pattern_list = [tup for tup in pattern_counter.items()]
    pattern_df = pd.DataFrame(pattern_list, columns =['pattern', 'count'])  
    filepath = output_pattern_csv_filepath % len(pattern_df)
    pattern_df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)
    
    err_df = pd.DataFrame(err_list, columns =['content', 'current_opinion_word', 'current_target_word', 'parse_error', 'opinion_words', 'targets', 'raw_targets'])  
    filepath = output_error_csv_filepath % len(err_df[err_df['parse_error']==True])
    err_df.to_csv(filepath, index = False, encoding='utf-8-sig')
    print('Created %s' % filepath)
                    
def main():
    df = pd.read_json(data_filepath)
    df['targets'] = df.apply(lambda x: process_targets(x['content'], x['target']), axis=1) 
    df['opinion_words'] = df.progress_apply(lambda x: match_opinion_words(x['content']), axis=1)
    
    dependency_handler = DependencyGraphHandler()
    pattern_counter, err_list = defaultdict(int), list()
    extract_patterns(df, pattern_counter, err_list, dependency_handler)
    save_results(pattern_counter, err_list)
    
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