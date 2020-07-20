import pickle, csv

def save_pkl(item_to_save, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(item_to_save, f)
    print('Created %s' % filepath)

def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        loaded_item = pickle.load(f)
    print('Loaded %s' % filepath)
    return loaded_item

def start_csv(filepath):
    f = open(filepath, 'w', encoding='utf-8-sig')
    wr = csv.writer(f)
    return f, wr

def end_csv(f, filepath):
    f.close()
    print('Created %s' % filepath)

def calculate_true_positive(predicted_list, correct_list):
    predicted_list = [item for item in predicted_list if item != '']
    correct_list = [item for item in correct_list if item != '']
    tp = 0
    for predicted_compound_target in predicted_list:
        if predicted_compound_target in correct_list:   # 'screen' <- predicted 'screen'
            correct_list.remove(predicted_compound_target)
            tp += 1
            continue
        for correct_target in correct_list:
            if predicted_compound_target.find(correct_target) > -1:   # 'audio' <- predicted 'audio aspects'
                correct_list.remove(correct_target)
                tp += 1
                break
    return tp    
    
def calculate_precision_recall(df):
    correct_targets_mul = list([item for sublist in df['targets'].values for item in sublist if item != ''])
    predicted_targets_mul = list([item for sublist in df['predicted_targets'].values for item in sublist if item != ''])
    tp_mul = calculate_true_positive(predicted_targets_mul, correct_targets_mul)
    
    if len(predicted_targets_mul) != 0: pre_mul = tp_mul / len(predicted_targets_mul)
    else: pre_mul = 0
    
    if len(correct_targets_mul) != 0: rec_mul = tp_mul / len(correct_targets_mul)
    else: rec_mul = 0
    
    correct_targets_dis = set([item for sublist in df['targets'].values for item in sublist if item != ''])
    predicted_targets_dis = set([item for sublist in df['predicted_targets'].values for item in sublist if item != ''])
    tp_dis = calculate_true_positive(predicted_targets_dis, correct_targets_dis)
    
    if len(predicted_targets_dis) != 0: pre_dis = tp_dis / len(predicted_targets_dis)
    else: pre_dis = 0
    
    if len(correct_targets_dis) != 0: rec_dis = tp_dis / len(correct_targets_dis)
    else: rec_dis = 0
    
    return pre_mul, rec_mul, pre_dis, rec_dis

def calculate_f1(precision, recall):
    denominator = precision + recall
    if denominator == 0: return 0
    return (2*precision*recall)/denominator 