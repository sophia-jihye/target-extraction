import pandas as pd
import copy

class SubsetHandler:
    def __init__(self, domain, whole_predicted_targets_df, pattern_evaluation_df):
        self.domain = domain
        self.whole_predicted_targets_df = whole_predicted_targets_df
        self.pattern_list = pattern_evaluation_df.sort_values(by='f1_multiple', ascending=False)['pattern'].values
        self.content_frame_df = pd.DataFrame(whole_predicted_targets_df[whole_predicted_targets_df['pattern']==whole_predicted_targets_df['pattern'].unique()[0]]['content']).sort_values(by='content')
    
    def sum_tps(self, content, including_patterns_list):
        df = self.whole_predicted_targets_df[self.whole_predicted_targets_df['pattern'].isin(including_patterns_list)]
        return df[df['content']==content]['tp'].sum()
    
    def evaluate_patterns_tp(self, including_patterns_list):
        df = copy.deepcopy(self.content_frame_df)
        df['tp'] = df.apply(lambda x: self.sum_tps(x['content'], including_patterns_list), axis=1)
        df['tp'] = df.apply(lambda row: 1 if row['tp']>0 else 0, axis=1)
        return df
    
        