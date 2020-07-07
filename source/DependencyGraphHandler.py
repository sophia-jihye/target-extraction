from collections import defaultdict, namedtuple
import networkx as nx
import re
from config import parameters

Node = namedtuple("Node", ["idx", "token", "pos", "dep", "governor"])
class DependencyGraphHandler:
    def __init__(self):
        self.number_pattern = re.compile('\d+')
        self.errlog_filepath = parameters.errlog_filepath
        
    def write_errlog(self, content):
        filepath = self.errlog_filepath
        text_file = open(filepath, "w", encoding='utf-8')
        text_file.write(content)
        text_file.close()
        print("Error occurred: ", filepath)
        
    def handle_hyphen_or_compound(self, word, delimiter, token2idx, nodes):
        indices = []
        for token in word.split(delimiter):
            indices.extend(token2idx[token])
        root_words = [nodes[i].token for i in indices if nodes[i].governor not in indices]
        return root_words[0]   
    
    def remove_number(self, word):
        return self.number_pattern.sub('', word)
    
    def extract_patterns(self, token2idx, nodes, graph, token2tagdep, o_word, t_word):
        entity1, entity2 = o_word, t_word
        if o_word not in token2idx.keys() and '-' in o_word: 
            entity1 = self.handle_hyphen_or_compound(o_word, '-', token2idx, nodes)
        if t_word not in token2idx.keys() and '-' in t_word: 
            entity2 = self.handle_hyphen_or_compound(t_word, '-', token2idx, nodes)
        if t_word not in token2idx.keys() and ' ' in t_word: 
            entity2 = self.handle_hyphen_or_compound(t_word, ' ', token2idx, nodes)
        if t_word not in token2idx.keys() and bool(re.search(r'\d', t_word)):
            entity2 = self.remove_number(t_word)
        shortest_path = nx.shortest_path(graph, source=entity1, target=entity2)
                
        return [token2tagdep[token] for token in shortest_path]
    
    def next_focused_tokens(self, current_tokens, token2idx, nodes, dep_rel):
        focused_tokens = set()
        for current_token in current_tokens:
            indices = token2idx[current_token]
            for current_token_idx in indices:
                if nodes[current_token_idx].dep == dep_rel:
                    focused_tokens.add(nodes[nodes[current_token_idx].governor].token)
                child_nodes = [nodes[i] for i in range(len(nodes)) if nodes[i].governor==current_token_idx]
                focused_tokens.update([child_node.token for child_node in child_nodes if child_node.dep == dep_rel])
        return focused_tokens

    def next_compound_token(self, current_token, token2idx, nodes):
        compound_child_nodes = [nodes[i] for i in range(len(nodes)) if nodes[i].governor in token2idx[current_token] and nodes[i].dep.startswith('compound')]
        if len(compound_child_nodes) > 0: return compound_child_nodes[0].token
        return None
    
    def compound(self, new_targets, token2idx, nodes):
        to_be_deleted, to_be_added = set(), set()
        for target in new_targets:
            compound_tokens = list()

            focused_token = target
            while True:
                focused_token = self.next_compound_token(focused_token, token2idx, nodes)
                if focused_token is None: break
                compound_tokens.append(focused_token)
            compound_tokens.append(target)

            if len(compound_tokens) > 1:
                to_be_deleted.add(target)
                to_be_added.add(' '.join(compound_tokens))
        for item in to_be_deleted:
            new_targets.remove(item)
        for item in to_be_added:
            new_targets.add(item)
    
    def extract_targets_using_pattern(self, token2idx, nodes, opinion_words, dep_rels):

        focused_tokens = set(opinion_words)
        for i in range(len(dep_rels)):
            focused_tokens = self.next_focused_tokens(focused_tokens, token2idx, nodes, dep_rels[i])

        self.compound(focused_tokens, token2idx, nodes)
        return focused_tokens