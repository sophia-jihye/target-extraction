from collections import defaultdict, namedtuple
import networkx as nx
import re

Node = namedtuple("Node", ["idx", "token", "pos", "dep", "governor"])
class DependencyGraphHandler:
    def __init__(self):
        self.number_pattern = re.compile('\d+')
        
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

    def extract_targets_using_pattern(self, token2idx, nodes, opinion_words, dep_rels):

        focused_tokens = opinion_words
        for i in range(len(dep_rels)):
            focused_tokens = self.next_focused_tokens(focused_tokens, token2idx, nodes, dep_rels[i])

        return set(list(focused_tokens))