from collections import defaultdict, namedtuple
import networkx as nx

Node = namedtuple("Node", ["idx", "token", "pos", "dep", "governor"])
class DependencyGraphHandler:
    
    def governor2idx(self, old_idx):
        if old_idx == 0:
            return None
        return old_idx -1
    
    def parse_sentence(self, sentence):
        token2idx, nodes = defaultdict(lambda: []), []
        parsed_sent = sentence.dependencies
        for i in range(len(parsed_sent)):
            token2idx[parsed_sent[i][2].text].append(i)
            node = Node(i, parsed_sent[i][2].text, parsed_sent[i][2].xpos, parsed_sent[i][2].dependency_relation, self.governor2idx(parsed_sent[i][2].governor))
            nodes.append(node)
        return token2idx, nodes
    
    NN = ['NN', 'NNS', 'NNP']
    def compound(self, new_targets, token2idx, nodes):
        to_be_deleted, to_be_added = set(), set()
        for target in new_targets:
            comp_child_nodes = [nodes[i] for i in range(len(nodes)) if nodes[i].governor in token2idx[target] and nodes[i].dep=='compound' and nodes[i].pos in NN]
            if len(comp_child_nodes) > 0:
                to_be_deleted.add(target)
                for child_node in comp_child_nodes:
                    to_be_added.add(' '.join([child_node.token, target]))
        for item in to_be_deleted:
            new_targets.remove(item)
        for item in to_be_added:
            new_targets.add(item)
            
    def handle_hyphen_or_compound(self, word, delimiter, token2idx, nodes):
        indices = []
        for token in word.split(delimiter):
            indices.extend(token2idx[token])
        root_words = [nodes[i].token for i in indices if nodes[i].governor not in indices]
        return root_words[0]   
    
    def get_pattern(self, sentence_from_doc, o_word, t_word):
        token2idx, nodes = self.parse_sentence(sentence_from_doc)

        edges = []
        token2tagdep = {}
        for item in sentence_from_doc.dependencies:
            token2tagdep[item[2].text]=(item[2].text, item[2].xpos, item[2].dependency_relation)
            if item[0].text.lower() != 'root':
                edges.append((item[0].text, item[2].text))
        graph = nx.Graph(edges)

        entity1, entity2 = o_word, t_word
        try: shortest_path = nx.shortest_path(graph, source=entity1, target=entity2)
        except:
            if '-' in o_word: entity1 = self.handle_hyphen_or_compound(o_word, '-', token2idx, nodes)
            if ' ' in t_word: entity2 = self.handle_hyphen_or_compound(t_word, ' ', token2idx, nodes)
            shortest_path = nx.shortest_path(graph, source=entity1, target=entity2)
                
        return [token2tagdep[token] for token in shortest_path]