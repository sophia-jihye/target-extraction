from collections import namedtuple, defaultdict
import networkx as nx

Node = namedtuple("Node", ["idx", "token", "pos", "dep", "governor"])
class DependencyGraph():
    def __init__(self, sentence_from_doc):
        self.token2idx, self.nodes = self.parse_sentence(sentence_from_doc)
        self.edges, self.graph = self.create_graph(sentence_from_doc)
    
    def governor2idx(self, old_idx):
        if old_idx == 0:
            return None
        return old_idx -1
    
    def parse_sentence(self, sentence_from_doc):
        token2idx, nodes = defaultdict(lambda: []), []
        parsed_sent = sentence_from_doc.dependencies
        for i in range(len(parsed_sent)):
            token2idx[parsed_sent[i][2].text].append(i)
            node = Node(i, parsed_sent[i][2].text, parsed_sent[i][2].xpos, parsed_sent[i][2].dependency_relation, self.governor2idx(parsed_sent[i][2].governor))
            nodes.append(node)
        return token2idx, nodes
    
    def create_graph(self, sentence_from_doc):
        edges = []
        token2tagdep = {}
        for item in sentence_from_doc.dependencies:
            token2tagdep[item[2].text]=(item[2].text, item[2].xpos, item[2].dependency_relation)
            if item[0].text.lower() != 'root':
                edges.append((item[0].text, item[2].text))
        graph = nx.Graph(edges)
        return edges, graph