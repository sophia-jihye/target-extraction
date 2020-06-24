import stanfordnlp, json
from collections import defaultdict
from collections import namedtuple
from config import parameters

Node = namedtuple("Node", ["idx", "token", "pos", "dep", "governor"])
class Rules:
    def __init__(self):
        self.nlp = stanfordnlp.Pipeline()  # use_gpu=True
        self.MR = ['amod', 'nmod', 'advmod', 'rcmod', 'nsubj', 'obl']
        self.CONJ = ['conj']
        self.NN = ['NN', 'NNS', 'NNP']
        self.JJ = ['JJ', 'JJR', 'JJS', 'JJP']
        self.opinion_words = self.get_opinion_words(parameters.lexicon_filepath)

    def get_opinion_words(self, filepath):
        with open(filepath,"r") as f:
            opinion_words = json.load(f)
        return opinion_words
        
    def governor2idx(self, old_idx):
        if old_idx == 0:
            return None
        return old_idx -1
    
    def parse_sentence(self, sentence):
        token2idx, node_list = defaultdict(lambda: []), []
        parsed_sent = sentence.dependencies
        for i in range(len(parsed_sent)):
            token2idx[parsed_sent[i][2].text].append(i)
            node = Node(i, parsed_sent[i][2].text, parsed_sent[i][2].xpos, parsed_sent[i][2].dependency_relation, self.governor2idx(parsed_sent[i][2].governor))
            node_list.append(node)
        return token2idx, node_list
    
    def compound(self, new_targets, token2idx, nodes):
        to_be_deleted, to_be_added = set(), set()
        for target in new_targets:
            comp_child_nodes = [nodes[i] for i in range(len(nodes)) if nodes[i].governor in token2idx[target] and nodes[i].dep=='compound' and nodes[i].pos in self.NN]
            if len(comp_child_nodes) > 0:
                to_be_deleted.add(target)
                for child_node in comp_child_nodes:
                    to_be_added.add(' '.join([child_node.token, target]))
        for item in to_be_deleted:
            new_targets.remove(item)
        for item in to_be_added:
            new_targets.add(item)
    
    def extract_targets_R11(self, token2idx, nodes):
        new_targets = set()
        opinions_in_sentence = [token for token in token2idx.keys() if token in self.opinion_words]
        for o in opinions_in_sentence:
            indices = token2idx[o]
            for o_idx in indices:

                # O(JJ) ~ MR <- T(NN)   *The phone has a big "screen"
                if nodes[o_idx].dep in self.MR and nodes[nodes[o_idx].governor].pos in self.NN:
                    new_targets.add(nodes[nodes[o_idx].governor].token)

                # O(JJ) -> MR ~ T(NN)   *The "screen" is big
                child_nodes = [nodes[i] for i in range(len(nodes)) if nodes[i].governor==o_idx]
                new_targets.update([child_node.token for child_node in child_nodes if child_node.dep in self.MR and child_node.pos in self.NN])
        
        # This is for expanding the opinion targets from single word to phrases (e.g., battery life)
        self.compound(new_targets, token2idx, nodes)
        return list(new_targets)
    
    def extract_targets_R12(self, token2idx, nodes):
        new_targets = set()
        opinions_in_sentence = [token for token in token2idx.keys() if token in self.opinion_words]
        for o in opinions_in_sentence:
            indices = token2idx[o]
            for o_idx in indices:

                # O(JJ) ~ MR <- H -> MR ~ T(NN)   *"iPod" is the best mp3 player
                if nodes[o_idx].dep in self.MR:
                    h_indices = token2idx[nodes[nodes[o_idx].governor].token]
                    child_nodes = [nodes[i] for i in range(len(nodes)) if nodes[i].governor in h_indices]
                    new_targets.update([child_node.token for child_node in child_nodes if child_node.dep in self.MR and child_node.pos in self.NN])
        
        # This is for expanding the opinion targets from single word to phrases (e.g., battery life)
        self.compound(new_targets, token2idx, nodes)
        return list(new_targets)
    