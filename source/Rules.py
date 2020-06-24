import stanfordnlp, json
from collections import defaultdict
from collections import namedtuple
from config import parameters

Node = namedtuple("Node", ["text", "lemma", "pos", "dep", "governor"])
class Rules:
    def __init__(self):
        self.nlp = stanfordnlp.Pipeline(use_gpu=True) 
        self.MR = ['amod', 'advmod', 'rcmod']
        self.NSUBJ = ['nsubj']
        self.CONJ = ['conj']
        self.NN = ['NN', 'NNS']
        self.JJ = ['JJ', 'JJR', 'JJS']
        self.opinion_words = self.get_opinion_words(parameters.lexicon_filepath)

    def get_opinion_words(self, filepath):
        with open(filepath,"r") as f:
            opinion_words = json.load(f)
        return opinion_words
        
    def governor2idx(self, old_idx):
        if old_idx == 0:
            return None
        return old_idx -1
    
    def parse_document(self, document):
        doc = self.nlp(document)
        lemma2idx = defaultdict(lambda: [])
        node_list = []
        for sent_id in range(len(doc.sentences)):
            parsed_sent = doc.sentences[sent_id].dependencies
            for i in range(len(parsed_sent)):
                lemma2idx[parsed_sent[i][2].lemma].append(i)
            for i in range(len(parsed_sent)):
                node = Node(parsed_sent[i][2].text, parsed_sent[i][2].lemma, parsed_sent[i][2].xpos, parsed_sent[i][2].dependency_relation, self.governor2idx(parsed_sent[i][2].governor))
                node_list.append(node)
        return lemma2idx, node_list
    
    def extract_targets_R11(self, lemma2idx, nodes):
        new_targets = set()
        opinions_in_sentence = [lemma for lemma in lemma2idx.keys() if lemma in self.opinion_words]
        for o in opinions_in_sentence:
            indices = lemma2idx[o]
            for o_idx in indices:

                # O(JJ) ~ MR <- T(NN)   *The phone has a big "screen"
                if nodes[o_idx].dep in self.MR and nodes[nodes[o_idx].governor].pos in self.NN:
                    new_targets.add(nodes[nodes[o_idx].governor].lemma)

                # O(JJ) -> nsubj ~ T   *The "screen" is big
                child_nodes = [nodes[i] for i in range(len(nodes)) if nodes[i].governor==o_idx]
                new_targets.update([child_node.lemma for child_node in child_nodes if child_node.dep in self.NSUBJ and child_node.pos in self.NN])

        if len(new_targets) == 0:
            return '-'
        
        # This is for expanding the opinion targets from single word to phrases (e.g., battery life)
        to_be_deleted, to_be_added = set(), set()
        for target in new_targets:
            comp_child_nodes = [nodes[i] for i in range(len(nodes)) if nodes[i].governor in lemma2idx[target] and nodes[i].dep=='compound' and nodes[i].pos in self.NN]
            if len(comp_child_nodes) > 0:
                to_be_deleted.add(target)
                for child_node in comp_child_nodes:
                    to_be_added.add(' '.join([child_node.lemma, target]))
        for item in to_be_deleted:
            new_targets.remove(item)
        for item in to_be_added:
            new_targets.add(item)
        return list(new_targets)
    