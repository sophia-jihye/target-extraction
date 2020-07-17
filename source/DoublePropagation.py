class DoublePropagation:
    def __init__(self):
        self.MR = ['amod', 'nmod', 'advmod', 'rcmod', 'nsubj', 'obl']
        self.NN = ['NN', 'NNS', 'NNP']

    def next_compound_token_idx(self, current_token_idx, nodes):
        compound_child_nodes = [nodes[i] for i in range(len(nodes)) if nodes[i].governor == current_token_idx and nodes[i].dep.startswith('compound')]
        
        if len(compound_child_nodes) > 0: return compound_child_nodes[0].idx
        return None        
        
    def compound(self, new_targets, indices_for_new_targets, nodes):
        to_be_deleted, to_be_added = set(), set()
        for target_idx in indices_for_new_targets:
            
            compound_token_indices = list()
            focused_token_idx = target_idx
            while True:
                focused_token_idx = self.next_compound_token_idx(focused_token_idx, nodes)
                if focused_token_idx is None or focused_token_idx in compound_token_indices: break
                compound_token_indices.append(focused_token_idx)
            compound_token_indices.append(target_idx)

            if len(compound_token_indices) > 1:
                to_be_deleted.add(nodes[target_idx].token)
                to_be_added.add(' '.join([nodes[i].token for i in compound_token_indices]))
        for item in to_be_deleted:
            new_targets.remove(item)
        for item in to_be_added:
            new_targets.add(item)
    
    def extract_targets_R11(self, opinion_words, token2indices, nodes):
        focused_token_indices = set()
        for o in opinion_words:
            indices = token2indices[o]
            for o_idx in indices:

                # O(JJ) ~ MR <- T(NN)   *The phone has a big "screen"
                if nodes[o_idx].dep in self.MR and nodes[nodes[o_idx].governor].pos in self.NN:
                    focused_token_indices.add(nodes[nodes[o_idx].governor].idx)

                # O(JJ) -> MR ~ T(NN)   *The "screen" is big
                child_nodes = [nodes[i] for i in range(len(nodes)) if nodes[i].governor==o_idx]
                focused_token_indices.update([child_node.idx for child_node in child_nodes if child_node.dep in self.MR and child_node.pos in self.NN])
        
        new_targets = set([nodes[i].token for i in focused_token_indices])
        self.compound(new_targets, focused_token_indices, nodes)
        return list(new_targets)
    
    def extract_targets_R12(self, opinion_words, token2indices, nodes):
        focused_token_indices = set()
        for o in opinion_words:
            indices = token2indices[o]
            for o_idx in indices:

                # O(JJ) ~ MR <- H -> MR ~ T(NN)   *"iPod" is the best mp3 player
                if nodes[o_idx].dep in self.MR:
                    h_indices = token2indices[nodes[nodes[o_idx].governor].token]
                    child_nodes = [nodes[i] for i in range(len(nodes)) if nodes[i].governor in h_indices]
                    focused_token_indices.update([child_node.idx for child_node in child_nodes if child_node.dep in self.MR and child_node.pos in self.NN])
        
        new_targets = set([nodes[i].token for i in focused_token_indices])
        self.compound(new_targets, focused_token_indices, nodes)
        return list(new_targets)