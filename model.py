import numpy as np
import collections

class UCT_Model_Node(object):
    def __init__(self, w, n):
        self.win = w
        self.n = n
    
class UCT_Model(object):
    def __init__(self, init_w=1, init_n=3):
        self.nodes = collections.defaultdict(lambda: UCT_Model_Node(init_w,init_n))
    
    def evaluate_and_select(self, moves, c=1.414):
        '''
        params:
            nodes: List
        return:
            the node who has the max ucb value
        '''
        nodes = [self.nodes[move] for move in moves]
        N = sum(node.n for node in nodes)
        ucb_vals =[self._ucb(node.win, node.n, N, c) for node in nodes]
        total_val = sum(ucb_vals)
        p = [v/total_val for v in ucb_vals]
        index = np.random.choice(len(ucb_vals), 1,p=p)[0]
        return moves[index]

    def update(self, node, res):
        node.win += res
        node.n += 1

    def _ucb(self, win, n, N, c):
        return win * 1.0 / n + c * ((np.log(N)/n) ** 0.5)

    def reset(self):
        self.nodes.clear()

    
