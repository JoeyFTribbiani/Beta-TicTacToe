from model import UCT_Model
from tictactoe import TicTacToe
import random
import re
import numpy as np
from value_network import ValueNetwork
import dill
import time
class Robot(object):
    def __init__(self, game):
        self.model = UCT_Model()
        self.game = game
        self.game_states = {}
        INPUT_SHAPE = self.game.n_rows*self.game.n_cols
        HIDDEN_DIMS = (100,70,50,20,3)
        OUTPUT_SHAPE = 1
        self.value_network = ValueNetwork(INPUT_SHAPE, HIDDEN_DIMS, OUTPUT_SHAPE)
        self._cache_mapping_set = self._create_mapping_state_set()
        self._state_group_counter = -1
        
    def load_value_network(self, model_path):
        self.value_network.load_model(model_path)
        
    def generate_random_game_state(self,next_player=0,max_chess=0):
        if max_chess == 0:
            n = 0
        else:
            n = random.randint(0,max_chess)
        i = next_player
        x_num, o_num = n+i, n
        state = 'x' * x_num + 'o' * o_num +' ' * (self.game.n_rows * self.game.n_cols - x_num - o_num)
        return ''.join(random.sample(state,len(state)))

    # this function is to compress board state as the game is symetric

    def _statestring_to_statematrix(self, s):
        mat = []
        s = s.split(",")
        for i in range(self.game.n_rows):
            mat += list(s[i*self.game.n_cols:(i+1)*self.game.n_cols]),
        return mat
    
    def _rotate_clockwise_state(self, s):
        mat = self._statestring_to_statematrix(s)
        new_mat = []
        for row in list(zip(*mat)):
            new_mat += ','.join(row[::-1]),
        return ','.join(new_mat)

    def _flip_state(self, s):
        mat = self._statestring_to_statematrix(s)
        new_mat = []
        for row in mat[::-1]:
            new_mat += ','.join(row),
        return ','.join(new_mat)

    def _create_mapping_state_set(self):
        r = self._rotate_clockwise_state
        f = self._flip_state
        s_set = set([])
        def dfs(s,s_set):
            if s in s_set:
                return
            s_set.add(s)
            dfs(r(s), s_set)
            dfs(f(s), s_set)
            return
        s = ','.join([str(i) for i in range(self.game.n_rows*self.game.n_cols)])
        dfs(s, s_set)
        return s_set
                    
    def _find_mapping(self,s):
        transforms = self._cache_mapping_set
        if s not in self.game_states:
            self._state_group_counter += 1
            for trans in transforms:
                tmp_state = ''.join([s[int(i)] for i in trans.split(',')])
                self.game_states[tmp_state]=self._state_group_counter
        return self.game_states[s]

    def _gameboard_to_state(self):
        s = ''
        for i in range(self.game.n_rows):
            for j in range(self.game.n_cols):
                s += str(self.game.board[(i,j)])
        return s

    def _state_to_gameboard(self, s):
        for i in range(self.game.n_rows):
            for j in range(self.game.n_cols):
                self.game.fast_place(s[i*self.game.n_rows+j],(i,j), with_judge=False)

    def get_optimal_move_from_value_network(self, s):
        indices = [m.start() for m in re.finditer(' ', s)]
        moves = {self.value_network.predict(np.asarray([self.encode_state(s[:i]+'x'+s[i+1:])]))[0][0]:i for i in indices}
        i = moves[max(moves.keys())]
        return (i/self.game.n_rows, i%self.game.n_cols)
        
    def get_optimal_move_from_UCT_model(self, s, is_training=True):
        indices = [m.start() for m in re.finditer(' ', s)]
        moves = {self._find_mapping(s[:i]+'x'+s[i+1:]):i for i in indices}
        best_move = self.model.evaluate_and_select(moves.keys(),using_ucb=is_training)
        i = moves[best_move]
        if not is_training:
            return (i/self.game.n_rows, i%self.game.n_cols)
        return (i/self.game.n_rows, i%self.game.n_cols), best_move

    def swap_state(self, s):
        new_state = ''
        for e in s:
            if e == 'x':
                new_state += 'o'
            elif e == 'o':
                new_state += 'x'
            else:
                new_state += ' '
        return new_state
    
    def sample(self, n):
        x_win,o_win,tie = 0,0,0
        for i in xrange(n):
            if i == 0:
                print 'start training...'
            if i % 1000 == 0:
                print '{n} trains'.format(n=i)
                print 'tie:{tie},x_win{x_win},o_win:{o_win}'.format(tie=tie*1.0/(i+1),x_win=x_win*1.0/(i+1),o_win=o_win*1.0/(i+1))
            chesses = ['x', 'o']
            new = True
            player = random.randint(0,1)
            while new or self.game.is_end():
                self.game.reset()
                new = False
                init_state = self.generate_random_game_state(next_player=player,max_chess=0)
                self._state_to_gameboard(init_state)
                self.game.judge(print_msg=False)
            init_state_mapping = self._find_mapping(init_state)
            all_states_trans = {'o':[],'x':[]}
            all_states_trans[('x','o')[~player]]+=init_state_mapping,
            while not self.game.is_end():
                chess = chesses[player]
                current_state = self._gameboard_to_state()
                
                if player == 1:
                    current_state = self.swap_state(current_state)
                pos, mapping = self.get_optimal_move_from_UCT_model(current_state)
                all_states_trans[chess] += mapping,
                self.game.fast_place(chess,pos)
                player = 1-player
            if self.game.check_win_for('x'):
                x_win += 1
                for move in all_states_trans['x']:
                    self.model.update(self.model.nodes[move],2)
                for move in all_states_trans['o']:
                    self.model.update(self.model.nodes[move],0)
            elif self.game.check_win_for('o'):
                o_win += 1
                for move in all_states_trans['o']:
                    self.model.update(self.model.nodes[move],2)
                for move in all_states_trans['x']:
                    self.model.update(self.model.nodes[move],0)
            else:
                tie += 1
                for move in all_states_trans['x']:
                    self.model.update(self.model.nodes[move],1)
                for move in all_states_trans['o']:
                    self.model.update(self.model.nodes[move],1)

    def train(self):
        x_all, y_all = self.generate_training_data()
        size = x_all.shape[0]
        training_indices = set(np.random.choice(range(size), int(0.8*size), False))
        eval_indices = set(range(size)) - training_indices
        training_indices = list(training_indices)
        eval_indices = list(eval_indices)
        x_train, y_train = x_all[training_indices,:], y_all[training_indices]
        x_eval, y_eval = x_all[eval_indices,:], y_all[eval_indices]
        self.value_network.train(x_train,y_train,x_eval,y_eval, batch_size = len(training_indices)/20, epochs=1000) 

    def encode_state(self, s):
        s = map(lambda x: 0 if x==' ' else 1 if x=='x' else -1, list(s))
        return map(int, list(s))

    def generate_training_data(self):
        x_batch, y_batch = [], []
        x_batch_set = set([])
        i = 0
        while i < 4000:
            s = self.generate_random_game_state(next_player=random.randint(0,1), max_chess=self.game.n_rows)
            if s in x_batch_set:
                continue
            x_batch_set.add(s)
            i += 1
            # only consider the state has been played over certain times (init_n = 3)
        for s in x_batch_set:
            x = self.encode_state(s)
            s = self._find_mapping(s)
            if self.model.nodes[s].n < 5:
                continue
            y = self.model.nodes[s].win * 1.0 / self.model.nodes[s].n
            x_batch += x,
            y_batch += y,
        return np.asarray(x_batch), np.asarray(y_batch)

    def save_model(self, model_path):
        dill.dump([self.game_states,self.model],open(model_path,'w'))

    def load_model(self, model_path):
        self.game_states, self.model = dill.load(open(model_path, 'r'))

if __name__=='__main__':
    tictactoe = TicTacToe(3,3)
    robot = Robot(tictactoe)
    robot.sample(30000)
    robot.save_model('model.pkl')
    robot.train()
