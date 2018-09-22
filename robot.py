from model import UCT_Model
from tictactoe import TicTacToe
import random
import re
import numpy as np
from value_network import ValueNetwork
import dill
class Robot(object):
    def __init__(self, game):
        self.model = UCT_Model()
        self.game = game
        self.game_states = set([])
        INPUT_SHAPE = 36
        HIDDEN_DIMS = (100,100,100)
        OUTPUT_SHAPE = 1
        self.value_network = ValueNetwork(INPUT_SHAPE, HIDDEN_DIMS, OUTPUT_SHAPE)
        self._cache_mapping_set = self._create_mapping_state_set()
        
    def load_value_network(self, model_path):
        self.value_network.load_model(model_path)
        
    def generate_random_game_state(self, is_training=False):
        n = random.randint(0,self.game.n_rows)
        i = 0
        x_num, o_num = n+i, n
        state = 'x' * x_num + 'o' * o_num +' ' * (self.game.n_rows * self.game.n_cols - x_num - o_num)
        return ''.join(random.sample(state,len(state)))

    # this function is to compress board state as the game is symetric

    def _statestring_to_statematrix(self, s):
        mat = []
        for i in range(self.game.n_rows):
            mat += list(s[i*self.game.n_cols:(i+1)*self.game.n_cols]),
        return mat
    
    def _rotate_clockwise_state(self, s):
        mat = self._statestring_to_statematrix(s)
        new_mat = []
        for row in list(zip(*mat)):
            new_mat += ''.join(row[::-1]),

        return ''.join(new_mat)

    def _flip_state(self, s):
        mat = self._statestring_to_statematrix(s)
        new_mat = []
        for row in mat[::-1]:
            new_mat += ''.join(row)
        return ''.join(new_mat)

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
        s = ''.join([str(i) for i in range(self.game.n_rows*self.game.n_cols)])
        dfs(s, s_set)
        return s_set
                    
    def _find_mapping(self,s):
        transforms = self._cache_mapping_set
        for trans in transforms:
            tmp_state = ''.join([s[int(i)] for i in trans])
            if tmp_state in self.game_states:
                s = tmp_state
                break
        self.game_states.add(s)
        return s

    def _gameboard_to_state(self):
        s = ''
        for i in range(self.game.n_rows):
            for j in range(self.game.n_cols):
                s += str(self.game.board[(i,j)])
        return s

    def _state_to_gameboard(self, s):
        for i in range(self.game.n_rows):
            for j in range(self.game.n_cols):
                self.game.fast_place(s[i*self.game.n_rows+j],(i,j))

    def get_optimal_move_from_value_network(self, s):
        indices = [m.start() for m in re.finditer(' ', s)]
        moves = {self.value_network.predict(np.asarray([self.encode_state(self._find_mapping(s[:i]+'x'+s[i+1:]))]))[0][0]:i for i in indices}
        i = moves[max(moves.keys())]
        return (i/self.game.n_rows, i%self.game.n_cols)
        
    def get_optimal_move_from_UCT_model(self, s):
        indices = [m.start() for m in re.finditer(' ', s)]
        moves = {(self._find_mapping(s[:i]+'x'+s[i+1:])):i for i in indices}
        best_move = self.model.evaluate_and_select(moves.keys())
        i = moves[best_move]
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
            player = 0
            while new or self.game.is_end():
                self.game.reset()
                new = False
                init_state = self.generate_random_game_state()
                self._state_to_gameboard(init_state)
                self.game.judge(print_msg=False)
            init_state = self._find_mapping(init_state)
            all_states_trans = {'o':[],'x':[]}
            while not self.game.is_end():
                chess = chesses[player]
                current_state = self._gameboard_to_state()
                if player == -1:
                    current_state = self.swap_state(current_state)
                pos, move = self.get_optimal_move_from_UCT_model(current_state)
                all_states_trans[chess] += move,
                self.game.place(chess,pos,print_msg=False)
                player = ~player
            if self.game.check_win_for('x'):
                x_win += 1
                for move in all_states_trans['x']:
                    self.model.update(self.model.nodes[move],1)
                for move in all_states_trans['o']:
                    self.model.update(self.model.nodes[move],-1)
            elif self.game.check_win_for('o'):
                o_win += 1
                for move in all_states_trans['o']:
                    self.model.update(self.model.nodes[move],1.2)
                for move in all_states_trans['x']:
                    self.model.update(self.model.nodes[move],-1)
            else:
                tie += 1
                for move in all_states_trans['x']:
                    self.model.update(self.model.nodes[move],0)
                for move in all_states_trans['o']:
                    self.model.update(self.model.nodes[move],0.2)

    def train(self):
        x_all, y_all = self.generate_training_data()
        size = x_all.shape[0]
        training_indices = set(np.random.choice(range(size), int(0.8*size), False))
        eval_indices = set(range(size)) - training_indices
        training_indices = list(training_indices)
        eval_indices = list(eval_indices)
        import pdb; pdb.set_trace()
        x_train, y_train = x_all[training_indices,:], y_all[training_indices]
        x_eval, y_eval = x_all[eval_indices,:], y_all[eval_indices]
        self.value_network.train(x_train,y_train,x_eval,y_eval, batch_size = len(training_indices)/100, epochs=200) 

    def encode_state(self, s):
        s = s.replace(' ','0')
        s = s.replace('x','1')
        s = s.replace('o','2')
        return map(int, list(s))

    def decode_state(self, arr):
        s = ''.join(arr)
        s = s.replace('0', ' ')
        s = s.replace('1', 'x')
        s = s.replace('2', 'o')
        return s

    def generate_training_data(self):
        x_batch, y_batch = [], []
        x_batch_set = set([])
        i = 0
        while i < 1000:
            s = self.generate_random_game_state(is_training=True)
            if s in x_batch_set:
                continue
            x_batch_set.add(s)
            i += 1
            # only consider the state has been played over certain times (init_n = 3)
        for s in x_batch_set:
            x = self.encode_state(s)
            s = self._find_mapping(s)
            if self.model.nodes[s].n < 10:
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
    robot.sample(2000)
    robot.save_model('model.pkl')
    robot.train()
