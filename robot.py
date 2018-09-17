from model import UCT_Model
from tictactoe import TicTacToe
import random
import re
class Robot(object):
    def __init__(self, game):
        self.model = UCT_Model()
        self.game = game
        self.game_states = set([])
        
    def generate_random_game_state(self):
        n = 0
        i = random.randint(0,1)
        x_num, o_num = n, n+i
        state = 'x' * x_num + 'o' * o_num +' ' * (self.game.n_rows * self.game.n_cols - x_num - o_num)
        return ''.join(random.sample(state,len(state)))

    # this function is to compress board state as the game is symetric
    def _find_mapping(self,s):
        transforms = {'630741852','876543210','258147036','210543876','678345012','852741630','036147258'}
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
                self.game.fast_place(s[i*3+j],(i,j))
    
    def get_optimal_move(self, s):
        indices = [m.start() for m in re.finditer(' ', s)]
        moves = {(self._find_mapping(s[:i]+'x'+s[i+1:])):i for i in indices}
        best_move = self.model.evaluate_and_select(moves.keys())
        i = moves[best_move]
        return (i/3, i%3), best_move

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
    
    def train(self, n):
        for i in xrange(n):
            if i == 0:
                print 'start training...'
            if i % 100 == 0:
                print '{n} trains'.format(n=i)
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
                if player == 1:
                    current_state = self.swap_state(current_state)
                pos, move = self.get_optimal_move(current_state)
                all_states_trans[chess] += move,
                self.game.place(chess,pos,print_msg=False)
                player = ~player
            
            if self.game.check_win_for('x'):
                for move in all_states_trans['x']:
                    self.model.update(self.model.nodes[move],1)
                for move in all_states_trans['o']:
                    self.model.update(self.model.nodes[move],0)
            elif self.game.check_win_for('o'):
                for move in all_states_trans['o']:
                    self.model.update(self.model.nodes[move],1)
                for move in all_states_trans['x']:
                    self.model.update(self.model.nodes[move],0)
            else:
                for move in all_states_trans['x']:
                    self.model.update(self.model.nodes[move],0)
                for move in all_states_trans['o']:
                    self.model.update(self.model.nodes[move],0)


if __name__=='__main__':
    tictactoe = TicTacToe()
    robot = Robot(tictactoe)
    robot.train(100000)

    chesses = ['x','o']
    for _ in range(3):
        tictactoe.reset()
        player = 0
        while not tictactoe.is_end():
            chess = chesses[player]
            if player:
                pos = raw_input("human play ")
                pos = tuple(int(x.strip()) for x in pos.split(','))
            else:
                print "robot play"
                pos, _ = robot.get_optimal_move(robot.swap_state(robot._gameboard_to_state()))
            print "{chess} move {pos}".format(chess=chess,pos=pos)
            tictactoe.place(chess, pos, print_board=True)
            player = ~player
