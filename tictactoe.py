import itertools
import collections

class Cell(object):
    def __init__(self, status):
        self.default_status = status
        self.status = status

    def _place(self, chess):
        self.status = chess

    def _reset(self):
        self.status = self.default_status

    def _is_empty(self):
        return self.status == self.default_status

    def __str__(self):
        return self.status

class Board(object):
    def __init__(self, n_rows, n_cols, default_status):
        self.board = {}
        self.n_rows = n_rows
        self.n_cols = n_cols
        for (i,j) in itertools.product(range(n_rows), range(n_cols)):
            self.board[(i,j)] = Cell(default_status)
            
    def _place(self, chess, pos, _check_rule, call_back_func=None, *args):
        if _check_rule(chess, pos):
            self.board[pos]._place(chess)
            if call_back_func:
                call_back_func(*args)
        else:
            print "violate the rule, cannot make this move"

    def _reset(self):
        for pos in self.board:
            self.board[pos]._reset()

    def _get_row(self, i):
        '''
        input: row_num
        return: list
        '''
        arr = []
        for j in range(self.n_cols):
            arr += self.board[(i,j)],
        return arr

    def _get_col(self, j):
        '''
        input: col_num
        return: list
        '''
        arr = []
        for i in range(self.n_rows):
            arr += self.board[(i,j)],
        return arr

    def _get_diag(self, left_up, length, direction='normal'):
        arr = []
        dirs = {'normal':1,'anti':-1}
        for i in range(length):
            j = i * dirs[direction]
            if 0 <= left_up[0]+i < self.n_rows and 0 <= left_up[1]+j < self.n_cols:
                arr += self.board[(left_up[0]+i,left_up[1]+j)],
            else:
                break
        return arr

    def _is_full(self):
        return all([not cell._is_empty() for cell in self.board.values()])
    
    def __str__(self):
        res = ''
        for i in range(self.n_rows):
            if i:
                res += '\n'
                res += '-'*(2*self.n_cols-1)
                res += '\n'
            for j in range(self.n_cols):
                if j: res += "|"
                res += str(self.board[(i,j)])
        return res

class STATUS:
    TERMINAL, CONTINUE = range(2)

class TicTacToe(Board):
    
    def __init__(self, n=3, chesses=['o','x']):
        Board.__init__(self, n, n, ' ')
        self.status = STATUS.CONTINUE
        self.chesses = chesses

    def place(self, chess, pos, print_board=False, print_msg=True):
        self._place(chess, pos, self._check_rule, self.judge, print_board, print_msg)

    def fast_place(self, chess, pos):
        self.board[pos].status = chess
        
    def _check_rule(self, chess, pos):
        if self.board[pos]._is_empty():
            return True
        else:
            return False

    def check_win_for(self, chess):
        chess_pos = [pos for pos, cell in self.board.items() if cell.status == chess]
        row = [pos[0] for pos in chess_pos]
        col = [pos[1] for pos in chess_pos]
        if (row and collections.Counter(row).most_common(1)[0][1] == self.n_rows) or (col and collections.Counter(col).most_common(1)[0][1] == self.n_cols) or set((i,i) for i in range(self.n_rows)) - set(chess_pos) == set([]) or set((i,self.n_cols - i - 1) for i in range(self.n_rows)) - set(chess_pos) == set([]):
            return True
        return False

    def judge(self, print_board=False, print_msg=True):
        if print_board:
            print self
        flag = -1
        chess = None
        if self.check_win_for(self.chesses[0]):
            flag = 0
            chess = self.chesses[0]
        elif self.check_win_for(self.chesses[1]):
            flag = 0
            chess = self.chesses[1]
        elif self._is_full():
            flag = 1
            
        msg = ['{chess} win, game end'.format(chess=chess), 'tie']
        if flag != -1:
            self.end_game(msg[flag], print_msg)

    def end_game(self, msg, print_msg):
        self.status = STATUS.TERMINAL
        if print_msg:
            print msg

    def is_end(self):
        return self.status == STATUS.TERMINAL
        
    def reset(self):
        self.status = STATUS.CONTINUE
        self._reset()
        
if __name__ == '__main__':
    n = 3
    tictactoe = TicTacToe(n)
    player = 1
    candidates = set((i,j) for (i,j) in itertools.product(range(n), range(n)))
    import random
    while not tictactoe.is_end():
        chess = ['o','x'][player]
        pos = list(candidates)[random.randint(0,len(candidates)-1)]
        candidates.remove(pos)
        print '{chess} move {pos}'.format(chess=chess, pos=pos)        
        tictactoe.place(chess, pos)
        player = ~player
