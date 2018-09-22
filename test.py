from tictactoe import TicTacToe
from robot import Robot
def test():
    tictactoe = TicTacToe(6,4)
    robot = Robot(tictactoe)
    robot.load_model('model.pkl')
    chesses = ['x', 'o']
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
                pos,_ = robot.get_optimal_move_from_UCT_model(robot.swap_state(robot._gameboard_to_state()))
            print "{chess} move {pos}".format(chess=chess,pos=pos)
            tictactoe.place(chess, pos, print_board=True)
            player = ~player

if __name__ == '__main__':
    test()
