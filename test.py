from tictactoe import TicTacToe
from robot import Robot
import random
def test():
    tictactoe = TicTacToe(3,3,play_on_line=False)
    robot = Robot(tictactoe)
    robot.load_model('model.pkl')
    robot.load_value_network('value_network.m')
    chesses = ['x', 'o']
    for _ in range(3):
        tictactoe.reset()
        player = 0
        human_player = random.randint(0,1)
        while not tictactoe.is_end():
            chess = chesses[player]
            if player:
                state = robot.swap_state(robot._gameboard_to_state())
            else:
                state = robot._gameboard_to_state()
            if player == human_player:
                pos = raw_input("human play ")
                pos = tuple(int(x.strip()) for x in pos.split(','))
            else:
                print "robot play"
                pos = robot.get_optimal_move_from_UCT_model(state,is_training=False)
            print "{chess} move {pos}".format(chess=chess,pos=pos)
            tictactoe.place(chess, pos, print_board=True)
            player = 1-player

if __name__ == '__main__':
    test()
