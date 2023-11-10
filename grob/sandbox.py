import chess

from grob import evaluator

board = chess.Board("rnbqkb1r/ppp1pppp/5n2/3p4/2QP4/8/PPP1PPPP/RNB1KBNR b KQkq - 0 1")

print(board)

print(evaluator.search(board, depth=3))

