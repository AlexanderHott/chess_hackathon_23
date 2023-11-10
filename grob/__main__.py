import logging
import chess

from bot import Bot
from grob import evaluator

from tests.random_bot import RandomBot

if __name__ == "__main__":
    # grob_bot = Bot(depth=5, debug=True)
    # random_bot = RandomBot()
    # random_bot.board = grob_bot.board
    #
    # while not grob_bot.board.is_game_over():
    #     evaluator.reset_debug_vars()
    #     move = grob_bot.next_move()
    #     grob_bot.board.push_san(move)
    #
    #     print(f"move: {move}, count: {evaluator.debug_search_count}, depth: {evaluator.debug_search_depth}")
    #     if grob_bot.board.is_game_over():
    #         break
    #     print("---\n\n")
    #     print(grob_bot.board)
    #
    #     move = random_bot.next_move()
    #     grob_bot.board.push_san(move)
    #     if grob_bot.board.is_game_over():
    #         break
    #     print("---\n\n")
    #     print(grob_bot.board)
    #
    # print(f"Checkmate? {grob_bot.board.is_checkmate()} {'white' if grob_bot.board.turn == chess.BLACK else 'black' }")
    # print(grob_bot.board)


    import chess

    from grob import evaluator

    board = chess.Board("rnbqkb1r/ppp1pppp/5n2/3p4/2QP4/8/PPP1PPPP/RNB1KBNR b KQkq - 0 1")

    print(board)

    print(evaluator.search(board, depth=3))

