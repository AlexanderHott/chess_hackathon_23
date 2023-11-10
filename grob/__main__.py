import logging
import chess

from bot import Bot
from grob import evaluator

from tests.random_bot import RandomBot

if __name__ == "__main__":
    grob_bot = Bot(depth=5, debug=True)
    random_bot = RandomBot()
    random_bot.board = grob_bot.board

    while not grob_bot.board.is_game_over():
        evaluator.reset_debug_vars()
        move = grob_bot.next_move()
        grob_bot.board.push_san(move)

        print(f"move: {move}, count: {evaluator.debug_search_count}, depth: {evaluator.debug_search_depth}")
        if grob_bot.board.is_game_over():
            break
        print("---\n\n")
        print(grob_bot.board)

        move = random_bot.next_move()
        grob_bot.board.push_san(move)
        if grob_bot.board.is_game_over():
            break
        print("---\n\n")
        print(grob_bot.board)

    print(f"Checkmate? {grob_bot.board.is_checkmate()} {'white' if grob_bot.board.turn == chess.BLACK else 'black' }")
    print(grob_bot.board)
