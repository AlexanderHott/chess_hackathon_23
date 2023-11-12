import chess

from bot import Bot
from grob import evaluator

from tests.random_bot import RandomBot

if __name__ == "__main__":
    grob_bot = Bot(fen="r1bq1rk1/ppp1ppbp/2np1np1/8/3P1B2/4PN1P/PPP1BPP1/RN1Q1RK1 b - - 2 7", depth=3, debug=True)
    #grob_bot.zobrist_numbers = None
    grob2 = Bot(depth=3, debug=True)
    #grob2.zobrist_numbers = None

    grob2.board = grob_bot.board

    while not grob_bot.board.is_game_over():
        evaluator.reset_debug_vars()
        move = grob_bot.next_move()
        grob_bot.board.push_san(move)
        print(
            f"move 1: {move}, count: {evaluator.debug_search_count}, depth: {evaluator.debug_search_depth}, tt hits: {evaluator.debug_tt_cache_hits}"
        )
        if grob_bot.board.is_game_over():
            break

        evaluator.reset_debug_vars()
        move = grob2.next_move()
        grob_bot.board.push_san(move)
        print(
            f"move 2: {move}, count: {evaluator.debug_search_count}, depth: {evaluator.debug_search_depth}, tt hits: {evaluator.debug_tt_cache_hits}"
        )
        if grob_bot.board.is_game_over():
            break

    print(
        f"Checkmate? {grob_bot.board.is_checkmate()} {'white' if grob_bot.board.turn == chess.BLACK else 'black' }"
    )
    print(grob_bot.board)
