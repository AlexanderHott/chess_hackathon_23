import chess

from bot import Bot
from grob import evaluator

from tests.random_bot import RandomBot

if __name__ == "__main__":
    grob_bot = Bot(fen="rnq2kr1/p6p/2p2Bp1/4Pp1b/2QP4/8/PP3P1P/R4RK1 b - - 0 1", depth=4, debug=True)
    random_bot = RandomBot()
    random_bot.board = grob_bot.board

    while not grob_bot.board.is_game_over():
        evaluator.reset_debug_vars()
        move = grob_bot.next_move()
        grob_bot.board.push_san(move)

        print(
            f"move: {move}, count: {evaluator.debug_search_count}, depth: {evaluator.debug_search_depth}, tt hits: {evaluator.debug_tt_cache_hits}"
        )
        print(f"{len(grob_bot.transition_table)=}")
        if grob_bot.board.is_game_over():
            break

        move = random_bot.next_move()
        grob_bot.board.push_san(move)
        if grob_bot.board.is_game_over():
            break

    print(
        f"Checkmate? {grob_bot.board.is_checkmate()} {'white' if grob_bot.board.turn == chess.BLACK else 'black' }"
    )
    print(grob_bot.board)
