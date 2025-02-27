"""
The Brandeis Quant Club ML/AI Competition (November 2023)

Author: @Ephraim Zimmerman
Email: quants@brandeis.edu
Website: brandeisquantclub.com; quants.devpost.com

Description:

For any technical issues or questions please feel free to reach out to
the "on-call" hackathon support member via email at quants@brandeis.edu

Website/GitHub Repository:
You can find the latest updates, documentation, and additional resources for this project on the
official website or GitHub repository: https://github.com/EphraimJZimmerman/chess_hackathon_23

License:
This code is open-source and released under the MIT License. See the LICENSE file for details.
"""
import os
import random

import chess
import time
from collections.abc import Iterator
from contextlib import contextmanager

from grob import evaluator
import test_bot


@contextmanager
def game_manager() -> Iterator[None]:
    """Creates context for game."""

    print("===== GAME STARTED =====")
    ping: float = time.perf_counter()
    try:
        # DO NOT EDIT. This will be replaced w/ judging context manager.
        yield
    finally:
        pong: float = time.perf_counter()
        total = pong - ping
        print(f"Total game time = {total:.3f} seconds")
    print("===== GAME ENDED =====")


class Bot:
    def __init__(self, fen=None, time_per_turn=1, use_square_scores=True, debug=False):
        self.board = chess.Board(
            fen if fen else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        self.time_per_turn = time_per_turn
        self.use_square_scores = use_square_scores
        self.debug = debug
        possible_paths = [
            "../../data/opening_book.txt",
            "../data/opening_book.txt",
            "data/opening_book.txt",
        ]
        for path in possible_paths:
            try:
                self.opening_book = evaluator.get_opening_book(path)
                break
            except FileNotFoundError:
                continue
        self.transposition_table = evaluator.TranspositionTable(max_size=3_000_000)

    def check_move_is_legal(self, initial_position, new_position) -> bool:
        """
        To check if, from an initial position, the new position is valid.

        Args:
            initial_position (str): The starting position given chess notation.
            new_position (str): The new position given chess notation.

        Returns:
            bool: If this move is legal
        """

        return (
            chess.Move.from_uci(initial_position + new_position)
            in self.board.legal_moves
        )

    def next_move(self) -> str:
        """
        The main call and response loop for playing a game of chess.

        Returns:
            str: The current location and the next move.
        """

        # Assume that you are playing an arbitrary game. This function, which is
        # the core "brain" of the bot, should return the next move in any circumstance.

        _, move = evaluator.iterative_deepening_search(
            self.board,
            search_time=self.time_per_turn,
            opening_book=self.opening_book,
            transposition_table=self.transposition_table,
            debug_counts=self.debug,
        )
        if self.transposition_table is not None:
            self.board.push(move)
            if (tup := evaluator.get_representation_tuple(self.board)) in self.transposition_table:
                self.transposition_table.pop(tup)  # prevents repetition glitches
            self.board.pop()
        return str(move)


# Add promotion stuff

if __name__ == "__main__":
    chess_bot = Bot()  # you can enter a FEN here, like Bot("...")
    with game_manager():
        """

        Feel free to make any adjustments as you see fit. The desired outcome
        is to generate the next best move, regardless of whether the bot
        is controlling the white or black pieces. The code snippet below
        serves as a useful testing framework from which you can begin
        developing your strategy.

        """

        playing = True

        while playing:
            if chess_bot.board.turn:
                chess_bot.board.push_san(
                    test_bot.get_move(chess_bot.board, best_move=False)
                )
            else:
                chess_bot.board.push_san(chess_bot.next_move())
            print(chess_bot.board, end="\n\n")

            if chess_bot.board.is_game_over():
                if chess_bot.board.is_stalemate():
                    print("Is stalemate")
                elif chess_bot.board.is_insufficient_material():
                    print("Is insufficient material")

                # EX: Outcome(termination=<Termination.CHECKMATE: 1>, winner=True)
                print(chess_bot.board.outcome())

                playing = False
