import random

import chess

import bot


class RandomBot(bot.Bot):
    def next_move(self, update_zobrist_hash: bool = True) -> str:
        return str(random.choice([_ for _ in self.board.legal_moves]))
