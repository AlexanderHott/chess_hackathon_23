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

import itertools
import random
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import chain

import chess
from chess import pgn

import test_bot


PIECE_VALUES = {
    chess.PAWN: 10,
    chess.KNIGHT: 30,
    chess.BISHOP: 31,
    chess.ROOK: 50,
    chess.QUEEN: 90,
    chess.KING: 0,
}

ENDGAME_WEIGHT_CONTROL = (
    1 / 29
)  # such that the weight hits 1.0 when there are 3 pieces remaining, (32 - 3) / 29 = 1.0
CORNER_KING_MATERIAL_DIFFERENCE_REQ = PIECE_VALUES[chess.PAWN] * 2
CORNER_KING_ENDGAME_WEIGHT_REQ = 0.5

SQUARE_SCORE_WEIGHT = 0.1
ENDGAME_KING_PUSH_WEIGHT = 2

WHITE_PIECE_SQUARE_TABLES: dict[chess.PieceType, list[int]] = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
          0,  0,  0,  5,  5,  0,  0,  0,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          5, 10, 10, 10, 10, 10, 10,  5,
          0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -10,  5,  5,  5,  5,  5,  0,-10,
          0,  0,  5,  5,  5,  5,  0, -5,
         -5,  0,  5,  5,  5,  5,  0, -5,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    chess.KING: [
         20, 30, 10,  0,  0, 10, 30, 20,
         20, 20,  0,  0,  0,  0, 20, 20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
    ]
}

ENDGAME_WHITE_PIECE_SQUARE_TABLES: dict[chess.PieceType, list[int]] = {
    chess.KING: [
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    ],
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10,
        20, 20, 20, 20, 20, 20, 20, 20,
        30, 30, 30, 30, 30, 30, 30, 30,
        50, 50, 50, 50, 50, 50, 50, 50,
        80, 80, 80, 80, 80, 80, 80, 80,
        0, 0, 0, 0, 0, 0, 0, 0
    ]
}

BLACK_PIECE_SQUARE_TABLES: dict[chess.PieceType, list[int]] = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
          0,  0,  0,  0,  0,  0,  0,  0,
          5, 10, 10, 10, 10, 10, 10,  5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          0,  0,  0,  5,  5,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ]
}

ENDGAME_BLACK_PIECE_SQUARE_TABLES: dict[chess.PieceType, list[int]] = {
    chess.KING: [
        -50, -30, -30, -30, -30, -30, -30, -50,
        -30, -30, 0, 0, 0, 0, -30, -30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30, -20, -10, 0, 0, -10, -20, -30,
        -50, -40, -30, -20, -20, -30, -40, -50,
    ],
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        80, 80, 80, 80, 80, 80, 80, 80,
        50, 50, 50, 50, 50, 50, 50, 50,
        30, 30, 30, 30, 30, 30, 30, 30,
        20, 20, 20, 20, 20, 20, 20, 20,
        10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10,
         0,  0,  0,  0,  0,  0,  0,  0
    ]
}


# INF = float("inf")
INF = 99_999_999

TableEntryType = int
TABLE_ENTRY_TYPE = [EXACT, LOWER_BOUND, UPPER_BOUND] = range(3)


class TranspositionTable(OrderedDict[tuple[int, ...], tuple[int | None, float, TableEntryType]]):
    def __init__(self, max_size=5, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: tuple[int, ...], value: tuple[int | None, float, TableEntryType]):
        if key not in self:
            if len(self) == self.max_size:
                self.popitem(last=False)
        super().__setitem__(key, value)


def get_representation_tuple(board: chess.Board):
    return (*(board.pieces_mask(p_type, color)
              for p_type, color in itertools.product(chess.PIECE_TYPES, chess.COLORS)), board.turn,
            board.has_kingside_castling_rights(chess.WHITE), board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.BLACK))


def is_entry_applicable(cached_eval: float, alpha: float, beta: float, entry_type: TableEntryType) -> bool:
    """
    returns: whether a stored entry can be used, given the current alpha-beta context
    """
    return entry_type == EXACT or (entry_type == UPPER_BOUND and cached_eval <= alpha) or \
        (entry_type == LOWER_BOUND and cached_eval >= beta)


debug_search_count = 0
debug_search_depth = 0
debug_tt_cache_hits = 0


def reset_debug_vars() -> None:
    """
    This resets the global debug variables
    """
    global debug_search_count
    global debug_search_depth
    global debug_tt_cache_hits
    debug_search_count = 0
    debug_search_depth = 0
    debug_tt_cache_hits = 0


def get_opening_book(book_file: str) -> dict[str, dict[str, int]]:
    """
    returns: an opening book dict, where a fen entry corresponds to a dict of moves and their corresponding weights
    """
    opening_book = {}
    with open(book_file, "r") as lines:
        fen = ""
        for line in lines:
            if line.startswith("pos "):  # a start of a position in the file
                fen = line[4:-1]
                opening_book[fen] = {}
            else:
                move, weight = line.split(" ")
                opening_book[fen][move] = int(weight)
    return opening_book


def material_score(board: chess.Board, color: chess.Color) -> int:
    """
    Returns: the piece material score for a color
    """
    material_value = 0
    for piece_type in PIECE_VALUES:
        material_value += (
            board.pieces_mask(piece_type, color).bit_count() * PIECE_VALUES[piece_type]
        )
    return material_value


def get_piece_square_bonus(
    square: chess.Square, piece: chess.PieceType, color: chess.Color, endgame: bool = False
) -> int:
    if color == chess.WHITE:
        return ENDGAME_WHITE_PIECE_SQUARE_TABLES[piece][square] if endgame else WHITE_PIECE_SQUARE_TABLES[piece][square]
    else:
        return ENDGAME_BLACK_PIECE_SQUARE_TABLES[piece][square] if endgame else BLACK_PIECE_SQUARE_TABLES[piece][square]


def get_square_scores(board: chess.Board, color: chess.Color) -> int:
    """
    Returns: the square bonus scores for each piece
    """
    total_square_bonus = 0
    for piece_type in chess.PIECE_TYPES:
        # is there a more efficient way of doing this using the bit mask directly?
        for square in board.pieces(piece_type, color):
            if piece_type == chess.KING or piece_type == chess.PAWN:
                endgame_weight = (32 - board.occupied.bit_count()) * ENDGAME_WEIGHT_CONTROL
                total_square_bonus += (1 - endgame_weight) * get_piece_square_bonus(square, piece_type, color) + \
                    endgame_weight * get_piece_square_bonus(square, piece_type, color, endgame=True)
            else:
                total_square_bonus += get_piece_square_bonus(square, piece_type, color)
    return total_square_bonus


def endgame_corner_king(
    board: chess.Board, color: chess.Color, my_material: float, enemy_material: float
) -> float:
    """
    Returns: the proximity of kings in the board, used to corner the king in endgames
    """
    endgame_weight = (
        32 - board.occupied.bit_count()
    ) * ENDGAME_WEIGHT_CONTROL
    evaluation = 0
    if (
        my_material > enemy_material + CORNER_KING_MATERIAL_DIFFERENCE_REQ
        and endgame_weight > CORNER_KING_ENDGAME_WEIGHT_REQ
    ):
        # reward distance from center
        enemy = board.king(not color)
        if enemy is None:
            raise ValueError("King is None")
        enemy_rank, enemy_file = chess.square_rank(enemy), chess.square_file(enemy)
        file_distance = max(3 - enemy_file, enemy_file - 4)
        rank_distance = max(3 - enemy_rank, enemy_rank - 4)
        evaluation += file_distance + rank_distance

        # reward closer kings
        friendly = board.king(color)
        if friendly is None:
            raise ValueError("King is None")
        friendly_rank, friendly_file = chess.square_rank(friendly), chess.square_file(
            friendly
        )
        distance = abs(friendly_rank - enemy_rank) + abs(friendly_file - enemy_file)
        evaluation += 14 - distance
    return evaluation * endgame_weight


def evaluate(board: chess.Board, use_square_scores: bool = True) -> float:
    """
    Returns: board evaluation
    """
    white_sum = 0
    black_sum = 0

    white_sum += (white_material := material_score(board, chess.WHITE))
    black_sum += (black_material := material_score(board, chess.BLACK))

    if use_square_scores:
        white_sum += get_square_scores(board, chess.WHITE) * SQUARE_SCORE_WEIGHT
        black_sum += get_square_scores(board, chess.BLACK) * SQUARE_SCORE_WEIGHT

    white_sum += endgame_corner_king(board, chess.WHITE, white_material, black_material) * ENDGAME_KING_PUSH_WEIGHT
    black_sum += endgame_corner_king(board, chess.BLACK, black_material, white_material) * ENDGAME_KING_PUSH_WEIGHT

    evaluation = white_sum - black_sum
    if board.turn == chess.BLACK:
        evaluation = -evaluation
    return evaluation


def guess_move_evaluation(board: chess.Board, move: chess.Move) -> int:
    """
    Returns: guesses the evaluation of a move for move ordering
    """
    guess = 0
    move_piece_type = board.piece_type_at(move.from_square)
    capture_piece_type = board.piece_type_at(move.to_square)

    # prioritize easy captures
    if capture_piece_type is not None and move_piece_type is not None:
        # smallest attacker, biggest capture
        guess += 10 * PIECE_VALUES[capture_piece_type] - PIECE_VALUES[move_piece_type]

    # prioritize promotions
    if move.promotion is not None:
        guess += PIECE_VALUES[move.promotion]

    # prioritize avoiding pawns
    opposite_color = not board.turn
    attacking_pawns = board.attackers_mask(
        opposite_color, move.to_square
    ) & board.pieces_mask(chess.PAWN, opposite_color)
    if attacking_pawns != 0 and move_piece_type is not None:
        guess -= PIECE_VALUES[move_piece_type]

    return guess


def order_moves(
    board: chess.Board, moves: chess.LegalMoveGenerator, priority_move: chess.Move | None = None
) -> list[chess.Move]:
    """
    Returns: sorts a list of moves in place according to guess_move_evaluation
    """
    return sorted(moves, key=lambda m: INF if priority_move == m else guess_move_evaluation(board, m), reverse=True)


def calculate_search_extensions(board: chess.Board, move: chess.Move) -> int:
    extension = 0
    if board.is_check():
        extension += 1
    piece = board.piece_at(move.to_square)
    rank = chess.square_rank(move.to_square)
    if piece.piece_type == chess.PAWN and ((piece.color == chess.WHITE and rank == 6) or
                                           (piece.color == chess.BLACK and rank == 1)):
        extension += 1
    return extension


def search_all_captures(
    board: chess.Board,
    alpha: float,
    beta: float,
    levels_deep: int = 0,
    transposition_table: TranspositionTable | None = None,
    search_checks: bool = True,
    use_square_scores: bool = True,
    debug_counts: bool = False,
) -> tuple[float, chess.Move | None]:
    """
    Returns: an alpha-beta evaluation that only considers capture moves (and check moves)
    """
    if debug_counts:
        global debug_search_count
        global debug_search_depth
        debug_search_count += 1
        # debug_search_depth = max(debug_search_depth, levels_deep)

    tuple_representation = None
    if transposition_table is not None:
        tuple_representation = get_representation_tuple(board)
        if tuple_representation in transposition_table:
            cached_depth, cached_eval, entry_type = transposition_table[tuple_representation]
            if cached_depth is None and is_entry_applicable(cached_eval, alpha, beta, entry_type):
                if debug_counts:
                    global debug_tt_cache_hits
                    debug_tt_cache_hits += 1
                return cached_eval, None

    evaluation = evaluate(board, use_square_scores=use_square_scores)
    if evaluation >= beta:
        if transposition_table is not None:
            transposition_table[tuple_representation] = (None, beta, LOWER_BOUND)
        return beta, None
    alpha = max(alpha, evaluation)

    # unclear if this is the most efficient way of generating these moves, could also use board.gives_check
    capture_moves = board.generate_legal_moves(
        chess.BB_ALL, board.occupied_co[not board.turn]
    )
    important_moves = capture_moves
    if search_checks:
        king = board.king(not board.turn)
        if king is None:
            raise ValueError("Board is invalid")
        # mask logic followed according to board.attacks_mask()
        pawn_check_moves = board.generate_legal_moves(
            board.pieces_mask(chess.PAWN, board.turn),
            chess.BB_PAWN_ATTACKS[not board.turn][king],
        )
        rook_checks_mask = (
            chess.BB_RANK_ATTACKS[king][chess.BB_RANK_MASKS[king] & board.occupied]
            | chess.BB_FILE_ATTACKS[king][chess.BB_FILE_MASKS[king] & board.occupied]
        )
        rook_check_moves = board.generate_legal_moves(
            board.pieces_mask(chess.ROOK, board.turn), rook_checks_mask
        )
        knight_check_moves = board.generate_legal_moves(
            board.pieces_mask(chess.KNIGHT, board.turn), chess.BB_KNIGHT_ATTACKS[king]
        )
        bishop_checks_mask = chess.BB_DIAG_ATTACKS[king][
            chess.BB_DIAG_MASKS[king] & board.occupied
        ]
        bishop_check_moves = board.generate_legal_moves(
            board.pieces_mask(chess.BISHOP, board.turn), bishop_checks_mask
        )
        queen_check_moves = board.generate_legal_moves(
            board.pieces_mask(chess.QUEEN, board.turn),
            rook_checks_mask | bishop_checks_mask,
        )
        important_moves = chain(
            capture_moves,
            pawn_check_moves,
            rook_check_moves,
            knight_check_moves,
            bishop_check_moves,
            queen_check_moves,
        )

    important_moves = sorted(
        important_moves, key=lambda m: guess_move_evaluation(board, m), reverse=True
    )

    best_move = None
    for move in important_moves:
        board.push(move)
        evaluation = -search_all_captures(
            board,
            -beta,
            -alpha,
            levels_deep=levels_deep + 1,
            search_checks=search_checks,
            debug_counts=debug_counts,
        )[0]
        board.pop()
        if evaluation >= beta:
            if transposition_table is not None:
                transposition_table[tuple_representation] = (None, beta, LOWER_BOUND)
            return beta, None
        if evaluation > alpha:  # must not be >=
            alpha = evaluation
            best_move = move

    if transposition_table is not None:
        transposition_table[tuple_representation] = (None, alpha, UPPER_BOUND if best_move is None else EXACT)
    return alpha, best_move


def search(
    board: chess.Board,
    depth: int,
    alpha: float = -INF,
    beta: float = INF,
    levels_deep: int = 0,
    total_extensions: int = 0,
    transposition_table: TranspositionTable | None = None,
    _use_transposition_table: bool = False,
    opening_book: dict[str, dict[str, int]] | None = None,
    end_time: float | None = None,
    priority_move: chess.Move | None = None,
    using_opening_book: bool = True,
    use_square_scores: bool = True,
    guess_move_order: bool = True,
    search_captures: bool = True,
    search_checks: bool = True,
    debug_counts: bool = False,
) -> tuple[float, chess.Move | None]:
    """
    Args:
        board:
        depth: depth to run a full search on
        alpha: see alpha-beta pruning
        beta: see alpha-beta pruning
        levels_deep: how many levels deep the current function call is
        total_extensions: how many levels the search has been extended
        transposition_table: a table of already searched positions and their evaluations
        opening_book: an opening book
        end_time: the time to break at
        priority_move: best move of previous search iteration
        using_opening_book: whether the book is still being used
        use_square_scores: whether to use square scores
        guess_move_order: whether to sort moves according to an initial guess evaluation
        search_captures: whether to search all captures after depth limit is reached
        search_checks: whether to search all checks after depth limit is reached
        debug_counts: whether to update global count variables
        _use_transposition_table: whether to use the transposition_table. First search shouldn't
    Returns: the evaluation of the current position, along with the best move if the depth has not been reached
    """
    if end_time is not None and time.time() > end_time:
        raise TimeoutError("Out of time!")

    if using_opening_book:
        if opening_book is not None:
            fen = board.fen()[:-4]
            if fen in opening_book:
                # remove the moves portion
                opening_moves: dict[str, int] = opening_book[fen]
                move = random.choices(
                    list(opening_moves.keys()), list(opening_moves.values())
                )[0]
                return INF, chess.Move.from_uci(move)
        else:
            using_opening_book = False

    if debug_counts:
        global debug_search_count
        global debug_search_depth
        debug_search_count += 1
        debug_search_depth = max(debug_search_depth, levels_deep)

    if board.is_game_over():
        if board.is_checkmate():
            return -INF + levels_deep, None  # current player has lost
        else:
            return 0, None  # game is a draw

    tuple_representation = None
    if transposition_table is not None and _use_transposition_table:
        tuple_representation = get_representation_tuple(board)
        if tuple_representation in transposition_table:
            cached_depth, cached_eval, entry_type = transposition_table[tuple_representation]
            if cached_depth is not None and depth <= cached_depth and \
                    is_entry_applicable(cached_eval, alpha, beta, entry_type):
                if debug_counts:
                    global debug_tt_cache_hits
                    debug_tt_cache_hits += 1
                return cached_eval, None

    if depth == 0:
        if search_captures:
            return search_all_captures(
                board,
                alpha,
                beta,
                levels_deep=levels_deep,
                transposition_table=transposition_table,
                search_checks=search_checks,
                use_square_scores=use_square_scores,
                debug_counts=debug_counts,
            )
        else:
            instant_evaluation = evaluate(board, use_square_scores=use_square_scores)
            if transposition_table is not None:
                transposition_table[tuple_representation] = (depth, instant_evaluation, EXACT)
            return instant_evaluation, None

    moves = board.legal_moves
    if guess_move_order:
        moves = order_moves(board, moves, priority_move=priority_move)
    best_move = None
    _debug_move_evals = {}
    for move in moves:
        board.push(move)
        try:
            extension = max(0, min(8 - total_extensions, calculate_search_extensions(board, move)))
            evaluation = -search(
                board,
                depth - 1 + extension,
                -beta,
                -alpha,
                levels_deep=levels_deep + 1,
                total_extensions=total_extensions + extension,
                transposition_table=transposition_table,
                opening_book=opening_book,
                end_time=end_time,
                priority_move=priority_move,
                using_opening_book=using_opening_book,
                use_square_scores=use_square_scores,
                guess_move_order=guess_move_order,
                search_captures=search_captures,
                search_checks=search_checks,
                debug_counts=debug_counts,
                _use_transposition_table=True,
            )[0]
        except TimeoutError:
            board.pop()
            raise TimeoutError("Ran out of time!")
        board.pop()
        _debug_move_evals[move] = evaluation
        # logging.debug(f"Eval for {move}: {evaluation}")
        if evaluation >= beta != INF and transposition_table is not None:  # prune the tree
            transposition_table[tuple_representation] = (depth, beta, LOWER_BOUND)
            return beta, None
        if evaluation > alpha or evaluation == -INF:
            alpha = evaluation
            best_move = move

    if transposition_table is not None:
        transposition_table[tuple_representation] = (depth, alpha, UPPER_BOUND if best_move is None else EXACT)
    return alpha, best_move


def iterative_deepening_search(board: chess.Board, search_time: float,
                               transposition_table: TranspositionTable | None = None,
                               opening_book: dict[str, dict[str, int]] | None = None,
                               debug_counts: bool = False) -> tuple[float, chess.Move | None]:
    end_time = time.time() + search_time
    best_eval, best_move = -INF, None
    try:
        iterative_depth = 1
        while True:
            best_eval, best_move = search(board, iterative_depth, transposition_table=transposition_table,
                                          opening_book=opening_book, end_time=end_time, priority_move=best_move,
                                          debug_counts=debug_counts)
            if debug_counts and debug_search_count == 0:
                raise TimeoutError
            iterative_depth += 1
    except TimeoutError as _:
        pass
    if best_move is None:
        best_eval, best_move = search(board, depth=1, transposition_table=transposition_table,
                                      opening_book=opening_book, debug_counts=debug_counts)
    return best_eval, best_move


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
            "opening_book.txt",
            "/opening_book.txt",
            "../opening_book.txt",
        ]
        for path in possible_paths:
            try:
                self.opening_book = get_opening_book(path)
                break
            except FileNotFoundError:
                continue
        self.transposition_table = TranspositionTable(max_size=3_000_000)
        self.estimated_time_taken = 0

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
        ASSUMES the move will be pushed to board afterwards!
        Call iterative_deepening_search directly if you just want a move

        Returns:
            str: The current location and the next move.
        """

        # Assume that you are playing an arbitrary game. This function, which is
        # the core "brain" of the bot, should return the next move in any circumstance.

        allotted_time = 1
        if self.estimated_time_taken >= 40:
            allotted_time = 3 - self.estimated_time_taken / 20
        _, move = iterative_deepening_search(
            self.board,
            search_time=self.time_per_turn,
            opening_book=self.opening_book,
            transposition_table=self.transposition_table,
            debug_counts=self.debug,
        )
        self.estimated_time_taken += allotted_time
        if self.transposition_table is not None:
            self.board.push(move)
            if (tup := get_representation_tuple(self.board)) in self.transposition_table:
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
                    test_bot.get_move(chess_bot.board, best_move=True)
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
        print(pgn.Game.from_board(chess_bot.board))
