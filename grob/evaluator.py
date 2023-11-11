import random
from datetime import datetime
from itertools import chain
from collections import OrderedDict
import logging

import chess

from grob import parameters
from grob.parameters import (
    WHITE_PIECE_SQUARE_TABLES,
    BLACK_PIECE_SQUARE_TABLES,
    PIECE_VALUES,
)

logging.basicConfig(filename=f"log/{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log", level=logging.DEBUG)
eval_logger = logging.getLogger("eval")

INF = float("inf")


class TransitionTable(OrderedDict[int, tuple[int, float]]):
    def __init__(self, max_size=5, *args, **kwds):
        self.max_size = max_size
        super().__init__(*args, **kwds)

    def __setitem__(self, key: int, value: tuple[int, float]):
        if key not in self:
            if len(self) == self.max_size:
                self.popitem(last=False)
        super().__setitem__(key, value)


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


def generate_zobrist_numbers() -> list[int]:
    """
    returns: a list of random integers to be used for Zobrist hashing
    """
    zobrist_numbers = []
    for _ in range(64 * 12 + 4):
        zobrist_numbers.append(random.getrandbits(64))
    return zobrist_numbers


def get_zobrist_number(
    square: chess.Square,
    color: chess.Color,
    piece_type: chess.PieceType,
    zobrist_numbers: list[int],
) -> int:
    """
    returns: the Zobrist number for a particular piece
    """
    return zobrist_numbers[square * 12 + color * 6 + piece_type]


def get_zobrist_castling(color: chess.Color, side: chess.PieceType, zobrist_numbers: list[int]):
    """
    returns: the Zobrist number representing a castle
    """
    return zobrist_numbers[64 * 12 + color * 2 + (side == chess.KING)]


def get_zobrist_hash(board: chess.Board, zobrist_numbers: list[int]) -> int:
    """
    returns: the Zobrist hash for a board
    """
    zobrist_hash = 0
    for square in range(64):
        piece = board.piece_at(square)
        if piece is not None:
            zobrist_hash ^= get_zobrist_number(
                square, piece.color, piece.piece_type, zobrist_numbers
            )
    for color in chess.COLORS:
        if board.has_kingside_castling_rights(color):
            zobrist_hash ^= zobrist_numbers[64 * 12 + color]
        if board.has_queenside_castling_rights(color):
            zobrist_hash ^= zobrist_numbers[64 * 12 + 2 + color]
    return zobrist_hash


def update_zobrist_hash(
    zobrist_hash: int, board: chess.Board, move: chess.Move, zobrist_numbers: list[int]
) -> int:
    """
    returns: an updated Zobrist hash
    """
    from_piece = board.piece_at(move.from_square)
    to_piece = board.piece_at(move.to_square)
    if from_piece is None:
        raise ValueError("from_piece must not be None")
    zobrist_hash ^= get_zobrist_number(move.from_square, from_piece.color, from_piece.piece_type, zobrist_numbers)
    if to_piece is not None:
        # remove taken piece
        zobrist_hash ^= get_zobrist_number(move.to_square, to_piece.color, to_piece.piece_type, zobrist_numbers)
    if move.promotion is None:
        # normal movement
        zobrist_hash ^= get_zobrist_number(move.to_square, from_piece.color, from_piece.piece_type, zobrist_numbers)
    else:
        # move promotion
        zobrist_hash ^= get_zobrist_number(move.to_square, from_piece.color, move.promotion, zobrist_numbers)
    if from_piece.piece_type == chess.PAWN and chess.square_file(move.from_square) != chess.square_file(move.to_square):
        if from_piece.color == chess.WHITE:
            # remove the en passanted pawn
            zobrist_hash ^= get_zobrist_number(move.to_square - 8, chess.BLACK, chess.PAWN, zobrist_numbers)
        elif from_piece.color == chess.BLACK:
            zobrist_hash ^= get_zobrist_number(move.to_square + 8, chess.WHITE, chess.PAWN, zobrist_numbers)
    # handle all the castling cases
    if from_piece.piece_type == chess.KING:
        if from_piece.color == chess.WHITE:
            if move.from_square == chess.E1 and move.to_square == chess.G1:
                zobrist_hash ^= get_zobrist_number(chess.H1, chess.WHITE, chess.ROOK, zobrist_numbers)
                zobrist_hash ^= get_zobrist_number(chess.F1, chess.WHITE, chess.ROOK, zobrist_numbers)
                zobrist_hash ^= get_zobrist_castling(chess.WHITE, chess.KING, zobrist_numbers)
            elif move.from_square == chess.E1 and move.to_square == chess.C1:
                zobrist_hash ^= get_zobrist_number(chess.A1, chess.WHITE, chess.ROOK, zobrist_numbers)
                zobrist_hash ^= get_zobrist_number(chess.D1, chess.WHITE, chess.ROOK, zobrist_numbers)
                zobrist_hash ^= get_zobrist_castling(chess.WHITE, chess.QUEEN, zobrist_numbers)
        elif from_piece.color == chess.BLACK:
            if move.from_square == chess.E8 and move.to_square == chess.G8:
                zobrist_hash ^= get_zobrist_number(chess.H8, chess.BLACK, chess.ROOK, zobrist_numbers)
                zobrist_hash ^= get_zobrist_number(chess.F8, chess.BLACK, chess.ROOK, zobrist_numbers)
                zobrist_hash ^= get_zobrist_castling(chess.BLACK, chess.KING, zobrist_numbers)
            elif move.from_square == chess.E8 and move.to_square == chess.C8:
                zobrist_hash ^= get_zobrist_number(chess.A8, chess.BLACK, chess.ROOK, zobrist_numbers)
                zobrist_hash ^= get_zobrist_number(chess.D8, chess.BLACK, chess.ROOK, zobrist_numbers)
                zobrist_hash ^= get_zobrist_castling(chess.BLACK, chess.QUEEN, zobrist_numbers)
    return zobrist_hash


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
    square: chess.Square, piece: chess.PieceType, color: chess.Color
) -> int:
    if color == chess.WHITE:
        return WHITE_PIECE_SQUARE_TABLES[piece][square]
    else:
        return BLACK_PIECE_SQUARE_TABLES[piece][square]


def get_square_scores(board: chess.Board, color: chess.Color) -> int:
    """
    Returns: the square bonus scores for each piece
    """
    total_square_bonus = 0
    for piece_type in chess.PIECE_TYPES:
        # is there a more efficient way of doing this using the bit mask directly?
        for square in board.pieces(piece_type, color):
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
    ) * parameters.ENDGAME_WEIGHT_CONTROL
    evaluation = 0
    if (
        my_material > enemy_material + parameters.CORNER_KING_MATERIAL_DIFFERENCE_REQ
        and endgame_weight > parameters.CORNER_KING_ENDGAME_WEIGHT_REQ
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


def evaluate(board: chess.Board) -> float:
    """
    Returns: board evaluation
    """
    if board.is_repetition():
        eval_logger.debug("Eval on draw")
        return 0
    white_sum = 0
    black_sum = 0

    white_sum += (white_material := material_score(board, chess.WHITE))
    black_sum += (black_material := material_score(board, chess.BLACK))

    white_sum += get_square_scores(board, chess.WHITE) * parameters.SQUARE_SCORE_WEIGHT
    black_sum += get_square_scores(board, chess.BLACK) * parameters.SQUARE_SCORE_WEIGHT

    white_sum += endgame_corner_king(board, chess.WHITE, white_material, black_material)
    black_sum += endgame_corner_king(board, chess.BLACK, black_material, white_material)

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
    board: chess.Board, moves: chess.LegalMoveGenerator
) -> list[chess.Move]:
    """
    Returns: sorts a list of moves in place according to guess_move_evaluation
    """
    return sorted(moves, key=lambda m: guess_move_evaluation(board, m), reverse=True)


def search_all_captures(
    board: chess.Board,
    alpha: float,
    beta: float,
    levels_deep: int = 0,
    search_checks: bool = True,
    debug_counts: bool = False,
) -> tuple[float, chess.Move | None]:
    """
    Returns: an alpha-beta evaluation that only considers capture moves (and check moves)
    """
    if debug_counts:
        global debug_search_count
        global debug_search_depth
        debug_search_count += 1
        debug_search_depth = max(debug_search_depth, levels_deep)

    evaluation = evaluate(board)
    if evaluation >= beta:
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
    eval_logger.debug(f"Checking {len(important_moves)} capture moves at depth {debug_search_depth}")

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
            return beta, None
        if evaluation > alpha:  # must not be >=
            alpha = evaluation
            best_move = move
    return alpha, best_move


def search(
    board: chess.Board,
    depth: int,
    alpha: float = -INF,
    beta: float = INF,
    levels_deep: int = 0,
    transition_table: TransitionTable | None = None,
    _use_transition_table: bool = False,
    zobrist_numbers: list[int] | None = None,
    zobrist_hash: int = 0,
    opening_book: dict[str, dict[str, int]] | None = None,
    using_opening_book: bool = True,
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
        transition_table: a table of already searched positions and their evaluations, using Zobrist hashing
        zobrist_numbers: random numbers to be used for Zobrist hashing, or none if hashing shouldn't be used
        zobrist_hash: the current board's zobrist hash, if zobrist_numbers is not None
        opening_book: an opening book
        using_opening_book: whether the book is still being used
        guess_move_order: whether to sort moves according to an initial guess evaluation
        search_captures: whether to search all captures after depth limit is reached
        search_checks: whether to search all checks after depth limit is reached
        debug_counts: whether to update global count variables
        _use_transition_table: whether to use the transition_table. First search shouldn't
    Returns: the evaluation of the current position, along with the best move if the depth has not been reached
    """
    if using_opening_book:
        if opening_book is not None:
            fen = board.fen()[:-4]
            if fen in opening_book:
                opening_moves = opening_book[
                    fen
                ]  # remove the moves portion
                move = random.choices(
                    list(opening_moves.keys()), list(opening_moves.values())
                )[0]
                eval_logger.info(f"Using book move {move}")
                return INF, chess.Move.from_uci(move)
        else:
            using_opening_book = False

    if debug_counts:
        global debug_search_count
        global debug_search_depth
        debug_search_count += 1
        debug_search_depth = max(debug_search_depth, levels_deep)

    # Zob hash
    if zobrist_numbers is not None and transition_table is not None and _use_transition_table:
        if zobrist_hash in transition_table:
            cached_depth, cached_eval = transition_table[zobrist_hash]
            if depth <= cached_depth:
                if debug_counts:
                    global debug_tt_cache_hits
                    debug_tt_cache_hits += 1
                eval_logger.debug(f"Zob cache hit {zobrist_hash} with eval {cached_eval} at depth {depth}")
                return cached_eval, None

    if depth == 0:
        if search_captures:
            eval_logger.debug("Reached depth 0, searching captures")
            return search_all_captures(
                board,
                alpha,
                beta,
                levels_deep=levels_deep,
                search_checks=search_checks,
                debug_counts=debug_counts,
            )
        else:
            return evaluate(board), None

    moves = board.legal_moves
    if moves.count() == 0:
        if board.is_checkmate():
            return -INF, None  # current player has lost
        else:
            return 0, None  # game is a draw

    if guess_move_order:
        moves = order_moves(board, moves)
    best_move = None
    for move in moves:
        updated_hash = zobrist_hash
        if zobrist_numbers is not None:
            updated_hash = update_zobrist_hash(
                zobrist_hash, board, move, zobrist_numbers
            )
        board.push(move)
        board_fen = board.fen()
        evaluation = -search(
            board,
            depth - 1,
            -beta,
            -alpha,
            levels_deep=levels_deep + 1,
            transition_table=transition_table,
            zobrist_numbers=zobrist_numbers,
            zobrist_hash=updated_hash,
            opening_book=opening_book,
            using_opening_book=using_opening_book,
            guess_move_order=guess_move_order,
            search_captures=search_captures,
            search_checks=search_checks,
            debug_counts=debug_counts,
            _use_transition_table=True,
        )[0]
        eval_logger.debug(f"{board.fen()} {evaluation}")
        board.pop()
        # logging.debug(f"Eval for {move}: {evaluation}")
        if evaluation >= beta != INF:
            if transition_table is not None and _use_transition_table:
                transition_table[zobrist_hash] = (depth, beta)
            eval_logger.debug(f"Trimming {board_fen} with eval {evaluation} with beta {beta}")
            return beta, None
        if evaluation > alpha:
            alpha = evaluation
            best_move = move

    if transition_table is not None and _use_transition_table:
        eval_logger.debug(f"Setting zob hash {zobrist_hash} to eval {alpha} at depth {depth}")
        transition_table[zobrist_hash] = (depth, alpha)

    eval_logger.info(f"Found best move {best_move} for '{board.fen()}' with alpha {alpha}")
    return alpha, best_move
