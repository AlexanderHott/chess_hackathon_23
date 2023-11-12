import datetime
import itertools
import logging
import random
from itertools import chain
from collections import OrderedDict

import chess

from grob import parameters
from grob.parameters import (
    WHITE_PIECE_SQUARE_TABLES,
    BLACK_PIECE_SQUARE_TABLES,
    PIECE_VALUES,
)

zlogger = logging.getLogger("zlogger")
logging.basicConfig(filename=f"{datetime.datetime.now().strftime('%H%M%S')}.log", level=logging.INFO)

INF = float("inf")

TableEntryType = int
TABLE_ENTRY_TYPE = [EXACT, LOWER_BOUND, UPPER_BOUND] = range(3)


class TranspositionTable(OrderedDict[tuple[int, ...], tuple[int, float, TableEntryType]]):
    def __init__(self, max_size=5, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: tuple[int, ...], value: tuple[int, float, TableEntryType]):
        if key not in self:
            if len(self) == self.max_size:
                self.popitem(last=False)
        super().__setitem__(key, value)


def get_representation_tuple(board: chess.Board):
    return (*(board.pieces_mask(p_type, color)
              for p_type, color in itertools.product(chess.PIECE_TYPES, chess.COLORS)), board.turn)


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


def evaluate(board: chess.Board, use_square_scores: bool = True) -> float:
    """
    Returns: board evaluation
    """
    white_sum = 0
    black_sum = 0

    white_sum += (white_material := material_score(board, chess.WHITE))
    black_sum += (black_material := material_score(board, chess.BLACK))

    if use_square_scores:
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
        debug_search_depth = max(debug_search_depth, levels_deep)

    evaluation = evaluate(board, use_square_scores=use_square_scores)
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
    transposition_table: TranspositionTable | None = None,
    _use_transposition_table: bool = False,
    opening_book: dict[str, dict[str, int]] | None = None,
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
        transposition_table: a table of already searched positions and their evaluations
        opening_book: an opening book
        using_opening_book: whether the book is still being used
        use_square_scores: whether to use square scores
        guess_move_order: whether to sort moves according to an initial guess evaluation
        search_captures: whether to search all captures after depth limit is reached
        search_checks: whether to search all checks after depth limit is reached
        debug_counts: whether to update global count variables
        _use_transposition_table: whether to use the transposition_table. First search shouldn't
    Returns: the evaluation of the current position, along with the best move if the depth has not been reached
    """
    if using_opening_book:
        if opening_book is not None:
            fen = board.fen()[:-4]
            if fen in opening_book:
                opening_moves: dict[str, int] = opening_book[
                    fen
                ]  # remove the moves portion
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
            return -INF, None  # current player has lost
        else:
            return 0, None  # game is a draw

    tuple_representation = None
    if transposition_table is not None and _use_transposition_table:
        tuple_representation = get_representation_tuple(board)
        if tuple_representation in transposition_table:
            cached_depth, cached_eval, entry_type = transposition_table[tuple_representation]
            if depth <= cached_depth and is_entry_applicable(cached_eval, alpha, beta, entry_type):
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
                search_checks=search_checks,
                use_square_scores=use_square_scores,
                debug_counts=debug_counts,
            )
        else:
            instant_evaluation = evaluate(board, use_square_scores=use_square_scores)
            return instant_evaluation, None

    moves = board.legal_moves
    if guess_move_order:
        moves = order_moves(board, moves)
    best_move = None
    _debug_move_evals = {}
    for move in moves:
        board.push(move)
        evaluation = -search(
            board,
            depth - 1,
            -beta,
            -alpha,
            levels_deep=levels_deep + 1,
            transposition_table=transposition_table,
            opening_book=opening_book,
            using_opening_book=using_opening_book,
            use_square_scores=use_square_scores,
            guess_move_order=guess_move_order,
            search_captures=search_captures,
            search_checks=search_checks,
            debug_counts=debug_counts,
            _use_transposition_table=True,
        )[0]
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
