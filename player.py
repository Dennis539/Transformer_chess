from typing import Optional
import chess
import torch
import torch.nn as nn
import requests
from chess_tournament.players import Player

import numpy as np
from functools import partial
from typing import Tuple

from huggingface_hub import PyTorchModelHubMixin


class TransformerPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        json_path_github = "https://raw.githubusercontent.com/Dennis539/Transformer_chess/refs/heads/main/move_to_idx.json"
        model_id_policy = "DVriend/Transformer-encodec-policy"
        model_id_value = "DVriend/Transformer-encodec-value"
        self.move_to_idx = requests.get(json_path_github).json()
        model_value = ValueTransformer(num_layers=12).from_pretrained(model_id_value)
        model_policy = EndgamePolicyTransformer(
            num_moves=len(self.move_to_idx), num_layers=4
        ).from_pretrained(model_id_policy)

        self.model_value_compiled = torch.compile(model_value)
        self.model_policy_compiled = torch.compile(model_policy)
        self.model_value_compiled.to("cuda")
        self.model_policy_compiled.to("cuda")

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        move = choose_move(
            board,
            self.model_value_compiled,
            self.model_policy_compiled,
            depth=3,
            input_move_to_idx=self.move_to_idx,
        )
        return move.uci() if move else None


class ValueTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_channels: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.square_projection = nn.Linear(input_channels, d_model)

        self.square_position_embedding = nn.Parameter(torch.zeros(64, d_model))
        nn.init.normal_(self.square_position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.loss_fn = nn.SmoothL1Loss(beta=0.1)

    def forward(self, board_tensor=None, labels=None):
        """
        board_tensor: (batch_size, 20, 8, 8)
        returns: (B,) normalized value in roughly [-1,1]
        """
        batch_size, num_channels, _, _ = board_tensor.shape

        square_tokens = board_tensor.permute(0, 2, 3, 1).reshape(
            batch_size, 64, num_channels
        )

        square_embeddings = self.square_projection(square_tokens)
        square_embeddings = (
            square_embeddings + self.square_position_embedding.unsqueeze(0)
        )

        encoded_squares = self.encoder(square_embeddings)
        board_embedding = encoded_squares.mean(dim=1)

        pred = self.value_head(board_embedding).squeeze(-1)
        loss = None
        if labels is not None:
            loss = self.loss_fn(pred, labels)

        return {"loss": loss, "logits": pred}


class EndgamePolicyTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        num_moves: int,
        in_channels: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.square_projection = nn.Linear(in_channels, d_model)

        self.square_position_embedding = nn.Parameter(torch.zeros(64, d_model))
        nn.init.normal_(self.square_position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_moves),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, board_tensor, labels=None):
        """
        board_tensor: (batch_size, 20, 8, 8)
        returns: dict with
            logits: (B, num_moves)
            loss: scalar or None
        """
        batch_size, num_channels, _, _ = board_tensor.shape

        square_tokens = board_tensor.permute(0, 2, 3, 1).reshape(
            batch_size, 64, num_channels
        )
        square_embeddings = self.square_projection(square_tokens)

        square_embeddings = (
            square_embeddings + self.square_position_embedding.unsqueeze(0)
        )

        encoded_squares = self.encoder(square_embeddings)
        board_embedding = encoded_squares.mean(dim=1)

        logits = self.policy_head(board_embedding)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]
COLORS = [chess.WHITE, chess.BLACK]


def board_to_tensor(
    board,
    *,
    include_turn: bool = True,
):
    """
    example: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0
    """

    planes = []

    for color in COLORS:
        for ptype in PIECE_TYPES:
            plane = np.zeros((8, 8), dtype=np.float32)
            for sq in board.pieces(ptype, color):
                r = chess.square_rank(sq)
                f = chess.square_file(sq)
                plane[r, f] = 1.0
            planes.append(plane)

    if include_turn:
        turn_plane = (
            np.ones((8, 8), dtype=np.float32)
            if board.turn == chess.WHITE
            else np.zeros((8, 8), dtype=np.float32)
        )
        planes.append(turn_plane)

    # Partial allows for the creation of preloaded functions
    plane = partial(np.full, (8, 8), dtype=np.float32)
    planes.append(
        plane(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    )
    planes.append(
        plane(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    )
    planes.append(
        plane(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    )
    planes.append(
        plane(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    )

    ep_plane = np.zeros((8, 8), dtype=np.float32)
    if board.ep_square is not None:
        r = chess.square_rank(board.ep_square)
        f = chess.square_file(board.ep_square)
        ep_plane[r, f] = 1.0
    planes.append(ep_plane)

    halfmove = min(board.halfmove_clock, 100) / 100.0
    fullmove = min(board.fullmove_number, 200) / 200.0
    planes.append(np.full((8, 8), halfmove, dtype=np.float32))
    planes.append(np.full((8, 8), fullmove, dtype=np.float32))

    x = np.stack(planes, axis=0)
    return x


@torch.inference_mode()
def eval_board_value(
    board: chess.Board, model: ValueTransformer, device="cuda"
) -> float:
    """Evaluation the board according to the ValueTransformer
    model.

    During begin- and midgame, the state of the board will be evaluated
    using one of the models trained. This will output a score which can
    tell the chess-engine whether the particular board position is a
    board position containing a high score or not.
    As many board positions will be evaluatad multiple times, previous
    calculations will be cached using a board hash.
    Args:
        board (chess.Board): python-chess Board to be evaluated
        model_value (ValueTransformer): Value model trained for determining
            the value of the board using the current board.
        device (str, optional): Device for the model to run on.
            Defaults to "cuda".

    Returns:
        float: Determined score for this board
    """
    key = board._transposition_key()
    if key in eval_cache:
        return float(eval_cache[key])

    x = board_to_tensor(board).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    pred = model(x)["logits"].item()
    eval_cache[key] = pred
    return float(pred)


@torch.inference_mode()
def eval_board_policy(
    board: chess.Board, model: EndgamePolicyTransformer, device="cuda"
) -> float:
    """Evaluation the board according to the EndgamePolicyTransformer
    model.

    During endgame, another simple model will take over. Unlike the value
    model, the policy model outputs an array of logits, each logit corresponding
    with a move on which the Transformer was trained on. As not all of these moves
    are legal, an extra step is taken to find out what the highest logit is of
    all of the legal moves
    Args:
        board (chess.Board): python-chess Board to be evaluated
        model_value (EndgamePolicyTransformer): Trained Policy model
        device (str, optional): Device for the model to run on.
            Defaults to "cuda".
    Returns:
        float: The value of the move which contained the highest logit value.

    Note:
        The EndgamePolicyTransformer will only output an array with logits, where
        it is unclear which logit value corresponds to which move. The mapping of
        move_to_idx was therefore created to keep track of which index corresponds
        to which move.
    """

    temp_board = board.copy()
    move = temp_board.pop().uci()
    key = temp_board._transposition_key()
    if key in eval_cache_endgame:
        if move in eval_cache_endgame[key]:
            return eval_cache_endgame[key][move]
    legal_moves = list(board.legal_moves)
    x = board_to_tensor(board).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    logits = model(x)["logits"][0]

    max_val = float("-inf")
    eval_cache_endgame[key] = {}
    for mv, index in move_to_idx.items():
        val = logits[index].item()
        eval_cache_endgame[key][mv] = val

    return eval_cache_endgame[key][move]


PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def is_endgame(board: chess.Board) -> bool:
    """Checks if the game has reached endgame.

    With some simple rules, the engine determines whether the
    'endgame' has started. During endgame, different choices
    will be made by the engine

    Args:
        board (chess.Board): python-chess Board to be evaluated

    Returns:
        bool: returns if endgame
    """
    value_black = 0
    value_white = 0
    for piece, value in PIECE_VALUE.items():
        if piece not in (chess.PAWN, chess.KING):
            value_black += len(board.pieces(piece, chess.BLACK)) * value
            value_white += len(board.pieces(piece, chess.WHITE)) * value

    return value_white < 5 or value_black < 5 or (value_white + value_black) < 10


def is_passed_pawn(board: chess.Board, square: chess.Square) -> bool:
    """Check if a pawn is passed

    If a pawn is passed, that means that it has no pieces in front of
    it (both directly and diagonal). This is called a 'passed pawn' and
    is an important concept in the endgame of chess, where promoting
    of pawns becomes (one of) the most important win factors.

    Args:
        board (chess.Board): python-chess Board to be evaluated
        square (chess.Square): a particular square on the chess board.

    Returns:
        bool: pawn is passed
    """
    board_file = chess.square_file(square)
    board_rank = chess.square_rank(square)
    piece = board.piece_at(square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    color = piece.color
    # Check for pawns diagonal to the pawn in scope
    pawn_diagonals = [board_file - 1, board_file, board_file + 1]
    board_files_to_check = [fil for fil in pawn_diagonals if fil <= 7 and fil >= 0]
    for fil in board_files_to_check:
        for rank in range(8):
            if color == chess.WHITE and rank <= board_rank:
                continue
            elif color == chess.BLACK and rank >= board_rank:
                continue
            square = chess.square(fil, rank)
            target_piece = board.piece_at(square)
            if (
                target_piece
                and target_piece.piece_type == chess.PAWN
                and target_piece.color != color
            ):
                return False
    return True


def pawn_bonus(board: chess.Board, move: chess.Move) -> float:
    """Bonus pawn moves will get

    Movement of pawns should be more important in the end-game.
    Therefore, given the potential chess move a bonus will be
    given to pawns relative to the  position they are.
    A few components will be assessed:
    - How much will the pawn advance with the given move (1 or two places)
    - What will be the new rank of the pawn (ranks closer to the opposite
        side of the board will get a higher bonus)
    - Is is a 'passed pawn' (see is_passed_pawn)

    Args:
        board (chess.Board): python-chess Board to be evaluated
        move (chess.Move): A potential chess move


    Returns:
        float: Bonus score the particular pawn gets.
    """
    bonus = 0
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    advance = abs(to_rank - from_rank)
    bonus += advance * 0.5
    relative_rank = to_rank if board.turn == chess.WHITE else 7 - to_rank
    bonus += 0.3 * relative_rank

    board.push(move)
    square = move.to_square

    if is_passed_pawn(board, square):
        bonus += 3
        bonus += 0.8 * relative_rank
    board.pop()
    return bonus


def capture_score(board: chess.Board, move: chess.Move) -> float:
    """Bonus given to a potential move which captures an opposing
    piece.

    Args:
        board (chess.Board): python-chess Board to be evaluated
        move (chess.Move): A potential chess move

    Returns:
        bool: score this particular move gets

    Note:
        It is assumed that the piece might be captures afterwards
    """
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        if victim:
            victim_value = PIECE_VALUE[victim.piece_type]
        else:
            return 0
        attacker = board.piece_at(move.from_square)
        attacker_value = PIECE_VALUE[attacker.piece_type]
        return 10 + 1.01 * victim_value - attacker_value

    else:
        return 0


def ordered_moves(
    board: chess.Board, best_move: chess.Move | None = None
) -> list[chess.Move]:
    """Orders all potential moves based on potential score obtained

    Performance of alpha-beta trees is dependent on the ordering of moves.
    The main idea is that most moves will have a neutral effect on the board
    while some move will have a big effect (e.g. capturing a queen or promoting
    a pawn). Potential moves should thus be sorted such that potential moves causing
    bigger positive board swings should be evaluated first.
    See https://www.chessprogramming.org/Move_Ordering#Typical_move_ordering
    Args:
        board (chess.Board): python-chess Board to be evaluated
        best_move (chess.Move | None): Best move that was obtained through
            search of depth -1.

    Returns:
        list[chess.Move]: List of potential moves ordered by potential score obtained

    Note:
        The score is NOT the score assigned from the transformer but a score
        to determine the order in which potential moves will be evaluated by
        the transformer.
    """
    moves = list(board.legal_moves)

    def move_score(move: chess.Move) -> int:
        """Determines the score of a particular move.

        The score obtained from this function will be used to determine the
        potential this particular move might have.
        Args:
            move (chess.Move): _description_

        Returns:
            int: Final score of that move.
        """
        score = 0
        if best_move and move == best_move:
            score += 200
        if move.promotion:
            score += 9
        score += capture_score(board, move)
        endgame = is_endgame(board)
        if board.gives_check(move):
            if endgame:
                score += 1
            else:
                score += 5

        piece = board.piece_at(move.from_square)
        if endgame and piece.piece_type == chess.PAWN:
            score += pawn_bonus(board, move)
        if board.fullmove_number <= 15:
            if piece.piece_type == chess.KING:
                if board.is_castling(move):
                    score += 20
                else:
                    score -= 200
        return score

    moves.sort(key=move_score, reverse=True)
    return moves


def choose_move_value_alphabeta_it(
    board: chess.Board,
    model_value,
    model_policy,
    depth: int,
    device: str,
    ply: int,
    pv_table: list,
    endgame: bool,
) -> Tuple[float, list]:
    """Main caller of the alpha-beta search tree.

    Alpha-beta is a search algorithm that can aid a chess engine in finding the best
    move at various search depths. A native algorithm would look into all possible
    moves and consequently into all counter-moves for that particular moves, etc. However,
    if e.g. move 1 at depth = 1 from white results into black being able to capture a queen
    as its best possible counter move, there would be no reason to explore further move
    possibilities for this move. This would be most effective if the counter move of black
    would be searched first, something which will be done through move ordering based on
    its potential score.
    See https://www.chessprogramming.org/Alpha-Beta for more information.

    Args:
        board (chess.Board): python-chess Board to be evaluated
        model_value (ValueTransformer): Value model trained for determining
            the value of the board using the current board.
        model_policy (EndgamePolicyTransformer): Trained Policy model
        depth (int, optional): depth of the search tree.
        device (str, optional): Device for the model to run on.
        ply (int): search index
        pv_table (list): Triangular PV-table which tracked the best move found
            at previous depths.
        endgame (bool): Is it the end phase of the game.

    Returns:
        Tuple[float, list]: _description_


    """

    def alphabeta(
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        pv_table: list,
    ) -> Tuple[float, list]:
        """Main worker for alpha-beta algorithm.

        Two variables, alpha and beta, are maintained. In simple terms, both represent the
        minimun score that either player can be assured of.
        Args:
            depth (int, optional): depth of the search tree.
            alpha (float): The minimum score that the maximizing player
                (e.g., White) can guarantee so far along the current search path.
            beta (float): The maximum score that the minimizing player
                (e.g., Black) can guarantee so far along the current search path.
            ply (int): search index
            pv_table (list): Triangular PV-table which tracked the best move found
                at previous depths.

        Returns:
            Tuple[float, list]: _description_
        """
        if board.is_checkmate():
            return -1e9 if board.turn == chess.WHITE else 1e9, []
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_draw()
        ):
            return 0.0, []

        if depth == 0:
            if endgame:
                v = eval_board_policy(board, model_policy, device=device)

            else:
                v = eval_board_value(board, model_value, device=device)
            return v, []

        maximizing = board.turn == chess.WHITE
        best_line = []

        pv_move = pv_table[ply] if ply < len(pv_table) else None

        if maximizing:
            best = float("-inf")
            for ord_move in ordered_moves(board, pv_move):
                board.push(ord_move)
                val, child_line = alphabeta(depth - 1, alpha, beta, ply + 1, pv_table)
                board.pop()

                if val > best:
                    best_line = [ord_move] + child_line
                    best = max(best, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
        else:
            best = float("inf")
            for ord_move in ordered_moves(board, pv_move):
                board.push(ord_move)
                val, child_line = alphabeta(depth - 1, alpha, beta, ply + 1, pv_table)
                board.pop()

                if val < best:
                    best_line = [ord_move] + child_line
                    best = min(best, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break

        return best, best_line

    alpha = float("-inf")
    beta = float("inf")

    maximizing_root = board.turn == chess.WHITE
    best_score = float("-inf") if maximizing_root else float("inf")
    best_line = []

    pv_move = pv_table[ply] if ply < len(pv_table) else None
    for ord_move in ordered_moves(board, pv_move):
        board.push(ord_move)

        if board.is_checkmate():
            board.pop()
            return 1e9, [ord_move]

        score, child_line = alphabeta(depth - 1, alpha, beta, ply + 1, pv_table)
        board.pop()

        if maximizing_root:
            if score > best_score:
                best_score = score
                best_line = [ord_move] + child_line
            alpha = max(best_score, alpha)

        else:
            if score < best_score:
                best_score = score
                best_line = [ord_move] + child_line
            beta = min(best_score, beta)

    return best_score, best_line


def choose_move(
    board: chess.Board,
    model_value: ValueTransformer,
    model_policy: EndgamePolicyTransformer,
    depth: int = 3,
    input_move_to_idx: dict[str, int] = {},
) -> chess.Move:
    """Choose the best move for a given board

    This function performs an alpha-beta search using a simple version of
    iterative deepening using an Triangular PV-table. The idea of iterative
    deepening is search the best possible move per depth, where each depth
    will output a PV-move which will be used as input for the next depth.
    The idea is that moves at depth = n + 1 will more likely lead to the most
    optimal move as depth = n.
    Args:
        board (chess.Board): python-chess Board to be evaluated
        model_value (ValueTransformer): Value model trained for determining
            the value of the board using the current board.
        model_policy (EndgamePolicyTransformer): Trained Policy model
        depth (int, optional): depth of the search tree. Defaults to 3.

    Returns:
        chess.Move: The chess move that has been obtained throught the alpha-beta
            search tree.
    """
    MATE_SCORE = 1e9
    global eval_cache
    global move_to_idx
    global eval_cache_endgame
    move_to_idx = input_move_to_idx
    eval_cache = {}
    eval_cache_endgame = {}
    pv_table = []
    endgame = is_endgame(board)

    for dep in range(1, depth + 1):
        score, child_line = choose_move_value_alphabeta_it(
            board,
            model_value,
            model_policy,
            depth=dep,
            device="cuda",
            ply=0,
            pv_table=pv_table,
            endgame=endgame,
        )
        pv_table = child_line
        best_move = child_line[0] if child_line else None

        if best_move is None or abs(score) >= MATE_SCORE:
            break
    return best_move


class TransFormerPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        json_path_github = "https://raw.githubusercontent.com/Dennis539/Transformer_chess/refs/heads/main/move_to_idx.json"
        model_id_policy = "DVriend/Transformer-encodec-policy"
        model_id_value = "DVriend/Transformer-encodec-value"
        self.move_to_idx = requests.get(json_path_github).json()
        model_value = ValueTransformer(num_layers=12).from_pretrained(model_id_value)
        model_policy = EndgamePolicyTransformer(
            num_moves=len(self.move_to_idx), num_layers=4
        ).from_pretrained(model_id_policy)

        self.model_value_compiled = torch.compile(model_value)
        self.model_policy_compiled = torch.compile(model_policy)
        self.model_value_compiled.to("cuda")
        self.model_policy_compiled.to("cuda")

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        move = choose_move(
            board,
            self.model_value_compiled,
            self.model_policy_compiled,
            depth=3,
            input_move_to_idx=self.move_to_idx,
        )
        return move.uci() if move else None
