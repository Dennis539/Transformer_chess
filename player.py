from typing import Optional
from models import ValueTransformer, EndgamePolicyTransformer
import chess
import torch
import requests
from moves import choose_move
from chess_tournament.players import Player


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
