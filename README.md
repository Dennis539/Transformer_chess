# Transformer_chess

The chess engine contains out of three different building blocks:

- An encoder which has learned 'valuable positions', called value-model
- A small encoded which should be specialized in finding the best move during late game, called policy-model
- A chess engine using alpha-beta search to find the best move (using a depth of 3)

### Value-model

The value model was trained using a dataset containing around two million chess position (in fen-notation)
and a stockfish evaluation. The idea would be that the model is able to learn how valuable a particular
board position is based on a given fen-position. Thus, it can evaluate potential moves and give an output
for that. For this model, the following steps have been performed:

- Convert fen to tensor: To improve the learning of the model, the fen positions were transformed to tensors
  each tensor contained 20 matrices, where each matrix was 8x8 format containing either 0s or 1s (except when
  the matrix represents a piece on the board, then there will a 1 on its board position while the other values
  are 0). This convertion makes the state of the board more expicit.
- Tensors are reformatted into expected transformer shape and transformed to a 256 embedding token
- Positional information is added
- Encoding of interactions between squares
- Pooled into one vector representation of the board
- The mean value is through the value head to output one score which represents how good the position is.

Results were quite promising with it consistently beating the random player and able to beat a weak stockfish.
It was observed that the model still often made strange decisions against stronger opponents. Therefore,
it is theorized that this is because the model has become good at picking the right move when it is already winning,
but struggles when it isn't that clear.

### Policy model

During testing of the model, it was noticed that the Value-model seemed to struggle in the late-game. More
precisely, in winning positions (e.g. having much more material than the opponent) it would still always try
to check the opponent, leading to checks, following by opponent king movenent, followed by another check.
Therefore, it was attempted to train a separate model which could take over during the late-game, with the
hope of it playing better during this phase.
For training, a dataset was created using fen-positions and move created from two stockfish engines played
against each other (found on Huggingface). In total, around one million rows were sampled where the number of
total moves exceeded 100. It is important to note that this dataset does not contain all possible moves, thus
the model was only trained on moves which occured more than 50 times in the dataset.
In essence, this model was almost trained the same, with the exception that the input was not board value but
board move. As a result, the outcome was not one value, but a tensor which contained one value for each move
that the model saw during training.
However, it doesn't appear that the model gave the desired output. Thus, it will only be used at the very end
of matches.

### Chess engine

The value model managed to play a bit of chess but often failed to win matches. Therefore, a search algorithm
was build on top of the model to help it find more promising moves. The most basic search algorithm is called
alpha-beta searching, which is an optimized version of the MiniMax algorithm (which is a brute-force search).
In a nutshell, alpha-beta searching is able to do this by ordering moves bases on these moves having a higher
probability of yielding a good board position, while simultaneously tracking potential counter moves from
opponents at higher search depths. This is not the same as the score of the actual move (created from the
encoder).
