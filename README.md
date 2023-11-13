# The Legendary Grob Bot

The bot is made up of two major components: board evaluation and move searching.

## Features

### Minimax Search

> Minmax is a decision rule for minimizing the possible loss for a worst case
> (maximum loss) scenario. When dealing with gains, it is referred to as
> "maximin" – to maximize the minimum gain.
>
> [Wikipedia](https://en.wikipedia.org/wiki/Minimax)

### Alpha-beta pruning

> The Alpha-Beta pruning algorithm is a significant enhancement to the minimax
> search algorithm that eliminates the need to search large portions of the game
> tree applying a branch-and-bound technique. Remarkably, it does this without
> any potential of overlooking a better move.
>
> [Chess Programming Wiki](https://www.chessprogramming.org/Alpha-Beta)

### Move ordering

> For the alpha-beta algorithm to perform well, the best moves need to be
> searched first, so we guess an approximate evaluation of a move and sort the
> order we search the moves.
>
> [Chess Programming Wiki](https://www.chessprogramming.org/Move_Ordering)

### "Quiescence" search

> Most chess programs, at the end of the main search perform a more limited
> quiescence search, containing fewer moves. The purpose of this search is to
> only evaluate "quiet" positions, or positions where there are no winning
> tactical moves to be made. This search is needed to avoid the horizon effect.
> Simply stopping your search when you reach the desired depth and then
> evaluate, is very dangerous. Consider the situation where the last move you
> consider is QxP. If you stop there and evaluate, you might think that you have
> won a pawn. But what if you were to search one move deeper and find that the
> next move is PxQ? You didn't win a pawn, you actually lost a queen. Hence the
> need to make sure that you are evaluating only quiescent (quiet) positions.
>
> [Chess Programming Wiki](https://www.chessprogramming.org/Quiescence_Search)

### Transposition Tables

> A Transposition Table is a database that stores results of previously
> performed searches. It is a way to greatly reduce the search space of a chess
> tree with little negative impact.
>
> [Chess Programming Wiki](https://www.chessprogramming.org/Transposition_Table)

### Positional Piece Bonuses

> Piece-Square Tables, a simple way to assign values to specific pieces on
> specific squares. A table is created for each piece of each color, and values
> assigned to each square. This scheme is fast, since the evaluation term from
> the piece square tables can be incrementally updated as moves are made and
> unmade in the search tree.
>
> [Chess Programming Wiki](https://www.chessprogramming.org/Piece-Square_Tables)

### Smart Endgame Evaluation

To win at endgames, our bot prioritizes two things in the endgame:

- push opponent king towards edge of the board
- moves kings closer to each other

This is because most checkmates with few pieces happen on the edge of the board.

### Opening book (includes Bongcloud Rapid Response Technology™)

We store a small sample of grandmaster games and if there is a response to an
opening move we do it blindly.

If the opponent plays the
[bongcloud](https://www.chess.com/openings/Kings-Pawn-Opening-The-Bongcloud), we
detect it with our Bongcloud Rapid Response Technology™ (BRRT) and respectfully
respond with `2. Ke2 Ke7 3. Ke1 Ke8`.

### Stuff we didn't get to

- Zobrist hashing
- Pawn Structure
- Partial Move ordering
- Bishop Pair / choosing the correct bishop color
- Optimizing code to run faster

## Quickstart

```
pip install -r requirements.txt

python -m grob
```

## Visualization

```
pip install -r requirements.txt
pip install -r dev_requirements.txt

jupyter lab
```

## Performance Testing

### `cProfile`

<https://docs.python.org/3/library/profile.html>

```
python -m cProfile -m grob  # test a module
python -m cProfile perf.py  # test a script
```

### `line_profiler`

<https://github.com/pyutils/line_profiler>

```
kernprof -l perf.py

python -m line_profiler -rmt "perf.py.lprof"
```

# The Brandeis Quant Club ML/AI Competition

**Project Description**: In this Python-driven competition, you will be building
a model to play chess. Specifically, given any arbitrary position, what is the
next best move?

## Getting Started

1. Clone this repository.
2. Install required dependencies using `pip install -r requirements.txt`
3. Run the application using `python bot.py`

## Submission

1. A team member is responsible for uploading a link to your
   `chess_hackathon_23` fork, accompanied by a video that provides an in-depth
   explanation of your code and overall logic. All team members are expected to
   appear in the video, which should have a duration of 2-3 minutes.
2. You have until 11:59pm on November 12th, 2023 to submit your build to
   [DevPost](https://quants.devpost.com/).

## Rules

1. Apart from the libraries listed in the requirements.txt file, you're allowed
   to utilize only scikit-learn, pandas, and numpy.
2. You're free to consult online resources, such as research papers, ChatGPT, or
   YouTube videos, for reference. However, direct copying of open-source
   solutions from platforms like GitHub or using APIs is not permitted.
3. You are permitted to have up to 4 members working with your team. You must be
   a part of the Brandeis University community.

## Usage

This skeleton is heavily derived from the
[python-chess](https://python-chess.readthedocs.io/en/latest/) open-source
library. You may use any aspect of this library for the purposes of building
your bot (other than calling premade models to determine moves).

The code skeleton involves a straightforward interaction between your code and
an example bot inside of `test_bot.py`. Depending on the configuration, the
example bot will either select a random piece or opt for the best possible move,
if applicable.

The central logic of your bot should be contained within the
`next_move(self) -> str:` function. When provided with a chessboard, this
function is responsible for identifying the optimal next move for either the
white or black pieces. This task can be challenging and necessitates a good
grasp of the python-chess library. You may create any additional python
functions or classes. Do **not** create any additional Python files, though.

Ensure you have a solid understanding of Chess board notation. While there are
several methods to input commands (moves) into the `python-chess` library, it's
generally advisable to use the initial move-to, new move format, like`e2e3`.
While it's sometimes possible to use a simpler format, such as `e3`, where the
library will move the only valid piece to that location, it's recommended to
avoid this approach for the sake of simplicity.

## Forsyth-Edwards Notation (FEN)

The board can be initialized as a new game or it can be passed a FEN board
confirguation. I.e., `chess_bot = Bot()` or
`chess_bot = Bot("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")`,
for example.

A list of puzzles with their corresponding FEN has been added in the
`puzzles.txt` file. This will be extremely useful when testing the efficacy of
your bot. It is recommended you build additional testing functions in the
`test_bot.py` file to utilize these puzzles systematically.
