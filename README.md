# Cassandra
Towards solving Reversi.

# Base
Implements missing parts of the standard library in nvcc:
	countl_zero(uint64) -> int
	countl_one(uint64) -> int
	countr_zero(uint64) -> int
	countr_one(uint64) -> int
	popcount(uint64) -> int
	to_underlying(value) -> auto

Extends intrinsics with:
	not_si256(__m256i) -> __m256i
	neg_epi64(__m256i) -> __m256i
	reduce_or(__m256i) -> uint64
	
Implements bit manipulations:
	GetLSB(uint64) -> uint64
	ClearLSB(uint64&)
	BExtr(uint64, uint start, uint len) -> uint64
	PDep(uint64, uint64 mask) -> uint64
	PExt(uint64, uint64 mask) -> uint64
	BSwap(uint64) -> uint64
	
Extends the standard library with algorithms:
	Sample(int size, pool, seed) -> auto
	join(separator, iterable, projection) -> auto

Implements string formaters:
	class Table:
		Table(string title, string format)
		PrintHeader()
		PrintSeparator()
		PrintRow(content)
		
	HH_MM_SS(duration) -> string
	ShortTimeString(duration) -> string	
	MetricPrefix(int magnitude) -> string

# Board
Implements operations on bitboards:
	FlippedCodiagonal(uint64) -> uint64
	FlippedDiagonal(uint64) -> uint64
	FlippedHorizontal(uint64) -> uint64
	FlippedVertical(uint64) -> uint64

	IsCodiagonallySymmetric(uint64) -> bool
	IsDiagonallySymmetric(uint64) -> bool
	IsHorizontallySymmetric(uint64) -> bool
	IsVerticallySymmetric(uint64) -> bool

	ParityQuadrants(uint64) -> uint64
	EightNeighboursAndSelf(uint64) -> uint64
	
	MultiLine(uint64) -> string

Classes to represent a reversi position and the fields of the board

	class Position:
		Position(uint64 player, uint64 opponent)
		FromString(string) -> pos
		Start() -> Position
		Player() -> uint64
		Opponent() -> uint64
		Discs() -> uint64
		Empties() -> uint64
		EmptyCount() -> int
		
	enum Field:
		H8 ... A1, PS
	
with string serializers and deserializers

	SingleLine(pos) -> string
	MultiLine(pos) -> string
	to_string(pos) -> string
	
	to_string(Field) -> string
	FieldFromString(string) -> Field
	
Functions to get possible moves

	class Moves:
		Moves(uint64)
		empty() -> bool
		contains(Field) -> bool
		erase(Field)
		size() -> std::size_t
		front() -> Field
		pop_front()
		begin() -> iterator
		end() -> iterator

	PossibleMoves(pos) -> Moves

Functions to play them

	Play(pos, move) -> pos
	PlayPass(pos) -> pos
	PlayOrPass(pos, move) -> pos
	Play(pos, move, flips) -> pos
	Flips(pos, move) -> uint64
	
Functions to analyse stable stones

	StableEdges(pos) -> uint64
	StableStonesOpponent(pos) -> uint64
	
Functions to count how many stones the last move of a game flips

	CountLastFlip(pos, open_field) -> int

Functions to generate children of positions

	Children(pos, int plies, bool pass_is_a_ply) -> generator
	Children(pos, int empty_count) -> generator
	UniqueChildren(pos, int empty_count) -> set

Various other functions to work with the classes
	
	FlippedCodiagonal(pos) -> pos
	FlippedDiagonal(pos) -> pos
	FlippedHorizontal(pos) -> pos
	FlippedVertical(pos) -> pos
	FlippedToUnique(pos) -> pos
	operator""_pos(string) -> pos
	Bit(Field) -> uint64

and random position generators

	class RandomPositionGenerator:
		RandomPositionGenerator(seed)
		operator() -> pos
		
	RandomPosition(seed) -> pos
	RandomPositionWithEmptyCount(int empty_count, seed) -> pos

# Game
Implements classes to represent scores and confidence levels

	class Score = int
		FromString(string) -> Score
		
	class ConfidenceLevel = float
		IsInfinit() -> bool
	
with string serializers and deserializers

	to_string(Score) -> string
	to_string(int depth, confidence_level) -> string
	DepthClFromString(string) -> tuple<Score, float>
	
and some constants

	Score min_score
	Score max_score
	Score inf_score
	Score undefined_score
	ConfidenceLevel inf
	
as well as a funtion to evaluate the score of a terminal position and a stability based max score

	EndScore(pos) -> Score
	StabilityBasedMaxScore(pos) -> Score
	
Implements intervals:
	class OpenInterval:
		int lower
		int upper
		
	class ClosedInterval:
		int lower
		int upper

Implements classes to represent scored positions, scored and unscored games

	class ScoredPosition:
		pos
		Score scores
		FromString(string) -> ScoredPosition
	
	class Game:
		Game(start_pos, moves)
		FromString(string) -> Game
		StartPosition() -> pos
		Moves() -> vector
		Positions() -> positions
		
	class ScoredGame:
		Game game
		vector scores
		FromString(string) -> ScoredGame
		
and functions to stringify them

	to_string(ScoredPosition) -> string
	to_string(Game) -> string
	to_string(ScoredGame) -> string
	
Implements an abstract player and a random player

	Interface Player:
		ChooseMove(pos) -> move
		
	class RandomPlayer:
		RandomPlayer(seed)
		ChooseMove(pos) -> move

	PlayedGame(first_player, second_player, start_pos) -> Game
	PlayedGamesFrom(first_player, second_player, start_positions) -> vector
		
Implements transformations

	Positions(scored_positions) -> vector
	Positions(games) -> vector
	Positions(scored_games) -> vector
	Scores(scored_positions) -> vector
	Scores(scored_games) -> vector
	ScoredPositions(scored_games) -> vector
	ScoredPositions(positions, scores) -> vector

and filters

	EmptyCountFiltered(positions, int min_empty_count, int max_empty_count) -> vector
	EmptyCountFiltered(positions, int empty_count) -> vector
	EmptyCountFiltered(scored_positions, int min_empty_count, int max_empty_count) -> vector
	EmptyCountFiltered(scored_positions, int empty_count) -> vector
	
# Search
Provides an interface for Estimators, which estimate the score of a position and its accuracy.

	Interface Estimator:
		Score(pos) -> float
		Accuracy(int empty_count, int small_depth, int big_depth) -> float
	
Implements classes to track the status of a search and the result of a search

	class Status
		Status(int alpha)
		Update(result, move)
		GetResult() -> result
		
	class Result:
		FailLow(score, depth, confidence_level, best_move) -> Result
		Exact(score, depth, confidence_level, best_move) -> Result
		FailHigh(score, depth, confidence_level, best_move) -> Result
		EndScore(pos) -> Result
		operator-() -> Result
		IsFailLow() -> bool
		IsExact() -> bool
		IsFailHigh() -> bool
		Window() -> ClosedInterval
		BetaCut(move) -> Result
		
	to_string(result) -> string
	
A class that provides sorted possible moves

	class MoveSorter:
		MoveSorter(transposition_table, search_algorithm)
		
Search algorithms

	class Algorithm:
		virtual Eval(int guess, pos, window, depth, confidence_level) -> Result
		virtual Eval(pos, window, depth, confidence_level) -> Result
		        Eval(pos, depth, confidence_level) -> Result
				Eval(pos, depth) -> Result
				Eval(pos) -> Result
		virtual Nodes() -> uint64
		virtual Clear()
		
	class NegaMax:
		Eval_N(pos) -> int
		Eval_3(pos, move, move, move) -> int
		Eval_2(pos, move, move) -> int
		Eval_1(pos, move) -> int
		Eval_0(pos) -> int
		
	class AlphaBeta:
		Eval_N(pos, window) -> int
		Eval_N(pos, window) -> int
		Eval_3(pos, window, move, move, move) -> int
		Eval_2(pos, window, move, move) -> int
		
	class PVS:
		PVS(transposition_table, estimator) -> result
		
	class MTD:
		MTD(search_algorithm)
		
	class IDAB:
		IDAB(search_algorithm)
		
A class which solves positions

	class Solver
		Solver(Algorithm&, bool silent, int threads)
		
		Solve(positions, window, depth, confidence_level) -> scores
		Solve(positions) -> scores
		Solve(scored_positions, window, depth, confidence_level) -> scores
		Solve(scored_positions) -> scores
		Clear()
		PrintHeader()
		PrintSummary()

# Math
Defines a class to represent a mathematical vector of floats

	using Vector = vector<float>
	
and implements function on it

	+ - * /
	elementwise_multiplication(vector, vector) -> vector
	elementwise_division(vector, vector) -> vector
	inv(vector) -> vector
	dot(vector, vector) -> float
	norm(vector) -> float
	L1_norm(vector) -> float
	L2_norm(vector) -> float
	sum(vector) -> float
	
	to_string(vector) -> string
	
Implements a class to represent a dense matrix

	class Matrix:
		Matrix(rows, cols)
		Id(size) -> matrix
		Rows() -> size
		Cols() -> size
		Row(index) -> span
		(row, col) -> float
		
with functions on it

	+ - * /
	transposed(matrix) -> matrix
	
	to_string(matrix) -> string
	
Implements a class to represent a CSR (=compressed sparse row) matrix.
With fixed number of non-zero elements per row.
With only 1 as element but the matrix can have multiple entries per element.

	class MatrixCSR
		MatrixCSR(elements_per_row, rows, cols)
		Rows() -> size
		Cols() -> size
		Row(index) -> span
		
With functions on it

	*
	transposed(matrix) -> matrix
	JacobiPreconditionerOfATA(matrix) -> vector
	
Implements statistics functions with optional projections

	Average(range) -> float
	Variance(rangen) -> float
	StandardDeviation(range) -> float
	Covariance(range, range) -> float
	Covariance(matrix) -> matrix
	PopulationCovariance(range, range) -> float
	SampleCovariance(range, range) -> float
	Correlation(matrix) -> matrix
	AIC(range, num_parameters) -> float
	BIC(range, num_parameters) -> float
	
Provides an interface for iterative solvers and preconditioners

	Interface IterativeSolver:
		Iterate(num)
		Residuum() -> float
		Error() -> vector
		X() -> vector
	
	Interface Preconditioner:
		apply(vector) -> vector
		revert(vector) -> vector
		
and implements classes for various iterative solvers

	class CG: (Conjugate Gradient method)
		CG(matrix, initial_vector, vector b)

	class PCG: (PreconditionedConjugate Gradient method)
		PCG(matrix, preconditioner, initial_vector, vector b)
		
	class LSQR: (Least Squares QR Method)
		LSQR(matrix, initial_vector, vector b)
		
	class LSMR: (Least Squares Minres Method)
		LSMR(matrix, initial_vector, vector b)
		
and preconditioners

	class DiagonalPreconditioner:
		DiagonalPreconditioner(diagonal_vector)

and helper functions

	decompose(vector) -> unit_vector, lenght
	signum(value) -> float
	GivensRotation(a, b) -> float, float, float
	
	
# Architecture
The dependencies of the components looks like this
	Search -> Game -> Board -> Base
	           ^
	           |
	          IO
			   |
			   v
	Math	  Pattern