#pragma once

namespace pattern
{
	inline const BitBoard L0 = BitBoard::HorizontalLine(0);
	inline const BitBoard L1 = BitBoard::HorizontalLine(1);
	inline const BitBoard L2 = BitBoard::HorizontalLine(2);
	inline const BitBoard L3 = BitBoard::HorizontalLine(3);
	inline const BitBoard D4 = BitBoard::CodiagonalLine(4);
	inline const BitBoard D5 = BitBoard::CodiagonalLine(3);
	inline const BitBoard D6 = BitBoard::CodiagonalLine(2);
	inline const BitBoard D7 = BitBoard::CodiagonalLine(1);
	inline const BitBoard D8 = BitBoard::CodiagonalLine(0);
	inline const BitBoard L02X =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- # - - - - # -"
		"# # # # # # # #"_BitBoard;
	inline const BitBoard B4 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - # # # #"
		"- - - - # # # #"_BitBoard;
	inline const BitBoard B5 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - # # # # #"
		"- - - # # # # #"_BitBoard;
	inline const BitBoard B6 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - # # # # # #"
		"- - # # # # # #"_BitBoard;
	inline const BitBoard Q3 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - # # #"
		"- - - - - # # #"
		"- - - - - # # #"_BitBoard;
	inline const BitBoard C3p1 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - # #"
		"- - - - # # # #"_BitBoard;
	inline const BitBoard C3p2 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - # #"
		"- - - # # # # #"_BitBoard;
	inline const BitBoard Ep =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - # # # # - -"
		"# - # # # # - #"_BitBoard;
	inline const BitBoard Comet =
		"# - - - - - - -"
		"- # - - - - - -"
		"- - # - - - - -"
		"- - - # - - - -"
		"- - - - # - - -"
		"- - - - - # - -"
		"- - - - - - # #"
		"- - - - - - # #"_BitBoard;

	inline const std::vector<BitBoard> logistello{ // From "M. Buro, Logistello: A Strong Learning Othello Program , 19th Annual Conference Gesellschaft für Klassifikation e.V. (1995), Basel"
		D4, D5, D6, D7, D8, 
		L02X, L1, L2, L3,
		Q3, B5
	};

	inline const std::vector<BitBoard> edax{ // From https://github.com/abulmo/edax-reversi/blob/v4.4/src/eval.c#L38-L98
		D4, D5, D6, D7, D8,
		L02X, L1, L2, L3,
		Q3, C3p2, Ep
	};

	inline const std::vector<BitBoard> cassandra{
		D4, D5, D6, D7, Comet,
		L02X, L1, L2, L3,
		Q3, C3p2, B5
	};
}