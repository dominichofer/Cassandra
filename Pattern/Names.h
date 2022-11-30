#pragma once

namespace pattern
{
	inline constexpr BitBoard L0 = BitBoard::HorizontalLine(0);
	inline constexpr BitBoard L1 = BitBoard::HorizontalLine(1);
	inline constexpr BitBoard L2 = BitBoard::HorizontalLine(2);
	inline constexpr BitBoard L3 = BitBoard::HorizontalLine(3);
	inline constexpr BitBoard D4 = BitBoard::CodiagonalLine(4);
	inline constexpr BitBoard D5 = BitBoard::CodiagonalLine(3);
	inline constexpr BitBoard D6 = BitBoard::CodiagonalLine(2);
	inline constexpr BitBoard D7 = BitBoard::CodiagonalLine(1);
	inline constexpr BitBoard D8 = BitBoard::CodiagonalLine(0);
	inline constexpr BitBoard L02X =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- # - - - - # -"
		"# # # # # # # #"_BitBoard;
	inline constexpr BitBoard B4 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - # # # #"
		"- - - - # # # #"_BitBoard;
	inline constexpr BitBoard B5 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - # # # # #"
		"- - - # # # # #"_BitBoard;
	inline constexpr BitBoard B6 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - # # # # # #"
		"- - # # # # # #"_BitBoard;
	inline constexpr BitBoard Q3 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - # # #"
		"- - - - - # # #"
		"- - - - - # # #"_BitBoard;
	inline constexpr BitBoard C3p1 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - # #"
		"- - - - # # # #"_BitBoard;
	inline constexpr BitBoard C3p2 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - - #"
		"- - - - - - # #"
		"- - - # # # # #"_BitBoard;
	inline constexpr BitBoard Ep =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - # # # # - -"
		"# - # # # # - #"_BitBoard;
	inline constexpr BitBoard Comet =
		"# - - - - - - -"
		"- # - - - - - -"
		"- - # - - - - -"
		"- - - # - - - -"
		"- - - - # - - -"
		"- - - - - # - -"
		"- - - - - - # #"
		"- - - - - - # #"_BitBoard;

	const std::vector<BitBoard> logistello{ // From "M. Buro, Logistello: A Strong Learning Othello Program , 19th Annual Conference Gesellschaft für Klassifikation e.V. (1995), Basel"
		D4, D5, D6, D7, D8, 
		L02X, L1, L2, L3,
		Q3, B5
	};

	const std::vector<BitBoard> edax{ // From https://github.com/abulmo/edax-reversi/blob/v4.4/src/eval.c#L38-L98
		D4, D5, D6, D7, D8,
		L02X, L1, L2, L3,
		Q3, C3p2, Ep
	};

	const std::vector<BitBoard> cassandra{
		D4, D5, D6, D7, Comet,
		L02X, L1, L2, L3,
		Q3, C3p2, B5
	};
}