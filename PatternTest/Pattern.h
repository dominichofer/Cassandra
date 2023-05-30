#pragma once
#include "pch.h"

const BitBoard pattern_a =
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - # #"_BitBoard;

const BitBoard pattern_c =
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"# - - - - - - -"
	"- # - - - - - -"_BitBoard;

const BitBoard pattern_d =
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - #"
	"- - - - - - # -"_BitBoard;

const BitBoard pattern_h =
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"# - - - - - - #"_BitBoard;

const BitBoard pattern_v =
	"- - - - - - - #"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - #"_BitBoard;

const BitBoard pattern_vh =
	"- # - - - - # -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- # - - - - # -"_BitBoard;

const BitBoard pattern_dc =
	"- # - - - - - -"
	"# - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - #"
	"- - - - - - # -"_BitBoard;

const BitBoard pattern_vhdc =
	"# - - - - - - #"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"# - - - - - - #"_BitBoard;

const std::vector<BitBoard> pattern_mix{
	pattern_a,
	pattern_c,
	pattern_d,
	pattern_h,
	pattern_v,
	pattern_vh,
	pattern_dc,
	pattern_vhdc
};

template <typename T>
std::set<T> SymmetricVariants(const T& t)
{
	return {
		t,
		FlipCodiagonal(t),
		FlipDiagonal(t),
		FlipHorizontal(t),
		FlipVertical(t),
		FlipCodiagonal(FlipHorizontal(t)),
		FlipDiagonal(FlipHorizontal(t)),
		FlipVertical(FlipHorizontal(t))
	};
}