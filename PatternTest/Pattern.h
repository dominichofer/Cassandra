#pragma once
#include "pch.h"

const uint64_t pattern_a =
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - # #"_pattern;

const uint64_t pattern_c =
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"# - - - - - - -"
	"- # - - - - - -"_pattern;

const uint64_t pattern_d =
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - #"
	"- - - - - - # -"_pattern;

const uint64_t pattern_h =
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"# - - - - - - #"_pattern;

const uint64_t pattern_v =
	"- - - - - - - #"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - #"_pattern;

const uint64_t pattern_vh =
	"- # - - - - # -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- # - - - - # -"_pattern;

const uint64_t pattern_dc =
	"- # - - - - - -"
	"# - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - #"
	"- - - - - - # -"_pattern;

const uint64_t pattern_vhdc =
	"# - - - - - - #"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"# - - - - - - #"_pattern;

const std::vector<uint64_t> pattern_mix{
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
		FlippedCodiagonal(t),
		FlippedDiagonal(t),
		FlippedHorizontal(t),
		FlippedVertical(t),
		FlippedCodiagonal(FlippedHorizontal(t)),
		FlippedDiagonal(FlippedHorizontal(t)),
		FlippedVertical(FlippedHorizontal(t))
	};
}