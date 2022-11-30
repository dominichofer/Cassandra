#pragma once
#include "Core/Core.h"
#include "Confidence.h"
#include <array>
#include <set>

struct Intensity
{
	int depth;
	Confidence certainty;

	Intensity(int depth, Confidence certainty = Confidence::Certain()) noexcept : depth(depth), certainty(certainty) {}
	static Intensity Exact() noexcept { return { 99 }; }
	static Intensity None() noexcept { return { 0, Confidence::Uncertain() }; }
	static Intensity Limitted(Intensity i, int empty_count) noexcept { return { std::min(i.depth, empty_count), i.certainty }; }

	bool operator==(const Intensity&) const noexcept = default;
	bool operator!=(const Intensity&) const noexcept = default;
	bool operator<(const Intensity& o) const noexcept { return *this <= o and *this != o; }
	bool operator>(const Intensity& o) const noexcept { return *this >= o and *this != o; }
	bool operator<=(const Intensity& o) const noexcept { return depth <= o.depth and certainty <= o.certainty; }
	bool operator>=(const Intensity& o) const noexcept { return depth >= o.depth and certainty >= o.certainty; }
	Intensity operator+(int value) const noexcept { return { depth + value, certainty }; }
	Intensity operator-(int value) const noexcept { return { depth - value, certainty }; }
	Intensity operator/(int value) const noexcept { return { depth / value, certainty }; }

	bool IsCertain() const noexcept { return certainty.IsCertain(); }
	bool IsExact() const noexcept { return *this >= Exact(); }
	bool FullDepth() const noexcept { return depth >= Exact().depth; }
};

std::string to_string(const Intensity&);

template <> struct fmt::formatter<Intensity> : to_string_formatter<Intensity> {};


class IntensityTable
{
	std::array<std::set<Intensity>, 65> table;
public:
	IntensityTable() = default;

	static IntensityTable ExactTill(int empty_count, Intensity then);
	static IntensityTable AllDepthTill(int empty_count, Intensity then);
	static IntensityTable AllDepthTill(int empty_count, std::ranges::range auto&& then)
	{
		IntensityTable ret;
		for (int e = 0; e <= empty_count; e++)
		{
			for (int d = 0; d < e; d++)
				ret.insert(e, d);
			ret.insert(e, Intensity::Exact());
		}
		for (int e = empty_count + 1; e <= 64; e++)
			ret.insert(e, then);
		return ret;
	}

	void insert(const Intensity&);
	void insert(int empty_count, const Intensity&);
	void insert(int empty_count, std::ranges::range auto&& values) {
		for (const Intensity& i : values)
			insert(empty_count, i);
	}

	const std::set<Intensity>& Intensities(int empty_count) const { return table[empty_count]; }
};
