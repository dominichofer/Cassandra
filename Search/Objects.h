#pragma once
#include "Core/Core.h"

struct Intensity
{
	int depth;
	Confidence certainty;

	Intensity(int depth, Confidence certainty = Confidence::Certain()) noexcept : depth(depth), certainty(certainty) {}

	//[[nodiscard]] static Intensity Certain(int depth) noexcept { return { depth, Confidence::Certain() }; }
	[[nodiscard]] static Intensity Exact(const Position& pos) noexcept { return { pos.EmptyCount() }; }

	[[nodiscard]] bool operator==(const Intensity&) const noexcept = default;
	[[nodiscard]] bool operator!=(const Intensity&) const noexcept = default;
	[[nodiscard]] auto operator<=>(const Intensity&) const noexcept = default;
	[[nodiscard]] Intensity operator+(int d) const noexcept { return { depth + d, certainty }; }
	[[nodiscard]] Intensity operator-(int d) const noexcept { return { depth - d, certainty }; }

	[[nodiscard]] bool IsCertain() const noexcept { return certainty.IsCertain(); }
	[[nodiscard]] bool IsExact(const Position& pos) const noexcept { return *this == Exact(pos); }
};

namespace Search
{
	//class Request
	//{
	//public:
	//	Intensity intensity;
	//	OpenInterval window = OpenInterval::Whole();
	//
	//	Request(Intensity intensity, OpenInterval window) noexcept : intensity(intensity), window(window) {}
	//	Request(Intensity intensity) noexcept : intensity(intensity) {}
	//	Request(int depth, Confidence certainty, OpenInterval window) noexcept : intensity(depth, certainty), window(window) {}
	//	Request(int depth, Confidence certainty) noexcept : intensity(depth, certainty) {}
	//
	//	[[nodiscard]] static Request Certain(int depth, OpenInterval window = OpenInterval::Whole()) noexcept { return { Intensity::Certain(depth), window }; }
	//	[[nodiscard]] static Request Exact(const Position& pos) noexcept { return Certain(pos.EmptyCount()); }
	//
	//	[[nodiscard]] int depth() const noexcept { return intensity.depth; }
	//	[[nodiscard]] Confidence certainty() const noexcept { return intensity.certainty; }
	//
	//	[[nodiscard]] Request operator-() const noexcept { return { intensity.depth, intensity.certainty, -window }; }
	//	[[nodiscard]] operator OpenInterval() const noexcept { return window; }
	//
	//	[[nodiscard]] bool IsCertain() const noexcept { return intensity.IsCertain(); }
	//	[[nodiscard]] bool IsExact(const Position& pos) const noexcept { return intensity.IsExact(pos); }
	//};

	struct Result
	{
		Intensity intensity;// { -1, Confidence(0) };
		ClosedInterval window;// = ClosedInterval::Whole();

		Result(const Intensity& i, const ClosedInterval& window) noexcept : intensity(i), window(window) {}
		Result(int depth, const ClosedInterval& window) noexcept : intensity({ depth }), window(window) {}
		Result(const Intensity& i, int score) noexcept : intensity(i), window({ score, score }) {}

		[[nodiscard]] static Result Exact(const Position& pos, int score) noexcept { return { pos.EmptyCount(), score }; }

		[[nodiscard]] static Result FailHigh(const Intensity& i, int score) noexcept { return { i, {score, max_score} }; }
		[[nodiscard]] static Result FailHigh(const Position& pos, int score) noexcept { return FailHigh(pos.EmptyCount(), score); }
		[[nodiscard]] void FailedHigh() noexcept { window.TryIncreaseUpper(max_score); }
		//[[nodiscard]] static Result FailHigh(int depth, int score) noexcept { return { { depth }, {score, max_score} }; }

		[[nodiscard]] static Result FailLow(const Intensity& i, int score) noexcept { return { i, {min_score, score} }; }
		[[nodiscard]] static Result FailLow(const Position& pos, int score) noexcept { return FailLow(pos.EmptyCount(), score); }
		[[nodiscard]] void FailedLow() noexcept { window.TryDecreaseLower(min_score); }
		//[[nodiscard]] static Result FailLow(int depth, int score) noexcept { return { { depth }, {min_score, score} }; }

		//[[nodiscard]] static Result CertainFailSoft(const Intensity&, int depth, int score) noexcept;

		//[[nodiscard]] static Result ExactFailHard(const Request&, const Position&, int score) noexcept;
		//[[nodiscard]] static Result ExactFailSoft(const Request&, const Position&, int score) noexcept;

		[[nodiscard]] int lower() const { return window.lower(); }
		[[nodiscard]] int upper() const { return window.upper(); }
		[[nodiscard]] int depth() const noexcept { return intensity.depth; }
		[[nodiscard]] int Score() const noexcept { 
			if (window.IsSingleton())
				return window.lower();
			if (window.lower() == min_score)
				return window.upper();
			return window.lower();
		}

		[[nodiscard]] bool operator==(const Result&) const noexcept = default;
		[[nodiscard]] bool operator!=(const Result&) const noexcept = default;

		[[nodiscard]] Result operator-() const noexcept { return { intensity, -window }; }
		[[nodiscard]] Result operator+(int i) const noexcept { return { { intensity.depth + i, intensity.certainty }, window }; }

		[[nodiscard]] bool Satisfies(const Intensity& i, const OpenInterval& request) const noexcept { return (intensity >= i) and (window > request); }

		[[nodiscard]] bool IsExact(const Position& pos) const noexcept { return intensity.IsExact(pos) && window.IsSingleton(); }
		//[[nodiscard]] bool IsFailHigh() const noexcept { return window.upper() == max_score; }
		//[[nodiscard]] bool IsFailLow() const noexcept { return window.lower() == min_score; }
	};

	struct Findings
	{
		int best_score = -inf_score;
		Field best_move = Field::invalid;
		Field best_move_2 = Field::invalid;
		Intensity lowest_intensity{ 99, Confidence::Certain() };

		void Add(const Result& result, Field move) noexcept;
	};
}

inline Intensity min(const Intensity& l, const Intensity& r)
{
	return { std::min(l.depth, r.depth), std::min(l.certainty, r.certainty) };
}

inline std::string to_string(const Intensity& i)
{
	std::string ret = std::to_string(i.depth);
	if (not i.IsCertain())
		ret += " " + to_string(i.certainty);
	return ret;
}
inline std::ostream& operator<<(std::ostream& os, const Intensity& intensity) { return os << to_string(intensity); }

inline std::string to_string(const Search::Result& result)
{
	return to_string(result.window) + " " + to_string(result.intensity);
}
inline std::ostream& operator<<(std::ostream& os, const Search::Result& result) { return os << to_string(result); }
