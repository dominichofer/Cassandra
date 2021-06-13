#pragma once
#include "Core/Core.h"

namespace Search
{
	class Intensity
	{
	public:
		int depth;
		Confidence certainty;

		Intensity(int depth, Confidence certainty) noexcept : depth(depth), certainty(certainty) {}

		[[nodiscard]] static Intensity Certain(int depth) noexcept { return { depth, Confidence::Certain() }; }
		[[nodiscard]] static Intensity Exact(const Position& pos) noexcept { return Certain(pos.EmptyCount()); }

		[[nodiscard]] bool operator==(const Intensity&) const noexcept = default;
		[[nodiscard]] bool operator!=(const Intensity&) const noexcept = default;
		[[nodiscard]] auto operator<=>(const Intensity&) const noexcept = default;

		[[nodiscard]] bool IsCertain() const noexcept { return certainty.IsCertain(); }
		[[nodiscard]] bool IsExact(const Position& pos) const noexcept { return *this == Exact(pos); }
	};

	class Request
	{
	public:
		Intensity intensity;
		OpenInterval window = OpenInterval::Whole();

		Request(Intensity intensity, OpenInterval window) noexcept : intensity(intensity), window(window) {}
		Request(Intensity intensity) noexcept : intensity(intensity) {}
		Request(int depth, Confidence certainty, OpenInterval window) noexcept : intensity(depth, certainty), window(window) {}
		Request(int depth, Confidence certainty) noexcept : intensity(depth, certainty) {}

		[[nodiscard]] static Request Certain(int depth, OpenInterval window = OpenInterval::Whole()) noexcept { return { Intensity::Certain(depth), window }; }
		[[nodiscard]] static Request Exact(const Position& pos) noexcept { return Certain(pos.EmptyCount()); }

		[[nodiscard]] int depth() const noexcept { return intensity.depth; }
		[[nodiscard]] Confidence certainty() const noexcept { return intensity.certainty; }

		[[nodiscard]] Request operator-() const noexcept { return { intensity.depth, intensity.certainty, -window }; }
		[[nodiscard]] operator OpenInterval() const noexcept { return window; }

		[[nodiscard]] bool IsCertain() const noexcept { return intensity.IsCertain(); }
		[[nodiscard]] bool IsExact(const Position& pos) const noexcept { return intensity.IsExact(pos); }
	};

	struct Result
	{
		Intensity intensity{ -1, Confidence(0) };
		ClosedInterval window = ClosedInterval::Whole();

		[[nodiscard]] static Result FoundScore(const Intensity&, int score) noexcept;
		[[nodiscard]] static Result FailHigh(const Intensity&, int score) noexcept;
		[[nodiscard]] static Result FailLow(const Intensity&, int score) noexcept;

		[[nodiscard]] static Result FailHigh(const Result&) noexcept;
		[[nodiscard]] static Result FailLow(const Result&) noexcept;

		[[nodiscard]] static Result Certain(int depth, int score) noexcept;
		[[nodiscard]] static Result Exact(const Position& pos, int score) noexcept;

		[[nodiscard]] static Result CertainFailHigh(int depth, int score) noexcept;
		[[nodiscard]] static Result CertainFailLow(int depth, int score) noexcept;
		[[nodiscard]] static Result CertainFailSoft(const Request&, int depth, int score) noexcept;

		[[nodiscard]] static Result ExactFailHigh(const Position&, int score) noexcept;
		[[nodiscard]] static Result ExactFailLow(const Position&, int score) noexcept;
		[[nodiscard]] static Result ExactFailHard(const Request&, const Position&, int score) noexcept;
		[[nodiscard]] static Result ExactFailSoft(const Request&, const Position&, int score) noexcept;

		[[nodiscard]] bool operator==(const Result&) const noexcept = default;
		[[nodiscard]] bool operator!=(const Result&) const noexcept = default;

		[[nodiscard]] Result operator-() const noexcept { return { intensity, -window }; }
		[[nodiscard]] Result operator+(int i) const noexcept { return { { intensity.depth + i, intensity.certainty }, window }; }

		[[nodiscard]] bool operator>(const Request& request) const noexcept;

		[[nodiscard]] int depth() const noexcept { return intensity.depth; }
		[[nodiscard]] int Score() const noexcept { assert(window.IsSingleton());  return window.lower(); }

		[[nodiscard]] bool IsExact(const Position& pos) const noexcept { return intensity.IsExact(pos) && window.IsSingleton(); }
		[[nodiscard]] bool IsFailHigh() const noexcept { return window.upper() == max_score; }
		[[nodiscard]] bool IsFailLow() const noexcept { return window.lower() == min_score; }
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

inline Search::Intensity min(const Search::Intensity& l, const Search::Intensity& r)
{
	return { std::min(l.depth, r.depth), std::min(l.certainty, r.certainty) };
}

inline std::string to_string(const Search::Intensity& intensity)
{
	std::string ret = std::to_string(intensity.depth);
	if (intensity.certainty != Confidence::Certain())
		ret += " " + to_string(intensity.certainty);
	return ret;
}
inline std::ostream& operator<<(std::ostream& os, const Search::Intensity& intensity) { return os << to_string(intensity); }

inline std::string to_string(const Search::Request& request)
{
	return to_string(request.window) + " " + to_string(request.intensity);
}
inline std::ostream& operator<<(std::ostream& os, const Search::Request& request) { return os << to_string(request); }

inline std::string to_string(const Search::Result& result)
{
	return to_string(result.window) + " " + to_string(result.intensity);
}
inline std::ostream& operator<<(std::ostream& os, const Search::Result& result) { return os << to_string(result); }
