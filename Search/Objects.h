#pragma once
#include "Core/Core.h"

struct Intensity
{
	int depth;
	Confidence certainty;

	Intensity(int depth, Confidence certainty = Confidence::Certain()) noexcept : depth(depth), certainty(certainty) {}

	//static Intensity Certain(int depth) noexcept { return { depth, Confidence::Certain() }; }
	static Intensity Exact(const Position& pos) noexcept { return { pos.EmptyCount() }; }

	bool operator==(const Intensity&) const noexcept = default;
	bool operator!=(const Intensity&) const noexcept = default;
	auto operator<=>(const Intensity&) const noexcept = default;
	Intensity operator+(int d) const noexcept { return { depth + d, certainty }; }
	Intensity operator-(int d) const noexcept { return { depth - d, certainty }; }

	bool IsCertain() const noexcept { return certainty.IsCertain(); }
	bool IsExact(const Position& pos) const noexcept { return *this == Exact(pos); }
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
	//	static Request Certain(int depth, OpenInterval window = OpenInterval::Whole()) noexcept { return { Intensity::Certain(depth), window }; }
	//	static Request Exact(const Position& pos) noexcept { return Certain(pos.EmptyCount()); }
	//
	//	int depth() const noexcept { return intensity.depth; }
	//	Confidence certainty() const noexcept { return intensity.certainty; }
	//
	//	Request operator-() const noexcept { return { intensity.depth, intensity.certainty, -window }; }
	//	operator OpenInterval() const noexcept { return window; }
	//
	//	bool IsCertain() const noexcept { return intensity.IsCertain(); }
	//	bool IsExact(const Position& pos) const noexcept { return intensity.IsExact(pos); }
	//};

	struct Result
	{
		Intensity intensity;// { -1, Confidence(0) };
		ClosedInterval window;// = ClosedInterval::Whole();

		Result(const Intensity& i, const ClosedInterval& window) noexcept : intensity(i), window(window) {}
		Result(int depth, const ClosedInterval& window) noexcept : intensity({ depth }), window(window) {}
		Result(const Intensity& i, int score) noexcept : intensity(i), window({ score, score }) {}

		static Result Exact(const Position& pos, int score) noexcept { return { pos.EmptyCount(), score }; }

		static Result FailHigh(const Intensity& i, int score) noexcept { return { i, {score, max_score} }; }
		static Result FailHigh(const Position& pos, int score) noexcept { return FailHigh(pos.EmptyCount(), score); }
		void FailedHigh() noexcept { window.TryIncreaseUpper(max_score); }
		//static Result FailHigh(int depth, int score) noexcept { return { { depth }, {score, max_score} }; }

		static Result FailLow(const Intensity& i, int score) noexcept { return { i, {min_score, score} }; }
		static Result FailLow(const Position& pos, int score) noexcept { return FailLow(pos.EmptyCount(), score); }
		void FailedLow() noexcept { window.TryDecreaseLower(min_score); }
		//static Result FailLow(int depth, int score) noexcept { return { { depth }, {min_score, score} }; }

		//static Result CertainFailSoft(const Intensity&, int depth, int score) noexcept;

		//static Result ExactFailHard(const Request&, const Position&, int score) noexcept;
		//static Result ExactFailSoft(const Request&, const Position&, int score) noexcept;

		int lower() const { return window.lower(); }
		int upper() const { return window.upper(); }
		int depth() const noexcept { return intensity.depth; }
		int Score() const noexcept { 
			if (window.IsSingleton())
				return window.lower();
			if (window.lower() == min_score)
				return window.upper();
			return window.lower();
		}

		bool operator==(const Result&) const noexcept = default;
		bool operator!=(const Result&) const noexcept = default;

		Result operator-() const noexcept { return { intensity, -window }; }
		Result operator+(int i) const noexcept { return { { intensity.depth + i, intensity.certainty }, window }; }

		bool Satisfies(const Intensity& i, const OpenInterval& request) const noexcept { return (intensity >= i) and (window > request); }

		bool IsExact(const Position& pos) const noexcept { return intensity.IsExact(pos) && window.IsSingleton(); }
		//bool IsFailHigh() const noexcept { return window.upper() == max_score; }
		//bool IsFailLow() const noexcept { return window.lower() == min_score; }
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
