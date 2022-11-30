#pragma once
#include "Core/Core.h"
#include "Intensity.h"
#include "Interval.h"

// This is returned from a search if a best move is needed
//struct ScoreMove
//{
//	int score = -inf_score;
//	Field move = Field::invalid;
//
//	ScoreMove() = default;
//	ScoreMove(int score) noexcept : score(score) {}
//	ScoreMove(int score, Field move) noexcept : score(score), move(move) {}
//
//	operator int() const noexcept { return score; }
//	ScoreMove operator-() const noexcept { return { -score, move }; }
//
//	void ImproveWith(int score, Field move);
//};

// This is returned from a search internally if prob-cut is used
struct IntensityScore
{
	Intensity intensity;
	int score;

	//bool operator==(const IntensityScore&) const noexcept = default;
	//bool operator!=(const IntensityScore&) const noexcept = default;

	IntensityScore(Intensity intensity, int score) noexcept : intensity(intensity), score(score) {}
	IntensityScore(int score) noexcept : intensity(Intensity::Exact()), score(score) {}

	operator int() const noexcept { return score; }

	IntensityScore operator-() const noexcept { return { intensity, -score }; }
	IntensityScore operator+(int depth) const noexcept { return { intensity + depth, score }; }
	IntensityScore operator-(int depth) const noexcept { return { intensity - depth, score }; }
};

struct ContextualResult
{
	Intensity intensity = Intensity::Exact();
	int score = -inf_score;
	Field move = Field::invalid;

	ContextualResult() noexcept = default;
	ContextualResult(int score) noexcept : score(score) {}
	ContextualResult(int score, Field move) noexcept : score(score), move(move) {}
	ContextualResult(Intensity intensity, int score) noexcept : intensity(intensity), score(score) {}
	ContextualResult(Intensity intensity, int score, Field move) noexcept : intensity(intensity), score(score), move(move) {}

	operator int() const noexcept { return score; }

	ContextualResult operator-() const noexcept { return { intensity, -score, move }; }
	ContextualResult operator+(int depth) const noexcept { return { intensity + depth, score, move }; }
	ContextualResult operator-(int depth) const noexcept { return { intensity - depth, score, move }; }

	void ImproveWith(int score, Field move)
	{
		if (score > this->score)
		{
			this->score = score;
			this->move = move;
		}
	}
};

struct ContextfreeResult
{
	Intensity intensity;
	ClosedInterval window;
	Field move;

	ContextfreeResult(Intensity intensity, ClosedInterval window, Field move) noexcept : intensity(intensity), window(window), move(move) {}
	ContextfreeResult(ContextualResult result, OpenInterval search_window) noexcept : intensity(result.intensity), window(ClosedInterval{ result.score, result.score }), move(result.move)
	{
		if (result.score > search_window)
			window = { result.score, max_score };
		else if (result.score < search_window)
			window = { min_score, result.score };
	}
};

// This is used in a search to collect relevant information
class Findings
{
	const Intensity search_intensity; // for context
	const OpenInterval search_window; // for context
	ContextualResult current_best{ Intensity::Exact(), -inf_score, Field::invalid };
public:
	Findings(Intensity search_intensity, OpenInterval search_window) noexcept
		: search_intensity(search_intensity), search_window(search_window) {}

	operator ContextualResult() const noexcept { return current_best + 1; }
	operator ContextfreeResult() const noexcept { return { current_best + 1, search_window }; }

	Field Move() const noexcept { return current_best.move; }
	void SetMove(Field move) noexcept { current_best.move = move; }

	OpenInterval NextFullWindow() { int alpha = std::max(current_best.score, search_window.Lower()); return { alpha, search_window.Upper() }; }
	OpenInterval NextZeroWindow() { int alpha = std::max(current_best.score, search_window.Lower()); return { alpha, alpha + 1 }; }

	bool Add(ContextfreeResult novum)
	{
		if (current_best.move == Field::invalid)
			current_best.move = novum.move; // any move is better than no move
		if (novum.intensity >= search_intensity)
		{
			if (novum.window.IsSingleton())
			{
				current_best = { novum.intensity, novum.window.Lower(), novum.move };
				return true;
			}
			if (novum.window > search_window)
			{
				current_best = { novum.intensity, novum.window.Lower(), novum.move };
				return true;
			}
			if (novum.window < search_window)
			{
				current_best = { novum.intensity, novum.window.Upper(), novum.move };
				return true;
			}
			//current_best = { novum.intensity, novum.window.Lower(), novum.move }; // CONTROVERSAL
			current_best = { current_best.intensity, novum.window.Lower(), novum.move }; // CONTROVERSAL
		}
		return false;
	}
	bool AddOption(ContextualResult option, Field move)
	{
		//assert(option.intensity >= search_intensity);

		if (option.score > search_window) // beta cut
		{
			current_best = { option.intensity, option.score, move };
			return true;
		}
		if (option.score > current_best.score)
			current_best = { current_best.intensity, option.score, move };
		if (option.intensity < current_best.intensity)
			current_best.intensity = option.intensity;
		return false;
	}
};

class Algorithm
{
public:
	// A requested certainty of 90% means, that any node can be cut (fail high, fail low) if a shallow search
	// showed that there's a 90% chance that the original search will result in a cut anyway.
	virtual ContextualResult Eval(const Position&, Intensity, OpenInterval) = 0;
	ContextualResult Eval(const Position&, Intensity);
	ContextualResult Eval(const Position&, OpenInterval);
	ContextualResult Eval(const Position&);

	virtual uint64 Nodes() const = 0;
	virtual void clear() {}
};

//class XmlTag
//{
//	struct Property {
//		std::string name, value;
//		std::string to_string() const { return name + "='" + value + "'"; }
//	};
//	std::string to_string(const Property& prop) const { return prop.to_string(); }

//	std::string name;
//	std::vector<Property> props;

//	std::string content() const {
//		std::string str = name;
//		for (const auto& prop : props)
//			str += " " + to_string(prop);
//		return str;
//	}
//public:
//	XmlTag(std::string name) noexcept : name(std::move(name)) {}

//	void Add(std::string name, std::string value) {
//		props.emplace_back(std::move(name), std::move(value));
//	}
//	std::string start_tag() const { return "<" + content() + ">"; }
//	std::string end_tag() const { return "</" + name + ">"; }
//	std::string empty_tag() const { return "<" + content() + " />"; }
//};

//class Node
//{
//public:
//	virtual std::string to_string(int indentations = 0) const = 0;
//};

//class Cut_Node final : public Node
//{
//	std::string reason;
//	Result result;
//public:
//	Cut_Node(std::string reason, Result result) noexcept : reason(std::move(reason)), result(std::move(result)) {}
//	std::string to_string(int indentations = 0) const override {
//		XmlTag tag("Cut");
//		tag.Add("reason", reason);
//		tag.Add("result", ::to_string(result));
//		return std::string(indentations, '\t') + tag.empty_tag();
//	}
//};

//class TT_Node final : public Node
//{
//	ContextfreeResult info;
//public:
//	TT_Node(ContextfreeResult info) noexcept : info(std::move(info)) {}
//	std::string to_string(int indentations = 0) const override {
//		XmlTag tag("TT_Node");
//		tag.Add("best_move", ::to_string(info.best_move));
//		tag.Add("result", ::to_string(info.result));
//		return std::string(indentations, '\t') + tag.empty_tag();
//	}
//};

//class Search_Node final : public Node
//{
//protected:
//	Field move = Field::invalid;
//	std::string name;
//	Position pos;
//	Request request;
//	std::vector<std::unique_ptr<Node>> children;
//	std::optional<Result> result;

//public:
//	Search_Node(std::string name, const Position& pos, const Request& request) noexcept : name(std::move(name)), pos(pos), request(request) {}
//	Search_Node() = default;
//	Search_Node(const Search_Node&) = delete;
//	Search_Node(Search_Node&&) = default;
//	Search_Node& operator=(const Search_Node&) = delete;
//	Search_Node& operator=(Search_Node&&) = default;
//	~Search_Node() = default;

//	void Add(std::unique_ptr<Node>&& child) { children.push_back(std::move(child)); }

//	void Add(Field move, Result result) {
//		dynamic_cast<Search_Node&>(*children.back()).move = move;
//		dynamic_cast<Search_Node&>(*children.back()).result = std::move(result);
//	}

//	std::string to_string(int indentations = 0) const {
//		using std::to_string;
//		using ::to_string;
//		XmlTag tag(name);
//		tag.Add("request", to_string(request));
//		if (result.has_value())
//			tag.Add("result", to_string(result.value()));
//		tag.Add("move", to_string(move));
//		tag.Add("empty_count", to_string(pos.EmptyCount()));
//		tag.Add("pos", to_string(pos));

//		std::string str = std::string(indentations, '\t') + tag.start_tag() + '\n';
//		for (const auto& child : children)
//			str += child->to_string(indentations + 1) + '\n';
//		return str + std::string(indentations, '\t') + tag.end_tag();
//	}
//};

//class Logger
//{
//	std::stack<Search_Node> stack;
//public:
//	Logger() = default;
//	Logger(const Logger&) {}
//	Logger(Logger&&) = default;
//	Logger& operator=(const Logger&) {}
//	Logger& operator=(Logger&&) = default;
//	~Logger() = default;

//	void AddSearch(std::string name, const Position& pos, const Request& request) {
//		stack.emplace(std::move(name), pos, request);
//	}
//	void FinalizeSearch() {
//		if (stack.size() > 1) {
//			auto top = std::move(stack.top());
//			stack.pop();
//			stack.top().Add(std::make_unique<Search_Node>(std::move(top)));
//		}
//	}
//	void Add(std::string reason, const Result& result) {
//		stack.top().Add(std::make_unique<Cut_Node>(std::move(reason), result));
//		FinalizeSearch();
//	}
//	void Add(ContextfreeResult info) {
//		stack.top().Add(std::make_unique<TT_Node>(std::move(info)));
//	}
//	void Add(Field move, Result result) {
//		stack.top().Add(move, std::move(result));
//	}
//	void clear() { while (not stack.empty()) stack.pop(); }
//	std::string to_string() const { return stack.top().to_string(); }
//};

//inline std::string to_string(const Logger& logger) { return logger.to_string(); }

//
//class IDAB final : public PVS
//{
//public:
//	IDAB(HashTablePVS& tt, const AAGLEM& evaluator) noexcept : PVS(tt, evaluator) {}
//	std::unique_ptr<Algorithm> Clone() const override { return std::make_unique<IDAB>(tt, evaluator); }
//
//	int Eval(const Position&, const Intensity&, const OpenInterval&) override;
//	int Eval(const Position&, const Intensity&) override;
//	int Eval(const Position&, const OpenInterval&) override;
//	int Eval(const Position&) override;
//private:
//	int Eval_N(const Position&, const Intensity&);
//	int MTD_f(const Position&, const Intensity&, int guess);
//};
//
//OpenInterval NextZeroWindow(const OpenInterval&, int best_score) noexcept;
//OpenInterval NextFullWindow(const OpenInterval&, int best_score) noexcept;
//
//Search::Result AllMovesSearched(const OpenInterval&, const Search::Findings&) noexcept;
//
//
//uint64_t PotentialMoves(const Position&) noexcept;
//
//CUDA_CALLABLE inline int DoubleCornerPopcount(const BitBoard& b) noexcept { return popcount(b) + popcount(b & BitBoard::Corners()); }
//CUDA_CALLABLE inline int DoubleCornerPopcount(const Moves& m) noexcept { return m.size() + (m & BitBoard::Corners()).size(); }
//
//int32_t MoveOrderingScorer(const Position&, Field move) noexcept;