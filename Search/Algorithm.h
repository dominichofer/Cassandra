#pragma once
#include "Core/Core.h"

#include "Pattern/Evaluator.h"
#include "Objects.h"
#include "HashTablePVS.h"

#include <cassert>
#include <cstdint>
#include <cmath>
#include <compare>
#include <string>
#include <stack>

namespace Search
{
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
	//	TT_Info info;
	//public:
	//	TT_Node(TT_Info info) noexcept : info(std::move(info)) {}
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
	//	void Add(TT_Info info) {
	//		stack.top().Add(std::make_unique<TT_Node>(std::move(info)));
	//	}
	//	void Add(Field move, Result result) {
	//		stack.top().Add(move, std::move(result));
	//	}
	//	void clear() { while (not stack.empty()) stack.pop(); }
	//	std::string to_string() const { return stack.top().to_string(); }
	//};

	//inline std::string to_string(const Logger& logger) { return logger.to_string(); }

	class Algorithm
	{
	public:
		uint64 node_count = 0;
		//Logger log;
		
		virtual std::unique_ptr<Algorithm> Clone() const = 0;

		// If the requested intensity is to search exact:
		// - and the score is in the requested window, the result window is a singleton, the score.
		// - and the score is above the requested window, the result window is [x,max_score] where x >= score.
		// - and the score is below the requested window, the result window is [min_score,x] where x <= score.
		// result.intensity >= requested.intensity.
		// A requested certainty of "90%" means, that any given node can be cut (fail high, fail low) if a shallow search
		// showed that there's a "90%" chance that the original search will result in a cut anyway.
		virtual Result Eval(const Position&, const Request&) = 0;

		int Score(const Position& pos, int depth) { return Eval(pos, Request::Certain(depth)).Score(); }
		int Score(const Position& pos) { return Score(pos, pos.EmptyCount()); }
	};
};

class NegaMax : public Search::Algorithm
{
public:
	std::unique_ptr<Algorithm> Clone() const override { return std::make_unique<NegaMax>(); }

	Search::Result Eval(const Position& pos, const Search::Request& request) override { return Search::Result::ExactFailSoft(request, pos, Eval(pos)); }
	int Eval(const Position&);
protected:
	int Eval_0(const Position&);
	int Eval_1(const Position&, Field);
private:
	int Eval_2(const Position&, Field, Field);
	int Eval_3(const Position&, Field, Field, Field);
	int Eval_N(const Position&);
};

class AlphaBetaFailHard final : public NegaMax
{
public:
	std::unique_ptr<Algorithm> Clone() const override { return std::make_unique<AlphaBetaFailHard>(); }

	Search::Result Eval(const Position&, const Search::Request&) override;
	int Eval(const Position&, OpenInterval);
private:
	int Eval_0(const Position&, OpenInterval);
	int Eval_1(const Position&, OpenInterval, Field);
	int Eval_2(const Position&, OpenInterval, Field, Field);
	int Eval_3(const Position&, OpenInterval, Field, Field, Field);
	int Eval_N(const Position&, OpenInterval);
};

class AlphaBetaFailSoft : public NegaMax
{
	static constexpr int Eval_to_ParitySort = 7;
public:
	std::unique_ptr<Algorithm> Clone() const override { return std::make_unique<AlphaBetaFailSoft>(); }

	Search::Result Eval(const Position&, const Search::Request&) override;
	int Eval(const Position&, OpenInterval);
private:
	int Eval_2(const Position&, OpenInterval, Field, Field);
	int Eval_3(const Position&, OpenInterval, Field, Field, Field);
	int Eval_P(const Position&, OpenInterval); // Parity based move ordering.
	int Eval_N(const Position&, OpenInterval);
};

class PVS : public AlphaBetaFailSoft
{
	static constexpr int PVS_to_AlphaBetaFailSoft = 10;
	//static constexpr int ZWS_to_AlphaBetaFailSoft = 10;
protected:
	HashTablePVS& tt;
	Pattern::Evaluator& evaluator;
public:
	PVS(HashTablePVS& tt, Pattern::Evaluator& evaluator) noexcept : tt(tt), evaluator(evaluator) {}
	std::unique_ptr<Algorithm> Clone() const override { return std::make_unique<PVS>(tt, evaluator); }

	Search::Result Eval(const Position&, const Search::Request&) override;
private:
	int32_t MoveOrderingScorer(const Position&, Field move, Field best_move, Field best_move_2, int sort_alpha, int sort_depth) noexcept;
	std::optional<Search::Result> MPC(const Position&, const Search::Request&);

	Search::Result PVS_N(const Position&, const Search::Request&);
	Search::Result ZWS_N(const Position&, const Search::Request&, bool cut_node);
	Search::Result PVS_shallow(const Position&, Search::Request);
	Search::Result ZWS_shallow(const Position&, const Search::Request&);

	int Eval_d0(const Position&);
	int Eval_dN(const Position&, OpenInterval, int depth);
	Search::Result Eval_dN(const Position&, const Search::Request&);

	Search::Result ZWS_endgame(const Position&, const Search::Request&);
};

class IDAB final : public PVS
{
	int MTD_f(const Position&, const Search::Intensity&, int guess);
public:
	IDAB(HashTablePVS& tt, Pattern::Evaluator& evaluator) noexcept : PVS(tt, evaluator) {}
	virtual std::unique_ptr<Algorithm> Clone() const { return std::make_unique<IDAB>(tt, evaluator); }

	Search::Result Eval(const Position&, const Search::Request&) override;
};

[[nodiscard]] Search::Request NextZWS(const Search::Request&, const Search::Findings&) noexcept;
[[nodiscard]] Search::Request NextFWS(const Search::Request&, const Search::Findings&) noexcept;

[[nodiscard]] Search::Result AllMovesSearched(const Search::Request&, const Search::Findings&) noexcept;


uint64_t PotentialMoves(const Position&) noexcept;

[[nodiscard]] CUDA_CALLABLE inline int DoubleCornerPopcount(const BitBoard& b) noexcept { return popcount(b) + popcount(b & BitBoard::Corners()); }
[[nodiscard]] CUDA_CALLABLE inline int DoubleCornerPopcount(const Moves& m) noexcept { return m.size() + m.Filtered(BitBoard::Corners()).size(); }

int32_t MoveOrderingScorer(const Position&, Field move) noexcept;
int32_t MoveOrderingScorer(const Position&, Field move, Field best_move, Field best_move_2) noexcept;