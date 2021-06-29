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
		uint64 nodes = 0;
		//Logger log;

		virtual std::unique_ptr<Algorithm> Clone() const = 0; // TODO: Remove Clone() and make nodes thread_local and time it!

		// If the requested intensity is to search exact:
		// - and the score is in the requested window, the result window is a singleton, the score.
		// - and the score is above the requested window, the result window is [x,max_score] where x >= score.
		// - and the score is below the requested window, the result window is [min_score,x] where x <= score.
		// result.intensity >= requested.intensity.
		// A requested certainty of "90%" means, that any given node can be cut (fail high, fail low) if a shallow search
		// showed that there's a "90%" chance that the original search will result in a cut anyway.
		virtual int Eval(const Position&, const Intensity&, const OpenInterval&) = 0;
		virtual int Eval(const Position&, const Intensity&) = 0;
		virtual int Eval(const Position&, const OpenInterval&) = 0;
		virtual int Eval(const Position&) = 0;
	};
};

class NegaMax : public Search::Algorithm
{
public:
	std::unique_ptr<Algorithm> Clone() const override { return std::make_unique<NegaMax>(); }

	int Eval(const Position&, const Intensity&, const OpenInterval&) override;
	int Eval(const Position&, const Intensity&) override;
	int Eval(const Position&, const OpenInterval&) override;
	int Eval(const Position&) override;
private:
	int Eval_N(const Position&);
	int Eval_3(const Position&, Field, Field, Field);
	int Eval_2(const Position&, Field, Field);
protected:
	int Eval_1(const Position&, Field);
	int Eval_0(const Position&);
};

class AlphaBetaFailHard final : public NegaMax
{
public:
	std::unique_ptr<Algorithm> Clone() const override { return std::make_unique<AlphaBetaFailHard>(); }

	int Eval(const Position&, const Intensity&, const OpenInterval&) override;
	int Eval(const Position&, const Intensity&) override;
	int Eval(const Position&, const OpenInterval&) override;
	int Eval(const Position&) override;
private:
	int Eval_N(const Position&, OpenInterval);
	int Eval_3(const Position&, OpenInterval, Field, Field, Field);
	int Eval_2(const Position&, OpenInterval, Field, Field);
	int Eval_1(const Position&, OpenInterval, Field);
	int Eval_0(const Position&, OpenInterval);
};

class AlphaBetaFailSoft : public NegaMax
{
	static constexpr int Eval_to_ParitySort = 7;
public:
	std::unique_ptr<Algorithm> Clone() const override { return std::make_unique<AlphaBetaFailSoft>(); }

	int Eval(const Position&, const Intensity&, const OpenInterval&) override;
	int Eval(const Position&, const Intensity&) override;
	int Eval(const Position&, const OpenInterval&) override;
	int Eval(const Position&) override;
private:
	int Eval_2(const Position&, OpenInterval, Field, Field);
	int Eval_3(const Position&, OpenInterval, Field, Field, Field);
	int Eval_P(const Position&, OpenInterval); // Parity based move sorting.
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

	int Eval(const Position&, const Intensity&, const OpenInterval&) override;
	int Eval(const Position&, const Intensity&) override;
	int Eval(const Position&, const OpenInterval&) override;
	int Eval(const Position&) override;
private:
	int32_t MoveOrderingScorer(const Position&, Field move, Field best_move, Field best_move_2, int sort_alpha, int sort_depth) noexcept;
	std::optional<Search::Result> MPC(const Position&, const Intensity&, const OpenInterval&);

	Search::Result PVS_N(const Position&, const Intensity&, const OpenInterval&);
	Search::Result ZWS_N(const Position&, const Intensity&, const OpenInterval&, bool cut_node);
	Search::Result PVS_shallow(const Position&, const Intensity&, const OpenInterval& = OpenInterval::Whole());
	Search::Result ZWS_shallow(const Position&, const Intensity&, const OpenInterval& = OpenInterval::Whole());

	Search::Result Eval_dN(const Position&, const Intensity&, const OpenInterval&);
	int Eval_dN(const Position&, int depth, OpenInterval);
	int Eval_d0(const Position&);

	Search::Result ZWS_endgame(const Position&, const Intensity&, const OpenInterval&);
};

class IDAB final : public PVS
{
public:
	IDAB(HashTablePVS& tt, Pattern::Evaluator& evaluator) noexcept : PVS(tt, evaluator) {}
	std::unique_ptr<Algorithm> Clone() const override { return std::make_unique<IDAB>(tt, evaluator); }

	int Eval(const Position&, const Intensity&, const OpenInterval&) override;
	int Eval(const Position&, const Intensity&) override;
	int Eval(const Position&, const OpenInterval&) override;
	int Eval(const Position&) override;
private:
	int Eval_N(const Position&, const Intensity&);
	int MTD_f(const Position&, const Intensity&, int guess);
};

[[nodiscard]] OpenInterval NextZeroWindow(const OpenInterval&, int best_score) noexcept;
[[nodiscard]] OpenInterval NextFullWindow(const OpenInterval&, int best_score) noexcept;

[[nodiscard]] Search::Result AllMovesSearched(const OpenInterval&, const Search::Findings&) noexcept;


uint64_t PotentialMoves(const Position&) noexcept;

[[nodiscard]] CUDA_CALLABLE inline int DoubleCornerPopcount(const BitBoard& b) noexcept { return popcount(b) + popcount(b & BitBoard::Corners()); }
[[nodiscard]] CUDA_CALLABLE inline int DoubleCornerPopcount(const Moves& m) noexcept { return m.size() + m.Filtered(BitBoard::Corners()).size(); }

int32_t MoveOrderingScorer(const Position&, Field move) noexcept;
int32_t MoveOrderingScorer(const Position&, Field move, Field best_move, Field best_move_2) noexcept;