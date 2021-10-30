#pragma once
#include "File.h"
#include "Search/Puzzle.h"
#include <filesystem>
#include <vector>
#include <iterator>
#include <ranges>
#include <span>
#include <string>
#include <functional>
#include <map>
#include <range/v3/action/join.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/cache1.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/subrange.hpp>

class DB
{
	using MetaData = std::vector<std::string>;
	struct Entry
	{
		int key;
		Puzzle value;
		operator Puzzle() const noexcept { return value; }
	};
	std::vector<MetaData> meta;
	std::vector<Entry> data;

	template <std::ranges::viewable_range View>
	class ViewClosure
	{
		const std::vector<MetaData>& meta;
		View view;

		template <typename Pred>
		auto Filtered(Pred pred) { auto new_view = view | std::views::filter(pred); return ViewClosure<decltype(new_view)>(meta, new_view); }
	public:
		ViewClosure(const std::vector<MetaData>& meta, View view) noexcept : meta(meta), view(view) {}

		auto begin() { return view.begin(); }
		auto end() { return view.end(); }

		auto Where(const std::string& meta_info) {
			std::vector<bool> preds;
			preds.reserve(meta.size());
			for (const MetaData& m : meta)
				preds.push_back(std::ranges::find(m, meta_info) != m.end());
			return Filtered([&preds](const Entry& e) { return preds[e.key]; });
		}
		auto Where(const std::function<bool(const Puzzle&)>& pred) {
			return Filtered(pred);
		}
		auto WhereAnyOf(const std::vector<std::string>& meta_infos) {
			std::vector<bool> preds;
			preds.reserve(meta.size());
			for (const MetaData& m : meta)
				preds.push_back(std::ranges::any_of(meta_infos, [&m](const std::string& meta_info) { return std::ranges::find(m, meta_info) != m.end(); }));
			return Filtered([&preds](const Entry& e) { return preds[e.key]; });
		}
		auto WhereEmptyCount(int min, int max) {
			return Where([min, max](const Puzzle& p) { int ec = p.pos.EmptyCount(); return min <= ec && ec <= max; });
		}
		auto WhereEmptyCount(int empty_count) {
			return Where([empty_count](const Puzzle& p) { return p.pos.EmptyCount() == empty_count; });
		}
		auto SplitEachEmptyCount(int size_1, int size_2) {
			auto indices = ranges::views::cartesian_product(ranges::views::iota(0, static_cast<int>(meta.size())), ranges::views::iota(0, 65));

			auto part_1 = indices | ranges::views::transform(
				[this, size_1, size_2](auto idx) {
					auto [key, empty_count] = idx;
					return view
						| ranges::views::filter([key, empty_count](const Entry& e) { return e.key == key && e.value.pos.EmptyCount() == empty_count; })
						| ranges::views::take(size_1);
				})
				| ranges::actions::join;

			auto part_2 = indices | ranges::views::transform(
				[this, size_1, size_2](auto idx) {
					auto [key, empty_count] = idx;
					return view
						| ranges::views::filter([key, empty_count](const Entry& e) { return e.key == key && e.value.pos.EmptyCount() == empty_count; })
						| ranges::views::drop(size_1)
						| ranges::views::take(size_2);
				})
				| ranges::actions::join;

			return std::make_pair(part_1, part_2);
		}
	};

public:
	DB() = default;
	DB(const std::filesystem::path& file);
	void WriteBack() const;
	void Add(std::vector<std::string> meta_info, const PuzzleRange auto& data);

	auto Where(const std::string& meta_info) { return ViewClosure(meta, std::views::all(data)).Where(meta_info); }
	auto Where(const std::string& meta_info) const { return ViewClosure(meta, std::views::all(data)).Where(meta_info); }
	auto Where(const std::function<bool(const Puzzle&)>& pred) { return ViewClosure(meta, std::views::all(data)).Where(pred); }
	auto Where(const std::function<bool(const Puzzle&)>& pred) const { return ViewClosure(meta, std::views::all(data)).Where(pred); }
	auto WhereAnyOf(const std::vector<std::string>& meta_infos) { return ViewClosure(meta, std::views::all(data)).WhereAnyOf(meta_infos); }
	auto WhereAnyOf(const std::vector<std::string>& meta_infos) const  { return ViewClosure(meta, std::views::all(data)).WhereAnyOf(meta_infos); }
	auto WhereEmptyCount(int min, int max) { return ViewClosure(meta, std::views::all(data)).WhereEmptyCount(min, max); }
	auto WhereEmptyCount(int min, int max) const  { return ViewClosure(meta, std::views::all(data)).WhereEmptyCount(min, max); }
	auto WhereEmptyCount(int empty_count) { return ViewClosure(meta, std::views::all(data)).WhereEmptyCount(empty_count); }
	auto WhereEmptyCount(int empty_count) const { return ViewClosure(meta, std::views::all(data)).WhereEmptyCount(empty_count); }
	auto SplitEachEmptyCount(int size_1, int size_2) { return ViewClosure(meta, std::views::all(data)).SplitEachEmptyCount(size_1, size_2); }
	auto SplitEachEmptyCount(int size_1, int size_2) const { return ViewClosure(meta, std::views::all(data)).SplitEachEmptyCount(size_1, size_2); }
};

inline void foo()
{
	const DB db;
	auto [a, b] = db.WhereAnyOf({ "play00", "play11" }).Where("eval_fit").WhereEmptyCount(1, 10).SplitEachEmptyCount(80, 20);
}

namespace views
{
	struct train_test
	{
		std::size_t train_size, test_size;
	};
}

template <typename T>
class DataBase
{
	struct FileSection
	{
		std::size_t first, last;
		std::filesystem::path file;
	};

	std::vector<T> vec;
	std::vector<FileSection> section;

	auto Subrange(const FileSection& s) const { return ranges::make_subrange(vec.begin() + s.first, vec.begin() + s.last); }
	void Add(const range<T> auto& data) { vec.insert(vec.end(), data.begin(), data.end()); }
public:
	DataBase() noexcept = default;
	DataBase(const std::filesystem::path& file) { Add(file); }
	DataBase(const std::vector<std::filesystem::path>& files) { for (const auto& file : files) Add(file); }

	void WriteBack() const
	{
		for (const FileSection& s : section)
			Save(s.file, vec.begin() + s.first, vec.begin() + s.last);
	}

	void Add(const std::filesystem::path& file, const range<T> auto& data)
	{
		auto first = vec.size();
		Add(data);
		auto last = vec.size();
		section.emplace_back(first, last, file);
	}

	void Add(const std::filesystem::path& file)
	{
		Add(file, Load<std::vector<T>>(file));
	}

	void Add(const DataBase<T>& o)
	{
		auto offset = vec.size();
		Add(o.vec);
		section.reserve(section.size() + o.section.size());
		for (const FileSection& s : o.section)
			section.emplace_back(s.first + offset, s.last + offset, s.file);
	}

	using value_type = T;
	[[nodiscard]] std::size_t size() const noexcept { return vec.size(); }
	[[nodiscard]] decltype(auto) begin() noexcept { return vec.begin(); }
	[[nodiscard]] decltype(auto) begin() const noexcept { return vec.begin(); }
	[[nodiscard]] decltype(auto) cbegin() noexcept { return vec.cbegin(); }
	[[nodiscard]] decltype(auto) end() noexcept { return vec.end(); }
	[[nodiscard]] decltype(auto) end() const noexcept { return vec.end(); }
	[[nodiscard]] decltype(auto) cend() noexcept { return vec.cend(); }
	[[nodiscard]] decltype(auto) operator[](std::size_t index) const noexcept { return vec[index]; }
	[[nodiscard]] decltype(auto) operator[](std::size_t index) noexcept { return vec[index]; }

	[[nodiscard]] std::size_t Sections() const noexcept { return section.size(); }
	[[nodiscard]] std::span<T> Section(std::size_t i) noexcept { return std::span<T>(vec.begin() + section[i].first, vec.begin() + section[i].last); }
	[[nodiscard]] std::vector<std::span<T>> AsSections()
	{
		std::vector<std::span<T>> ret;
		for (std::size_t i = 0; i < section.size(); i++)
			ret.push_back(Section(i));
		return ret;
	}

	auto operator|(views::train_test tt) const
	{
		auto view = ranges::views::cartesian_product(section, ranges::views::iota(0, 65));

		auto train = view | ranges::views::transform(
			[this, tt](auto idx) {
				auto [section, empty_count] = idx;
				return Subrange(section)
					| views::empty_count_filter(empty_count)
					| ranges::views::take(tt.train_size);
			})
			| ranges::actions::join;

		auto test = view | ranges::views::transform(
			[this, tt](auto idx) {
				auto [section, empty_count] = idx;
				return Subrange(section)
					| views::empty_count_filter(empty_count)
					| ranges::views::drop(tt.train_size)
					| ranges::views::take(tt.test_size);
			})
			| ranges::actions::join;

		return std::make_pair(std::move(train), std::move(test));
	}
};

static_assert(std::ranges::random_access_range<DataBase<int>>);


template <typename T>
class take_each_view : public std::ranges::view_interface<take_each_view<T>>
{
	std::size_t count;
	std::vector<std::span<T>> sections;
public:
	take_each_view(DataBase<T>& db, std::size_t count) : count(count), sections(db.AsSections()) {}

	auto begin() const noexcept {
		auto view = sections
			| std::views::transform([this](const std::span<T>& s) { return s | std::views::take(count); })
			| std::views::join;
		return view.begin();
	}
	auto end() const noexcept {
		auto view = sections
			| std::views::transform([this](const std::span<T>& s) { return s | std::views::take(count); })
			| std::views::join;
		return view.end();
	}
};

namespace details
{
	class take_each_range_adaptor_closure
	{
		std::size_t count;
	public:
		take_each_range_adaptor_closure(std::size_t count) : count(count) {}

		template <typename T>
		auto operator()(DataBase<T>& db) const { return take_each_view(db, count); }
	};

	struct take_each_range_adaptor
	{
		template <typename T>
		auto operator()(DataBase<T>& db, std::size_t count) const { return take_each_view(db, count); }

		auto operator()(std::size_t count) { return take_each_range_adaptor_closure(count); }
	};

	template <typename T>
	auto operator|(DataBase<T>& db, const take_each_range_adaptor_closure& closure) { return closure(db); }
}

template <typename T>
class drop_each_view : public std::ranges::view_interface<drop_each_view<T>>
{
	std::size_t count;
	std::vector<std::span<T>> sections;
public:
	drop_each_view(DataBase<T>& db, std::size_t count) : count(count), sections(db.AsSections()) {}

	auto begin() const noexcept {
		auto view = sections
			| std::views::transform([this](const auto& s) { return s | std::views::drop(count); })
			| std::views::join;
		return view.begin();
	}
	auto end() const noexcept {
		auto view = sections
			| std::views::transform([this](const auto& s) { return s | std::views::drop(count); })
			| std::views::join;
		return view.end();
	}
};

namespace details
{
	class drop_each_range_adaptor_closure
	{
		std::size_t count;
	public:
		drop_each_range_adaptor_closure(std::size_t count) : count(count) {}

		template <typename T>
		auto operator()(DataBase<T>& db) const { return drop_each_view(db, count); }
	};

	struct drop_each_range_adaptor
	{
		template <typename T>
		auto operator()(DataBase<T>& db, std::size_t count) const { return drop_each_view(db, count); }

		auto operator()(std::size_t count) { return drop_each_range_adaptor_closure(count); }
	};

	template <typename T>
	auto operator|(DataBase<T>& db, const drop_each_range_adaptor_closure& closure) { return closure(db); }
}

namespace views
{
	static details::take_each_range_adaptor take_each;
	static details::drop_each_range_adaptor drop_each;
}

inline DataBase<Puzzle> LoadDB(const std::string_view fmt)
{
	DataBase<Puzzle> db;
	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 10; j++)
		{
			try
			{
				db.Add(std::format(fmt, i, j));
			}
			catch (...)
			{}
		}
	return db;
}

inline DataBase<Puzzle> LoadEvalFit() { return LoadDB(R"(G:\Reversi\play{}{}_eval_fit.puz)"); }
inline DataBase<Puzzle> LoadAccuracyFit() { return LoadDB(R"(G:\Reversi\play{}{}_accuracy_fit.puz)"); }
inline DataBase<Puzzle> LoadMoveSort() { return LoadDB(R"(G:\Reversi\play{}{}_move_sort.puz)"); }
inline DataBase<Puzzle> LoadBenchmark() { return LoadDB(R"(G:\Reversi\play{}{}_benchmark.puz)"); }