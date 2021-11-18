#pragma once
#include "File.h"
#include "Search/Puzzle.h"
#include <filesystem>
#include <vector>
#include <string>
#include <iterator>
#include <span>
#include <string>
#include <functional>
#include <map>
#include <ranges>
#include <range/v3/all.hpp>


template <typename MetaData = std::string, typename Data = Puzzle>
class DB
{
	struct Entry
	{
		int key;
		Data value;
		operator Data() const noexcept { return value; }
	};

	std::filesystem::path file;
	std::vector<std::vector<MetaData>> metas;
	std::vector<Entry> entries;

	template <typename View>
	class ViewClosure
	{
		const std::vector<std::vector<MetaData>>& metas;
		View view;

		template <typename Pred>
		auto Filtered(Pred pred) { auto new_view = view | ranges::views::filter(pred); return ViewClosure<decltype(new_view)>(metas, new_view); }
	public:
		ViewClosure(const std::vector<std::vector<MetaData>>& metas, View view) noexcept : metas(metas), view(view) {}

		auto begin() { return view.begin(); }
		auto end() { return view.end(); }

		auto Where(const MetaData& fit) {
			std::vector<bool> preds;
			preds.reserve(metas.size());
			for (const auto& m : metas)
				preds.push_back(ranges::find(m, fit) != m.end());
			return Filtered([&preds](const Entry& e) { return preds[e.key]; });
		}
		auto Where(const std::function<bool(const Data&)>& pred) {
			return Filtered(pred);
		}
		auto WhereAnyOf(const std::vector<MetaData>& fits) {
			std::vector<bool> preds;
			preds.reserve(metas.size());
			for (const auto& m : metas)
				preds.push_back(ranges::any_of(fits, [&m](const MetaData& fit) { return ranges::find(m, fit) != m.end(); }));
			return Filtered([&preds](const Entry& e) { return preds[e.key]; });
		}
		auto WhereEmptyCount(int min, int max) requires requires (Data data) { EmptyCount(data); } {
			return Where([min, max](const Data& data) { int empty_count = EmptyCount(data); return min <= empty_count && empty_count <= max; });
		}
		auto WhereEmptyCount(int empty_count) requires requires (Data data) { EmptyCount(data); } {
			return Where([empty_count](const Data& data) { return EmptyCount(data) == empty_count; });
		}
		auto SplitEachEmptyCount(int size_1, int size_2) requires requires (Data data) { EmptyCount(data); } {
			auto indices = ranges::views::cartesian_product(ranges::views::iota(std::size_t(0), metas.size()), ranges::views::iota(0, 65));

			auto part_1 = indices | ranges::views::transform(
				[this, size_1, size_2](auto idx) {
					auto [key, empty_count] = idx;
					return view
						| ranges::views::filter([key, empty_count](const Entry& e) { return e.key == key && EmptyCount(e.value) == empty_count; })
						| ranges::views::take(size_1);
				})
				| ranges::actions::join;

			auto part_2 = indices | ranges::views::transform(
				[this, size_1, size_2](auto idx) {
					auto [key, empty_count] = idx;
					return view
						| ranges::views::filter([key, empty_count](const Entry& e) { return e.key == key && EmptyCount(e.value) == empty_count; })
						| ranges::views::drop(size_1)
						| ranges::views::take(size_2);
				})
				| ranges::actions::join;

			return std::make_pair(part_1, part_2);
		}
	};
	auto CreateViewClosure() { return ViewClosure{ metas, std::views::all(entries) }; }
	auto CreateViewClosure() const { return ViewClosure{ metas, std::views::all(entries) }; }
public:
	DB() = default;
	DB(std::filesystem::path file) : file(std::move(file))
	{
		BinaryFileStream stream{ file };
		metas = stream.read<decltype(metas)>();
		auto size = stream.read<std::size_t>();
		entries.reserve(size);
		for (std::size_t i = 0; i < size; i++)
		{
			auto key = stream.read<decltype(Entry::key)>();
			auto value = stream.read<decltype(Entry::value)>();
			entries.emplace_back(key, value);
		}
	}
	void WriteBack() const
	{
		BinaryFileStream stream{ file };
		stream.write(metas);
		stream.write(entries);
	}
	void Add(std::vector<MetaData> metas, Data value)
	{
		auto it = ranges::find(this->metas, metas);
		auto key = std::distance(this->metas.begin(), it);
		entries.emplace_back(key, std::move(value));
		if (it == this->metas.end())
			this->metas.push_back(std::move(metas));
	}
	void Add(std::vector<MetaData> metas, const std::ranges::range auto& values)
	{
		auto it = ranges::find(this->metas, metas);
		auto key = std::distance(this->metas.begin(), it);
		entries.reserve(entries.size() + ranges::size(values));
		for (const Data& value : values)
			entries.emplace_back(key, value);
		if (it == this->metas.end())
			this->metas.push_back(std::move(metas));
	}

	auto Where(const MetaData& meta_info) { return CreateViewClosure().Where(meta_info); }
	auto Where(const MetaData& meta_info) const { return CreateViewClosure().Where(meta_info); }
	auto Where(const std::function<bool(const Puzzle&)>& pred) { return CreateViewClosure().Where(pred); }
	auto Where(const std::function<bool(const Puzzle&)>& pred) const { return CreateViewClosure().Where(pred); }
	auto WhereAnyOf(const std::vector<MetaData>& meta_infos) { return CreateViewClosure().WhereAnyOf(meta_infos); }
	auto WhereAnyOf(const std::vector<MetaData>& meta_infos) const  { return CreateViewClosure().WhereAnyOf(meta_infos); }
	auto WhereEmptyCount(int min, int max) { return CreateViewClosure().WhereEmptyCount(min, max); }
	auto WhereEmptyCount(int min, int max) const  { return CreateViewClosure().WhereEmptyCount(min, max); }
	auto WhereEmptyCount(int empty_count) { return CreateViewClosure().WhereEmptyCount(empty_count); }
	auto WhereEmptyCount(int empty_count) const { return CreateViewClosure().WhereEmptyCount(empty_count); }
	auto SplitEachEmptyCount(int size_1, int size_2) { return CreateViewClosure().SplitEachEmptyCount(size_1, size_2); }
	auto SplitEachEmptyCount(int size_1, int size_2) const { return CreateViewClosure().SplitEachEmptyCount(size_1, size_2); }
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

inline DataBase<Puzzle> LoadDB(const std::string& pre, const std::string& post)
{
	DataBase<Puzzle> db;
	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 10; j++)
		{
			try
			{
				db.Add(pre + std::to_string(i) + std::to_string(j) + post);
			}
			catch (...)
			{}
		}
	return db;
}

inline DataBase<Puzzle> LoadEvalFit() { return LoadDB(R"(G:\Reversi\play)", "_eval_fit.puz)"); }
inline DataBase<Puzzle> LoadAccuracyFit() { return LoadDB(R"(G:\Reversi\play)", "_accuracy_fit.puz)"); }
inline DataBase<Puzzle> LoadMoveSort() { return LoadDB(R"(G:\Reversi\play)", "_move_sort.puz)"); }
inline DataBase<Puzzle> LoadBenchmark() { return LoadDB(R"(G:\Reversi\play)", "_benchmark.puz)"); }