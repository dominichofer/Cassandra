#pragma once
#include "Stream.h"
#include "Filebased.h"
#include <functional>
#include <string>
#include <vector>

template <typename Data, typename MetaData = std::string>
class DB
{
	struct Entry
	{
		int key;
		Data value;

		operator Data& () noexcept { return value; }
		operator const Data&() const noexcept { return value; }

		static Entry Deserialize(std::istream& stream)
		{
			auto k = ::Deserialize<int>(stream);
			auto v = ::Deserialize<Data>(stream);
			return { k, v };
		}
		void Serialize(std::ostream& stream) const
		{
			::Serialize(key, stream);
			::Serialize(value, stream);
		}
	};

	std::vector<std::vector<MetaData>> metas;
	std::vector<Entry> entries;

	DB(std::vector<std::vector<MetaData>> metas, std::vector<Entry> entries)
		: metas(std::move(metas)), entries(std::move(entries))
	{}
public:
	DB() = default;
	static auto EmptyView() { return std::ranges::empty_view<Entry>(); }
	static DB<Data, MetaData> Deserialize(std::istream& stream)
	{
		auto metas = std::move(::Deserialize<decltype(DB<Data, MetaData>::metas)>(stream));
		auto entries = std::move(::Deserialize<decltype(DB<Data, MetaData>::entries)>(stream));
		return { std::move(metas), std::move(entries) };
	}

	void Serialize(std::ostream& stream) const
	{
		::Serialize(metas, stream);
		::Serialize(entries, stream);
	}

	template <std::ranges::range ValueRange>
	void Add(std::vector<MetaData> metas, ValueRange&& values)
	{
		auto it = ranges::find(this->metas, metas);
		int key = std::distance(this->metas.begin(), it);
		entries.reserve(entries.size() + ranges::distance(values));
		for (const Data& value : values)
			entries.emplace_back(key, value);
		if (it == this->metas.end())
			this->metas.push_back(std::move(metas));
	}
	void Add(std::vector<MetaData> metas, const Data& value)
	{
		Add(metas, std::ranges::single_view(value));
	}
	template <std::ranges::range MetaDataRange, std::ranges::range ValueRange>
	void Add(MetaDataRange&& metas, ValueRange&& values)
	{
		Add(metas | ranges::to_vector, values);
	}
	template <std::ranges::range MetaDataRange>
	void Add(MetaDataRange&& metas, Data value)
	{
		Add(metas, std::ranges::single_view(value));
	}

	auto begin() { return entries.begin(); }
	auto begin() const { return entries.begin(); }
	auto end() { return entries.end(); }
	auto end() const { return entries.end(); }
	std::size_t size() const { return entries.size(); }
	void clear() { metas.clear(); entries.clear(); }

private:
	auto Filtered(std::vector<bool>&& preds)       { return *this | ranges::views::filter([preds = std::move(preds)](const Entry& e) { return preds[e.key]; }); }
	auto Filtered(std::vector<bool>&& preds) const { return *this | ranges::views::filter([preds = std::move(preds)](const Entry& e) { return preds[e.key]; }); }
	auto FilteredOnMetas(auto&& trafo)       { return Filtered(metas | ranges::views::transform(trafo) | ranges::to_vector); }
	auto FilteredOnMetas(auto&& trafo) const { return Filtered(metas | ranges::views::transform(trafo) | ranges::to_vector); }
	static auto contains(const MetaData& value) { return [value](auto&& range) { return ranges::find(range, value) != range.end(); }; }
	static auto is_contained_in(auto&& range) { return [range](const MetaData& value) { return ranges::find(range, value) != range.end(); }; }
	static auto contains_any_of(const std::vector<MetaData>& values) { return [values](auto&& range) { return ranges::any_of(values, is_contained_in(range)); }; }
	static auto contains_all_of(const std::vector<MetaData>& values) { return [values](auto&& range) { return ranges::all_of(values, is_contained_in(range)); }; }
	static auto contains_none_of(const std::vector<MetaData>& values) { return [values](auto&& range) { return ranges::none_of(values, is_contained_in(range)); }; }
public:
	auto Where(const std::function<bool(const Data&)>& pred)       { return *this | ranges::views::filter(pred); }
	auto Where(const std::function<bool(const Data&)>& pred) const { return *this | ranges::views::filter(pred); }
	auto Where(const MetaData& value)       { return FilteredOnMetas(contains(value)); }
	auto Where(const MetaData& value) const { return FilteredOnMetas(contains(value)); }
	auto WhereAnyOf(const std::vector<MetaData>& values)       { return FilteredOnMetas(contains_any_of(values)); }
	auto WhereAnyOf(const std::vector<MetaData>& values) const { return FilteredOnMetas(contains_any_of(values)); }
	auto WhereAllOf(const std::vector<MetaData>& values)       { return FilteredOnMetas(contains_all_of(values)); }
	auto WhereAllOf(const std::vector<MetaData>& values) const { return FilteredOnMetas(contains_all_of(values)); }
	auto WhereNoneOf(const std::vector<MetaData>& values)       { return FilteredOnMetas(contains_none_of(values)); }
	auto WhereNoneOf(const std::vector<MetaData>& values) const { return FilteredOnMetas(contains_none_of(values)); }
};

template <typename Data, typename MetaData = std::string>
using FilebasedDB = Filebased<DB<Data, MetaData>>;
