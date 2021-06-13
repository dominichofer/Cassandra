#pragma once
#include "File.h"
#include <filesystem>
#include <vector>

template <typename T>
class DataBase
{
	struct FileCache
	{
		std::vector<T> data;
		std::filesystem::path file;
	public:
		explicit FileCache(const std::filesystem::path& file) : file(file) { data = Load<std::vector<T>>(file); }
		void WriteBack() const { Save(file, data); }
		std::size_t size() const { return data.size(); }
		      T& operator[](std::size_t i)       { return data[i]; }
		const T& operator[](std::size_t i) const { return data[i]; }
	};

	class Iterator
	{
		std::size_t i, j;
		std::vector<FileCache>& vec;

		Iterator(std::size_t i, std::size_t j, std::vector<FileCache>& vec) noexcept : i(i), j(j), vec(vec) {}
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = std::remove_cv_t<T>;
		using pointer = T*;
		using reference = T&;
		using iterator_category = std::forward_iterator_tag; // TODO: Make this a random access iterator

		static Iterator Begin(std::vector<FileCache>& vec) noexcept { return Iterator(0, 0, vec); }
		static Iterator End(std::vector<FileCache>& vec) noexcept { return Iterator(vec.size(), 0, vec); }

		[[nodiscard]] bool operator==(const Iterator& o) const { return i == o.i and j == o.j; }
		[[nodiscard]] bool operator!=(const Iterator& o) const { return !(*this == o); }
		[[nodiscard]] auto operator<=>(const Iterator& o) const { return std::tie(i, j) <=> std::tie(o.i, o.j); }

		Iterator& operator++() {
			if (i >= vec.size())
				return *this;
			j++;
			while (i < vec.size() and j >= vec[i].size())
			{
				j = 0;
				i++;
			}
			return *this;
		}
		reference operator*() const { return vec[i][j]; }
		pointer operator->() const { return &vec[i][j]; }
		//difference_type operator-(Iterator o) const {
		//	difference_type diff = 0;
		//	for (; o != *this; ++o)
		//		++diff;
		//	return diff;
		//}
	};

	//std::vector<T> plain_data;
	std::vector<FileCache> file_data;
public:
	DataBase() noexcept = default;
	DataBase(const std::filesystem::path& file) { Add(file); }
	DataBase(const std::vector<std::filesystem::path>& files) { for (const auto& file : files) Add(file); }

	void WriteBack() const { for (const FileCache& fc : file_data) fc.WriteBack(); }

	using value_type = std::remove_cv_t<T>;

	//void Add(const std::vector<T>& data) { plain_data.insert(plain_data.end(), data.begin(), data.end()); }
	void Add(const std::filesystem::path& file) { file_data.emplace_back(file); }

	Iterator begin() { return Iterator::Begin(file_data); }
	Iterator end() { return Iterator::End(file_data); }

	//template <typename UnaryPredicate>
	//std::vector<std::reference_wrapper<T>> Where(UnaryPredicate p) const
	//{
	//	std::vector<std::reference_wrapper<T>> vec;
	//	for (const T& t : plain_data)
	//		if (p(t))
	//			vec.push_back(t);
	//	for (const FileCache& fc : file_data)
	//		for (const T& t : fc.data)
	//			if (p(t))
	//				vec.push_back(t);
	//	return vec;
	//}
};