#pragma once
#include <vector>
#include <filesystem>
#include <fstream>

template <typename T>
struct compact
{
	using type = typename T;
};

template <typename T>
using dense_t = typename compact<T>::type;


template <typename value_type>
[[nodiscard]]
std::vector<value_type> Load(const std::filesystem::path& file)
{
	std::fstream stream(file, std::ios::in | std::ios::binary);
	if (!stream.is_open())
		throw std::fstream::failure("Can not open '" + file.string() + "' for binary intput.");

	stream.seekg(0, stream.end);
	const int size = stream.tellg() / sizeof(dense_t<value_type>);
	stream.seekg(0, stream.beg);

	std::vector<value_type> data;
	data.reserve(size);
	dense_t<value_type> buffer;
	for (int i = 0; i < size; i++)
	{
		stream.read(reinterpret_cast<char*>(&buffer), sizeof buffer);
		data.push_back(buffer);
	}
	return data;
}

template <typename Iterator>
void Save(const std::filesystem::path& file, Iterator begin, Iterator end)
{
	std::fstream stream(file, std::ios::out | std::ios::binary);
	if (!stream.is_open())
		throw std::fstream::failure("Can not open '" + file.string() + "' for binary output.");

	for (auto it = begin; it < end; ++it)
	{
		dense_t<Iterator::value_type> buffer{*it};
		stream.write(reinterpret_cast<const char*>(std::addressof(buffer)), sizeof buffer);
	}
}

template <typename C>
void Save(const std::filesystem::path& file, const C& container)
{
	Save(file, container.begin(), container.end());
}
