#pragma once
#include "Core/Moves.h"
#include "Core/Position.h"
#include "Core/Puzzle.h"
#include <string>
#include <vector>
#include <filesystem>

// Maps 'Field::A1' -> "A1", ... , 'Field::invalid' -> "--".
std::wstring to_wstring(Field) noexcept;
Field ParseField(const std::wstring&) noexcept;

std::wstring SingleLine(const Position&);
std::wstring MultiLine(const Position&);

Position ParsePosition_SingleLine(const std::wstring&) noexcept(false);



// Format: "ddd:hh:mm:ss.ccc"
std::wstring time_format(const std::chrono::milliseconds duration);

// Format: "ddd:hh:mm:ss.ccc"
template <class U, class V>
std::wstring time_format(std::chrono::duration<U, V> duration)
{
	return time_format(std::chrono::duration_cast<std::chrono::milliseconds>(duration));
}

std::wstring short_time_format(std::chrono::duration<double> duration);

template <typename Inserter>
void ReadFile(Inserter& inserter, const std::filesystem::path& path, std::size_t count = std::numeric_limits<std::size_t>::max())
{
	std::fstream file(path, std::ios::in | std::ios::binary);
	if (!file.is_open())
		throw std::fstream::failure("File '" + filename + "' could not be opened for input.");

	while (count-- > 0)
	{
		std::iterator_traits<Inserter>::value_type buffer;
		const auto read = file.read(reinterpret_cast<char*>(&buffer), sizeof(std::iterator_traits<Inserter>::value_type));
		if (!read)
			break;
		inserter = std::move(buffer);
	}

	file.close();
}

template <typename Iterator>
void WriteToFile(const std::filesystem::path& path, Iterator begin, Iterator end)
{
	std::fstream file(path, std::ios::out | std::ios::binary);
	if (!file.is_open())
		throw std::fstream::failure("File '" + filename + "' could not be opened for output.");

	for (auto it = begin; it != end; ++it)
		file.write(reinterpret_cast<const char*>(std::addressof(*it)), sizeof(std::iterator_traits<Iterator>::value_type));

	file.close();
}