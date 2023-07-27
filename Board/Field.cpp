#include "Field.h"
#include <array>
#include <regex>
#include <unordered_map>

CUDA_CALLABLE uint64_t Bit(Field f)
{
	return 1ULL << std::to_underlying(f);
}

std::string to_string(Field field)
{
	static const std::array<std::string, 65> field_names =
	{
		"H8", "G8", "F8", "E8", "D8", "C8", "B8", "A8",
		"H7", "G7", "F7", "E7", "D7", "C7", "B7", "A7",
		"H6", "G6", "F6", "E6", "D6", "C6", "B6", "A6",
		"H5", "G5", "F5", "E5", "D5", "C5", "B5", "A5",
		"H4", "G4", "F4", "E4", "D4", "C4", "B4", "A4",
		"H3", "G3", "F3", "E3", "D3", "C3", "B3", "A3",
		"H2", "G2", "F2", "E2", "D2", "C2", "B2", "A2",
		"H1", "G1", "F1", "E1", "D1", "C1", "B1", "A1",
		"PS"
	};
	return field_names[std::to_underlying(field)];
}

bool IsField(std::string_view str)
{
	std::regex pattern("[A-H][1-8]|PS");
	return std::regex_match(str.begin(), str.end(), pattern);
}

Field FieldFromString(std::string_view str)
{
	static const std::unordered_map<std::string_view, Field> field_map = {
		   {"H8", Field::H8}, {"G8", Field::G8}, {"F8", Field::F8}, {"E8", Field::E8}, {"D8", Field::D8}, {"C8", Field::C8}, {"B8", Field::B8}, {"A8", Field::A8},
		   {"H7", Field::H7}, {"G7", Field::G7}, {"F7", Field::F7}, {"E7", Field::E7}, {"D7", Field::D7}, {"C7", Field::C7}, {"B7", Field::B7}, {"A7", Field::A7},
		   {"H6", Field::H6}, {"G6", Field::G6}, {"F6", Field::F6}, {"E6", Field::E6}, {"D6", Field::D6}, {"C6", Field::C6}, {"B6", Field::B6}, {"A6", Field::A6},
		   {"H5", Field::H5}, {"G5", Field::G5}, {"F5", Field::F5}, {"E5", Field::E5}, {"D5", Field::D5}, {"C5", Field::C5}, {"B5", Field::B5}, {"A5", Field::A5},
		   {"H4", Field::H4}, {"G4", Field::G4}, {"F4", Field::F4}, {"E4", Field::E4}, {"D4", Field::D4}, {"C4", Field::C4}, {"B4", Field::B4}, {"A4", Field::A4},
		   {"H3", Field::H3}, {"G3", Field::G3}, {"F3", Field::F3}, {"E3", Field::E3}, {"D3", Field::D3}, {"C3", Field::C3}, {"B3", Field::B3}, {"A3", Field::A3},
		   {"H2", Field::H2}, {"G2", Field::G2}, {"F2", Field::F2}, {"E2", Field::E2}, {"D2", Field::D2}, {"C2", Field::C2}, {"B2", Field::B2}, {"A2", Field::A2},
		   {"H1", Field::H1}, {"G1", Field::G1}, {"F1", Field::F1}, {"E1", Field::E1}, {"D1", Field::D1}, {"C1", Field::C1}, {"B1", Field::B1}, {"A1", Field::A1},
		   {"PS", Field::PS}
	};
	auto it = field_map.find(str);
	if (it != field_map.end())
		return it->second;
	else
		throw std::runtime_error("Invalid field format");
}