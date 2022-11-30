#include "SearchIO.h"
#include "CoreIO/CoreIO.h"
#include <algorithm>

Intensity ParseIntensity(const std::string& str) noexcept(false)
{
	int separator_index = std::distance(str.begin(), std::find(str.begin(), str.end(), '@'));
	int depth = std::stoi(str.substr(0, separator_index));
	Confidence certainty = Confidence::Certain();
	if (separator_index != str.size())
		certainty = Confidence{ std::stof(str.substr(separator_index + 1)) };
	return { depth, certainty };
}