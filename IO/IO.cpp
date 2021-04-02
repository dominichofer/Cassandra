#include "IO.h"
#include "Integers.h"
#include "Core/Core.h"
#include <array>
#include <sstream>

Field ParseField(const std::string& str)
{
	static const std::array<std::string, 65> field_names = {
		"A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1",
		"A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2",
		"A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3",
		"A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4",
		"A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5",
		"A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6",
		"A7", "B7", "C7", "D7", "E7", "F7", "G7", "H7",
		"A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8", "--"
	};
	const auto it = std::find(field_names.begin(), field_names.end(), str);
	const auto index = std::distance(field_names.begin(), it);
	return static_cast<Field>(index);
}

Position ParsePosition_SingleLine(const std::string& str) noexcept(false)
{
	BitBoard P, O;

	for (int i = 0; i < 64; i++)
	{
		if (str[i] == 'X')
			P.Set(static_cast<Field>(63 - i));
		if (str[i] == 'O')
			O.Set(static_cast<Field>(63 - i));
	}

	if (str[65] == 'X')
		return Position{ P, O };
	if (str[65] == 'O')
		return Position{ O, P };
	throw;
}

std::string short_time_format(std::chrono::duration<double> duration)
{
	using namespace std;

	const auto seconds = duration.count();
	const int magnitude_base_1000 = static_cast<int>(floor(log10(abs(seconds)) / 3));
	const double normalized = seconds * pow(1000.0, -magnitude_base_1000);

	ostringstream oss;
	oss.precision(2 - floor(log10(abs(normalized))));
	oss << fixed << setw(4) << setfill(' ') << normalized << ' ' << MetricPrefix(magnitude_base_1000) << 's';
	return oss.str();
}
