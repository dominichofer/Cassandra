#include "IO.h"
#include <cmath>

std::string short_time_format(std::chrono::duration<double> duration)
{
	using namespace std;

	auto seconds = duration.count();
	if (seconds == 0)
		return fmt::format("{:5.{}f} {}s", 0.0, 3, "");
	int degree = static_cast<int>(floor(log10(abs(seconds)) / 3));
	double normalized = seconds * pow(1000.0, -degree);
	auto prefix = MetricPrefix(degree);
	int decimal_places = 3 - static_cast<int>(log10(abs(normalized)));
	return fmt::format("{:5.{}f} {}s", normalized, decimal_places, prefix);
}
