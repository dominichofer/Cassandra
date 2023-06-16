#include "IO.h"
#include <cmath>
#include <format>

std::string short_time_format(std::chrono::duration<double> duration)
{
	auto seconds = duration.count();
	if (seconds == 0)
		return std::format("{:5.{}f} {}s", 0.0, 3, "");
	int degree = static_cast<int>(std::floor(std::log10(std::abs(seconds)) / 3));
	double normalized = seconds * std::pow(1000.0, -degree);
	auto prefix = MetricPrefix(degree);
	int decimal_places = 3 - static_cast<int>(std::log10(std::abs(normalized)));
	return std::format("{:5.{}f} {}s", normalized, decimal_places, prefix);
}
