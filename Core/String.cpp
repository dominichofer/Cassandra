#pragma once
#include "String.h"
#include <format>
#include <stdexcept>

std::string HH_MM_SS(std::chrono::duration<double> duration)
{
	int hours = std::chrono::duration_cast<std::chrono::hours>(duration).count();
	int minutes = std::chrono::duration_cast<std::chrono::minutes>(duration).count() % 60;
	int seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count() % 60;
	int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;

	std::string hour_str = "";
	if (hours > 0)
		hour_str = std::to_string(hours) + ":";

	std::string minute_str  = "";
	if (hours > 0)
		minute_str = std::format("{:02d}:", minutes);
	else if (minutes > 0)
		minute_str = std::format("{:d}:", minutes);

	std::string second_str = "";
	if (hours > 0 || minutes > 0)
		second_str = std::format("{:02d}.", seconds);
	else
		second_str = std::format("{:d}.", seconds);

	std::string millisecond_str = std::format("{:03d}", milliseconds);

	return hour_str + minute_str + second_str + millisecond_str;
}

std::string ShortTimeString(std::chrono::duration<double> duration)
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

std::string MetricPrefix(int magnitude_base_1000) noexcept(false)
{
	// From: https://en.wikipedia.org/wiki/Metric_prefix

	switch (magnitude_base_1000)
	{
	case -10: return "q";
	case -9: return "r";
	case -8: return "y";
	case -7: return "z";
	case -6: return "a";
	case -5: return "f";
	case -4: return "p";
	case -3: return "n";
	case -2: return "u";
	case -1: return "m";
	case  0: return "";
	case +1: return "k";
	case +2: return "M";
	case +3: return "G";
	case +4: return "T";
	case +5: return "P";
	case +6: return "E";
	case +7: return "Z";
	case +8: return "Y";
	case +9: return "R";
	case +10: return "Q";
	default: throw std::runtime_error("Magnitude out of range.");
	}
}

std::size_t ParseBytes(const std::string& bytes) noexcept(false)
{
	if (bytes.find("QB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024;
	if (bytes.find("RB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024;
	if (bytes.find("YB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024;
	if (bytes.find("ZB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024;
	if (bytes.find("EB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024 * 1024 * 1024;
	if (bytes.find("PB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024 * 1024;
	if (bytes.find("TB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024;
	if (bytes.find("GB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024;
	if (bytes.find("MB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024;
	if (bytes.find("kB") != std::string::npos) return std::stoll(bytes) * 1024;
	if (bytes.find('B') != std::string::npos) return std::stoll(bytes);
	throw std::runtime_error("Could not parse byte string.");
}
