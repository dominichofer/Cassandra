#include "Integers.h"
#include <stdexcept>

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