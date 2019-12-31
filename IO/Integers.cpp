#include "Integers.h"

std::wstring SignedInt(const Score score)
{
	const std::wstring sign = (score >= 0) ? L"+" : L"-";
	const std::wstring number = std::to_wstring(std::abs(score));
	return sign + number;
}

std::wstring DoubleDigitSignedInt(const Score score)
{
	const std::wstring sign = (score >= 0) ? L"+" : L"-";
	const std::wstring filling_zero = (std::abs(score) < 10) ? L"0" : L"";
	const std::wstring number = std::to_wstring(std::abs(score));
	return sign + filling_zero + number;
}

wchar_t MetricPrefix(int magnitude_base_1000)
{
	switch (magnitude_base_1000)
	{
		case -8: return L'y';
		case -7: return L'z';
		case -6: return L'a';
		case -5: return L'f';
		case -4: return L'p';
		case -3: return L'n';
		case -2: return L'u';
		case -1: return L'm';
		case  0: return L' ';
		case +1: return L'k';
		case +2: return L'M';
		case +3: return L'G';
		case +4: return L'T';
		case +5: return L'P';
		case +6: return L'E';
		case +7: return L'Z';
		case +8: return L'Y';
		default: throw;
	}
}

std::optional<std::size_t> ParseBytes(const std::wstring& bytes)
{
	if (bytes.find(L"EB") != std::wstring::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024 * 1024 * 1024;
	if (bytes.find(L"PB") != std::wstring::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024 * 1024;
	if (bytes.find(L"TB") != std::wstring::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024;
	if (bytes.find(L"GB") != std::wstring::npos) return std::stoll(bytes) * 1024 * 1024 * 1024;
	if (bytes.find(L"MB") != std::wstring::npos) return std::stoll(bytes) * 1024 * 1024;
	if (bytes.find(L"kB") != std::wstring::npos) return std::stoll(bytes) * 1024;
	if (bytes.find(L"B") != std::wstring::npos) return std::stoll(bytes);
	return std::nullopt;
}