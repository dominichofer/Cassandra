#pragma once
#include "Core/Core.h"
#include <string>
#include <optional>

// Maps input to (.., "-1", "+0", "+1", ..)
std::wstring SignedInt(Score);

// Maps input to (.., "-01", "+00", "+01", ..)
std::wstring DoubleDigitSignedInt(Score);

wchar_t MetricPrefix(int magnitude_base_1000);

std::optional<std::size_t> ParseBytes(const std::wstring& bytes);