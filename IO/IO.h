#pragma once
#include "Core/Core.h"
#include "FForum.h"
#include "Integers.h"
#include "PosScore.h"
#include "File.h"
#include <string>

[[nodiscard]] Field ParseField(const std::string&);

[[nodiscard]] Position ParsePosition_SingleLine(const std::string&) noexcept(false);

[[nodiscard]] std::string short_time_format(std::chrono::duration<double> duration);
