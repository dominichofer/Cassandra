#pragma once
#include "Core/Core.h"
#include "IO/IO.h"
#include "Stream.h"
#include <string>
#include <string_view>
#include <vector>

Field ParseField(std::string_view);

Position ParsePosition_SingleLine(std::string_view) noexcept(false);
std::vector<Position> ParsePositionFile(const std::string&) noexcept(false);

PosScore ParsePosScore_SingleLine(const std::string&) noexcept(false);
std::vector<PosScore> ParsePosScoreFile(const std::string&) noexcept(false);