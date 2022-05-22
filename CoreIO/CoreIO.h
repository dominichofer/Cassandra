#pragma once
#include "Core/Core.h"
#include "IO/IO.h"
#include "Stream.h"
#include <string>

Field ParseField(const std::string&);

Position ParsePosition_SingleLine(const std::string&) noexcept(false);