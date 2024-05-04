#include "ConfidenceLevel.h"
#include <cstdint>
#include <format>
#include <stdexcept>

bool ConfidenceLevel::IsInfinit() const
{
	return value == inf.value;
}
