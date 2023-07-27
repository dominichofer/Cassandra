#pragma once
#include <utility>

#ifdef __NVCC__
namespace std
{
	template <typename T>
	constexpr std::underlying_type_t<T> to_underlying(T value) noexcept
	{
		return static_cast<std::underlying_type_t<T>>(value);
	}
}
#endif
