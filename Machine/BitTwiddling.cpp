#pragma once
#include "Machine/BitTwiddling.h"
#include "MacrosHell.h"

[[nodiscard]]
unsigned int BitScanLSB(const uint64_t mask) noexcept
{
	return CountTrailingZeros(mask);
	// BitScanLSB(0) may be undefined.

	#if defined(_MSC_VER)
		unsigned long index = 0;
		_BitScanForward64(&index, mask);
		return index;
	#elif defined(__GNUC__)
		return __builtin_ctzll(mask); // __builtin_ctzll(0) is undefined
	#endif
}

[[nodiscard]]
uint64_t GetLSB(const uint64_t b) noexcept
{
	#ifdef HAS_BLSI
		return _blsi_u64(b);
	#else
	#pragma warning(suppress : 4146)
		return b & -b;
	#endif
}

void RemoveLSB(uint64_t& b) noexcept
{
	#ifdef HAS_BLSR
		detail::RemoveLSB_intrinsic(b);
	#else
		detail::RemoveLSB_generic(b);
	#endif
}

[[nodiscard]]
std::size_t PopCount(uint64_t b) noexcept
{
	#ifdef HAS_POPCNT
		return detail::PopCount_intrinsic(b);
	#else
		return detail::PopCount_generic(b);
	#endif
}

[[nodiscard]]
uint64_t PDep(uint64_t src, uint64_t mask) noexcept
{
	#ifdef HAS_PDEP
		return _pdep_u64(src, mask);
	#else
		uint64_t res = 0;
		for (uint64_t bb = 1; mask != 0u; bb += bb)
		{
			if ((src & bb) != 0u)
				res |= GetLSB(mask);
			RemoveLSB(mask);
		}
		return res;
	#endif
}

[[nodiscard]]
uint64_t PExt(uint64_t src, uint64_t mask) noexcept
{
	#ifdef HAS_PEXT
		return _pext_u64(src, mask);
	#else
		uint64_t res = 0;
		for (uint64_t bb = 1; mask != 0u; bb += bb)
		{
			if ((src & GetLSB(mask)) != 0u)
				res |= bb;
			RemoveLSB(mask);
		}
		return res;
	#endif
}

[[nodiscard]]
uint64_t BSwap(const uint64_t b) noexcept
{
	#if defined(_MSC_VER)
		return _byteswap_uint64(b);
	#elif defined(__GNUC__)
		return __builtin_bswap64(b);
	#endif
}