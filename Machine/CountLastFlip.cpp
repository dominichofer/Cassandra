#include "CountLastFlip.h"
#include "Machine/BitTwiddling.h"
#include "Machine/Flips.h"

class LastFlipCounter
{
public:
	LastFlipCounter();

	int CountLastFlip(uint64_t P, uint8_t move) const;

private:
	uint64_t Flips(uint64_t P, uint8_t move);

	uint8_t CLF_0[256]{};
	uint8_t CLF_1[256]{};
	uint8_t CLF_2[256]{};
	uint8_t CLF_3[256]{};
	uint8_t CLF_4[256]{};
	uint8_t CLF_5[256]{};
	uint8_t CLF_6[256]{};
	uint8_t CLF_7[256]{};
};

LastFlipCounter::LastFlipCounter()
{
	for (uint64_t P = 0; P < 256; P++)
	{
		CLF_0[P] = static_cast<uint8_t>(2 * PopCount(Flips(P, 0)));
		CLF_1[P] = static_cast<uint8_t>(2 * PopCount(Flips(P, 1)));
		CLF_2[P] = static_cast<uint8_t>(2 * PopCount(Flips(P, 2)));
		CLF_3[P] = static_cast<uint8_t>(2 * PopCount(Flips(P, 3)));
		CLF_4[P] = static_cast<uint8_t>(2 * PopCount(Flips(P, 4)));
		CLF_5[P] = static_cast<uint8_t>(2 * PopCount(Flips(P, 5)));
		CLF_6[P] = static_cast<uint8_t>(2 * PopCount(Flips(P, 6)));
		CLF_7[P] = static_cast<uint8_t>(2 * PopCount(Flips(P, 7)));
	}
}

uint64_t LastFlipCounter::Flips(const uint64_t i, const uint8_t move)
{
	//if (TestBit(i, move))
	//	return 0;
	const uint64_t P = i & ~Bit(move);
	const uint64_t O = ~i & ~Bit(move);
	return ::Flips(P, O, move);
}

int LastFlipCounter::CountLastFlip(uint64_t P, uint8_t move) const
{
	switch (move)
	{
	case  0: return CLF_0[P & 0xFFui64] + CLF_0[PExt(P, 0x0101010101010101ui64)] + CLF_0[PExt(P, 0x8040201008040201ui64)];
	case  1: return CLF_1[P & 0xFFui64] + CLF_0[PExt(P, 0x0202020202020202ui64)] + CLF_0[PExt(P, 0x0080402010080402ui64)];
	case  2: return CLF_2[P & 0xFFui64] + CLF_0[PExt(P, 0x0404040404040404ui64)] + CLF_2[((P & 0x0000804020110A04ui64) * 0x0101010101010101ui64) >> 56];
	case  3: return CLF_3[P & 0xFFui64] + CLF_0[PExt(P, 0x0808080808080808ui64)] + CLF_3[((P & 0x0000008041221408ui64) * 0x0101010101010101ui64) >> 56];
	case  4: return CLF_4[P & 0xFFui64] + CLF_0[PExt(P, 0x1010101010101010ui64)] + CLF_4[((P & 0x0000000182442810ui64) * 0x0101010101010101ui64) >> 56];
	case  5: return CLF_5[P & 0xFFui64] + CLF_0[PExt(P, 0x2020202020202020ui64)] + CLF_5[((P & 0x0000010204885020ui64) * 0x0101010101010101ui64) >> 56];
	case  6: return CLF_6[P & 0xFFui64] + CLF_0[PExt(P, 0x4040404040404040ui64)] + CLF_0[PExt(P, 0x0001020408102040ui64)];
	case  7: return CLF_7[P & 0xFFui64] + CLF_0[PExt(P, 0x8080808080808080ui64)] + CLF_0[PExt(P, 0x0102040810204080ui64)];

	case  8: return CLF_0[BExtr(P, 8, 8)] + CLF_1[PExt(P, 0x0101010101010101ui64)] + CLF_0[PExt(P, 0x4020100804020100ui64)];
	case  9: return CLF_1[BExtr(P, 8, 8)] + CLF_1[PExt(P, 0x0202020202020202ui64)] + CLF_0[PExt(P, 0x8040201008040200ui64)];
	case 10: return CLF_2[BExtr(P, 8, 8)] + CLF_1[PExt(P, 0x0404040404040404ui64)] + CLF_2[((P & 0x00804020110A0400ui64) * 0x0101010101010101ui64) >> 56];
	case 11: return CLF_3[BExtr(P, 8, 8)] + CLF_1[PExt(P, 0x0808080808080808ui64)] + CLF_3[((P & 0x0000804122140800ui64) * 0x0101010101010101ui64) >> 56];
	case 12: return CLF_4[BExtr(P, 8, 8)] + CLF_1[PExt(P, 0x1010101010101010ui64)] + CLF_4[((P & 0x0000018244281000ui64) * 0x0101010101010101ui64) >> 56];
	case 13: return CLF_5[BExtr(P, 8, 8)] + CLF_1[PExt(P, 0x2020202020202020ui64)] + CLF_5[((P & 0x0001020488502000ui64) * 0x0101010101010101ui64) >> 56];
	case 14: return CLF_6[BExtr(P, 8, 8)] + CLF_1[PExt(P, 0x4040404040404040ui64)] + CLF_0[PExt(P, 0x0102040810204000ui64)];
	case 15: return CLF_7[BExtr(P, 8, 8)] + CLF_1[PExt(P, 0x8080808080808080ui64)] + CLF_0[PExt(P, 0x0204081020408000ui64)];

	case 16: return CLF_0[BExtr(P, 16, 8)] + CLF_2[PExt(P, 0x0101010101010101ui64)] + CLF_2[PExt(P, 0x2010080402010204ui64)];
	case 17: return CLF_1[BExtr(P, 16, 8)] + CLF_2[PExt(P, 0x0202020202020202ui64)] + CLF_2[PExt(P, 0x4020100804020408ui64)];
	case 18: return CLF_2[BExtr(P, 16, 8)] + CLF_2[PExt(P, 0x0404040404040404ui64)] + CLF_2[PExt(P, 0x8040201008040201ui64)] + CLF_2[PExt(P, 0x0000000102040810ui64)];
	case 19: return CLF_3[BExtr(P, 16, 8)] + CLF_2[PExt(P, 0x0808080808080808ui64)] + CLF_2[PExt(P, 0x0080402010080402ui64)] + CLF_2[PExt(P, 0x0000010204081020ui64)];
	case 20: return CLF_4[BExtr(P, 16, 8)] + CLF_2[PExt(P, 0x1010101010101010ui64)] + CLF_2[PExt(P, 0x0000804020100804ui64)] + CLF_2[PExt(P, 0x0001020408102040ui64)];
	case 21: return CLF_5[BExtr(P, 16, 8)] + CLF_2[PExt(P, 0x2020202020202020ui64)] + CLF_2[PExt(P, 0x0000008040201008ui64)] + CLF_2[PExt(P, 0x0102040810204080ui64)];
	case 22: return CLF_6[BExtr(P, 16, 8)] + CLF_2[PExt(P, 0x4040404040404040ui64)] + CLF_2[PExt(P, 0x0204081020402010ui64)];
	case 23: return CLF_7[BExtr(P, 16, 8)] + CLF_2[PExt(P, 0x8080808080808080ui64)] + CLF_2[PExt(P, 0x0408102040804020ui64)];

	case 24: return CLF_0[BExtr(P, 24, 8)] + CLF_3[PExt(P, 0x0101010101010101ui64)] + CLF_3[PExt(P, 0x1008040201020408ui64)];
	case 25: return CLF_1[BExtr(P, 24, 8)] + CLF_3[PExt(P, 0x0202020202020202ui64)] + CLF_3[PExt(P, 0x2010080402040810ui64)];
	case 26: return CLF_2[BExtr(P, 24, 8)] + CLF_3[PExt(P, 0x0404040404040404ui64)] + CLF_2[PExt(P, 0x4020100804020100ui64)] + CLF_3[PExt(P, 0x0000010204081020ui64)];
	case 27: return CLF_3[BExtr(P, 24, 8)] + CLF_3[PExt(P, 0x0808080808080808ui64)] + CLF_3[PExt(P, 0x8040201008040201ui64)] + CLF_3[PExt(P, 0x0001020408102040ui64)];
	case 28: return CLF_4[BExtr(P, 24, 8)] + CLF_3[PExt(P, 0x1010101010101010ui64)] + CLF_3[PExt(P, 0x0080402010080402ui64)] + CLF_3[PExt(P, 0x0102040810204080ui64)];
	case 29: return CLF_5[BExtr(P, 24, 8)] + CLF_3[PExt(P, 0x2020202020202020ui64)] + CLF_3[PExt(P, 0x0000804020100804ui64)] + CLF_2[PExt(P, 0x0204081020408000ui64)];
	case 30: return CLF_6[BExtr(P, 24, 8)] + CLF_3[PExt(P, 0x4040404040404040ui64)] + CLF_3[PExt(P, 0x0408102040201008ui64)];
	case 31: return CLF_7[BExtr(P, 24, 8)] + CLF_3[PExt(P, 0x8080808080808080ui64)] + CLF_3[PExt(P, 0x0810204080402010ui64)];

	case 32: return CLF_0[BExtr(P, 32, 8)] + CLF_4[PExt(P, 0x0101010101010101ui64)] + CLF_4[PExt(P, 0x0804020102040810ui64)];
	case 33: return CLF_1[BExtr(P, 32, 8)] + CLF_4[PExt(P, 0x0202020202020202ui64)] + CLF_4[PExt(P, 0x1008040204081020ui64)];
	case 34: return CLF_2[BExtr(P, 32, 8)] + CLF_4[PExt(P, 0x0404040404040404ui64)] + CLF_2[PExt(P, 0x2010080402010000ui64)] + CLF_4[PExt(P, 0x0001020408102040ui64)];
	case 35: return CLF_3[BExtr(P, 32, 8)] + CLF_4[PExt(P, 0x0808080808080808ui64)] + CLF_3[PExt(P, 0x4020100804020100ui64)] + CLF_4[PExt(P, 0x0102040810204080ui64)];
	case 36: return CLF_4[BExtr(P, 32, 8)] + CLF_4[PExt(P, 0x1010101010101010ui64)] + CLF_4[PExt(P, 0x8040201008040201ui64)] + CLF_3[PExt(P, 0x0204081020408000ui64)];
	case 37: return CLF_5[BExtr(P, 32, 8)] + CLF_4[PExt(P, 0x2020202020202020ui64)] + CLF_4[PExt(P, 0x0080402010080402ui64)] + CLF_2[PExt(P, 0x0408102040800000ui64)];
	case 38: return CLF_6[BExtr(P, 32, 8)] + CLF_4[PExt(P, 0x4040404040404040ui64)] + CLF_4[PExt(P, 0x0810204020100804ui64)];
	case 39: return CLF_7[BExtr(P, 32, 8)] + CLF_4[PExt(P, 0x8080808080808080ui64)] + CLF_4[PExt(P, 0x1020408040201008ui64)];

	case 40: return CLF_0[BExtr(P, 40, 8)] + CLF_5[PExt(P, 0x0101010101010101ui64)] + CLF_5[PExt(P, 0x0402010204081020ui64)];
	case 41: return CLF_1[BExtr(P, 40, 8)] + CLF_5[PExt(P, 0x0202020202020202ui64)] + CLF_5[PExt(P, 0x0804020408102040ui64)];
	case 42: return CLF_2[BExtr(P, 40, 8)] + CLF_5[PExt(P, 0x0404040404040404ui64)] + CLF_2[PExt(P, 0x1008040201000000ui64)] + CLF_5[PExt(P, 0x0102040810204080ui64)];
	case 43: return CLF_3[BExtr(P, 40, 8)] + CLF_5[PExt(P, 0x0808080808080808ui64)] + CLF_3[PExt(P, 0x2010080402010000ui64)] + CLF_4[PExt(P, 0x0204081020408000ui64)];
	case 44: return CLF_4[BExtr(P, 40, 8)] + CLF_5[PExt(P, 0x1010101010101010ui64)] + CLF_4[PExt(P, 0x4020100804020100ui64)] + CLF_3[PExt(P, 0x0408102040800000ui64)];
	case 45: return CLF_5[BExtr(P, 40, 8)] + CLF_5[PExt(P, 0x2020202020202020ui64)] + CLF_5[PExt(P, 0x8040201008040201ui64)] + CLF_2[PExt(P, 0x0810204080000000ui64)];
	case 46: return CLF_6[BExtr(P, 40, 8)] + CLF_5[PExt(P, 0x4040404040404040ui64)] + CLF_5[PExt(P, 0x1020402010080402ui64)];
	case 47: return CLF_7[BExtr(P, 40, 8)] + CLF_5[PExt(P, 0x8080808080808080ui64)] + CLF_5[PExt(P, 0x2040804020100804ui64)];

	case 48: return CLF_0[BExtr(P, 48, 8)] + CLF_6[PExt(P, 0x0101010101010101ui64)] + CLF_6[PExt(P, 0x0001020408102040ui64)];
	case 49: return CLF_1[BExtr(P, 48, 8)] + CLF_6[PExt(P, 0x0202020202020202ui64)] + CLF_6[PExt(P, 0x0002040810204080ui64)];
	case 50: return CLF_2[BExtr(P, 48, 8)] + CLF_6[PExt(P, 0x0404040404040404ui64)] + CLF_2[((P & 0x00040A1120408000ui64) * 0x0101010101010101ui64) >> 56];
	case 51: return CLF_3[BExtr(P, 48, 8)] + CLF_6[PExt(P, 0x0808080808080808ui64)] + CLF_3[((P & 0x0008142241800000ui64) * 0x0101010101010101ui64) >> 56];
	case 52: return CLF_4[BExtr(P, 48, 8)] + CLF_6[PExt(P, 0x1010101010101010ui64)] + CLF_4[((P & 0x0010284482010000ui64) * 0x0101010101010101ui64) >> 56];
	case 53: return CLF_5[BExtr(P, 48, 8)] + CLF_6[PExt(P, 0x2020202020202020ui64)] + CLF_5[((P & 0x0020508804020100ui64) * 0x0101010101010101ui64) >> 56];
	case 54: return CLF_6[BExtr(P, 48, 8)] + CLF_6[PExt(P, 0x4040404040404040ui64)] + CLF_6[PExt(P, 0x0040201008040201ui64)];
	case 55: return CLF_7[BExtr(P, 48, 8)] + CLF_6[PExt(P, 0x8080808080808080ui64)] + CLF_6[PExt(P, 0x0080402010080402ui64)];

	case 56: return CLF_0[P >> 56] + CLF_7[PExt(P, 0x0101010101010101ui64)] + CLF_7[PExt(P, 0x0102040810204080ui64)];
	case 57: return CLF_1[P >> 56] + CLF_7[PExt(P, 0x0202020202020202ui64)] + CLF_6[PExt(P, 0x0204081020408000ui64)];
	case 58: return CLF_2[P >> 56] + CLF_7[PExt(P, 0x0404040404040404ui64)] + CLF_2[((P & 0x040A112040800000ui64) * 0x0101010101010101ui64) >> 56];
	case 59: return CLF_3[P >> 56] + CLF_7[PExt(P, 0x0808080808080808ui64)] + CLF_3[((P & 0x0814224180000000ui64) * 0x0101010101010101ui64) >> 56];
	case 60: return CLF_4[P >> 56] + CLF_7[PExt(P, 0x1010101010101010ui64)] + CLF_4[((P & 0x1028448201000000ui64) * 0x0101010101010101ui64) >> 56];
	case 61: return CLF_5[P >> 56] + CLF_7[PExt(P, 0x2020202020202020ui64)] + CLF_5[((P & 0x2050880402010000ui64) * 0x0101010101010101ui64) >> 56];
	case 62: return CLF_6[P >> 56] + CLF_7[PExt(P, 0x4040404040404040ui64)] + CLF_6[PExt(P, 0x4020100804020100ui64)];
	case 63: return CLF_7[P >> 56] + CLF_7[PExt(P, 0x8080808080808080ui64)] + CLF_7[PExt(P, 0x8040201008040201ui64)];
	}
}

static LastFlipCounter lfc;

int CountLastFlip(uint64_t P, uint8_t move)
{
	return lfc.CountLastFlip(P, move);
}