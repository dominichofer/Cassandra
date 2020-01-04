#include "Interval.h"

const Score Score::Min = Score(-64);
const Score Score::Max = Score(+64);
const Score Score::Infinity = Score(+65);

const InclusiveInterval InclusiveInterval::Full = InclusiveInterval(Score::Min, Score::Max);

const ExclusiveInterval ExclusiveInterval::Full = ExclusiveInterval(-Score::Infinity, +Score::Infinity);