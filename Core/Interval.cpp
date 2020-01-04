#include "Interval.h"

const Score Score::Min = Score(-64);
const Score Score::Max = Score(+64);
const Score Score::Infinity = Score(+65);

const ClosedInterval ClosedInterval::Full = ClosedInterval(Score::Min, Score::Max);

const OpenInterval OpenInterval::Full = OpenInterval(-Score::Infinity, +Score::Infinity);