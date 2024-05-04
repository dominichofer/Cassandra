#include "Algorithm.h"

Result Algorithm::Eval(int guess, const Position& pos, OpenInterval window, Intensity intensity)
{
	return Eval(pos, window, intensity);
}
