#include "Intensity.h"

IntensityTable IntensityTable::ExactTill(int empty_count, Intensity then)
{
	IntensityTable ret;
	for (int e = 0; e <= empty_count; e++)
		ret.insert(e, Intensity::Exact());
	for (int e = empty_count + 1; e <= 64; e++)
		ret.insert(e, then);
	return ret;
}

IntensityTable IntensityTable::AllDepthTill(int empty_count, Intensity then)
{
	IntensityTable ret;
	for (int e = 0; e <= empty_count; e++)
	{
		for (int d = 0; d < e; d++)
			ret.insert(e, d);
		ret.insert(e, Intensity::Exact());
	}
	for (int e = empty_count + 1; e <= 64; e++)
		ret.insert(e, then);
	return ret;
}

void IntensityTable::insert(const Intensity& value)
{
	for (auto& t : table)
		t.insert(value);
}

void IntensityTable::insert(int empty_count, const Intensity& value)
{
	table[empty_count].insert(value);
}

std::string to_string(const Intensity& i)
{
	using std::to_string;

	if (i.IsCertain())
		return to_string(i.depth);
	else
		return to_string(i.depth) + " " + to_string(i.certainty);
}
