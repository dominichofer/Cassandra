#pragma once
#include "Core/Core.h"
#include "PrincipalVariation.h"
#include <utility>

// Memory-enhanced Test Driver
class MTD : public PVS
{
public:
	template <typename... Args>
	MTD(Args&&... args) noexcept : PVS(std::forward<Args>(args)...) {}

	ResultTimeNodes Eval(int guess, const Position&);
	ResultTimeNodes Eval(int guess, const Position&, OpenInterval window, int depth, float confidence_level);
protected:
	Result Eval_N(int guess, const Position&, OpenInterval window, int depth, float confidence_level);
};
