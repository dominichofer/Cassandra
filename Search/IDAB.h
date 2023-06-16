#pragma once
#include "Core/Core.h"
#include "MTDf.h"
#include <utility>

// Iterative Deepening And Broadening
class IDAB : public MTD
{
public:
	template <typename... Args>
	IDAB(Args&&... args) noexcept : MTD(std::forward<Args>(args)...) {}

	ResultTimeNodes Eval(const Position&);
	ResultTimeNodes Eval(const Position&, OpenInterval window, int depth, float confidence_level);
protected:
	Result Eval_N(const Position&, OpenInterval window, int depth, float confidence_level);
};
