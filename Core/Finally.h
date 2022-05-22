#pragma once

template <typename F>
class Finally
{
	F clean;
public:
	Finally(F f) noexcept : clean(f) {}
	~Finally() { clean(); }
};
