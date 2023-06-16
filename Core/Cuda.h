#include <cstdint>

#ifdef __CUDA_ARCH__
	#define CUDA_CALLABLE __host__ __device__
#else
	#define CUDA_CALLABLE
#endif
