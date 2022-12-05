#include "reycode/reycode.h"
#include <cuda_runtime.h>

namespace reycode {
	Arena make_host_arena(uint64_t size) {
		Arena arena = {};
		arena.capacity = size;
		arena.data = (uint8_t*)malloc(arena.capacity);
		return arena;
	}

	void destroy_host_arena(Arena& arena) {
		free(arena.data);
	}

	Arena make_device_arena(uint64_t size, Cuda_Error& err) {
		Arena arena = {};
		arena.capacity = size;
		cudaMalloc((void**)(&arena.data), size);
		return arena;
	}

	void destroy_device_arena(Arena& arena) {
		cudaFree(arena.data);
	}
}