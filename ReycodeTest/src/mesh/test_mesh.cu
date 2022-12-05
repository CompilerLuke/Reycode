#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "reycode/mesh/mesh.h"
#include <cuda_runtime.h>

using namespace reycode;

/*
TEST(CoarseGridTest, make_coarse_grid) {
	cudaStream_t stream = 0;

	const size3 dims = { 2,2,2 };
	const uint32_t length = 8;
	const uvec3 tile_size = { 10, 10, 10 };

	vec3* positions = nullptr;
	cudaMalloc((void**)&positions, sizeof(vec3) * dims.x * dims.y * dims.z);



	make_coarse_grid(stream, positions, dims, tile_size);
	cudaDeviceSynchronize();

	vec3 expected[length] = {
		vec3(-5,-5,-5),
		vec3(5,-5,-5),
		vec3(-5,5,-5),
		vec3(5,5,-5),
		vec3(-5,-5,5),
		vec3(5,-5,5),
		vec3(-5,5,5),
		vec3(5,5,5),
	};

	vec3 got[length];
	cudaMemcpy(got, positions, sizeof(got), cudaMemcpyDeviceToHost);

	for (int i = 0; i < length; i++) ASSERT_EQ(expected[i], got[i]);
}
*/