#pragma once

#include "reycode/reycode.h"

namespace reycode {
	struct RHI;
	struct Cross_Section_Renderer;
	struct Adaptive_Mesh;

	Cross_Section_Renderer* cross_section_make(Arena& arena, RHI& rhi);
	void cross_section_update(Cross_Section_Renderer& renderer, Arena& tmp_gpu_arena, slice<real> data, vec3 cut_pos, Adaptive_Mesh& mesh, cudaStream_t stream);
	void cross_section_render(Cross_Section_Renderer&, vec3 dir_light, mat4x4& mvp);
	void cross_section_destroy(Cross_Section_Renderer*);
}