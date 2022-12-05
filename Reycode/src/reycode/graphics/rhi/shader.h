#pragma once

#include <glad/glad.h>

namespace reycode {
	struct RHI;

	GLuint shader_make(RHI& rhi, const char* vertex_shader_text, const char* fragment_shader);
	void shader_destroy(RHI& rhi, GLuint shader);
}