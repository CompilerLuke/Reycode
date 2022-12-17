#include "draw.h"
#include "vertex_buffer.h"
#include <glad/glad.h>

namespace reycode {
    int DRAW_MODE_TO_GL[DRAW_MODE_COUNT] = { GL_TRIANGLES, GL_LINES, GL_POINTS };

    void cmd_buffer_bind(Command_Buffer& buffer, const Vertex_Buffer& vertex_buffer) {
        glBindVertexArray(vertex_buffer.vao);
    }

    void cmd_buffer_draw(Command_Buffer& buffer, Draw_Mode mode, Vertex_Arena& arena) {
        int gl_mode = DRAW_MODE_TO_GL[mode];
        glDrawElementsBaseVertex(gl_mode, arena.index_count, GL_UNSIGNED_INT, (void*)(arena.index_offset * sizeof(uint32_t)), arena.vertex_offset);
    }
}