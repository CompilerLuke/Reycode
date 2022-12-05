#pragma once

namespace reycode {
	struct Vertex_Arena;
	struct Vertex_Buffer;
	struct Command_Buffer {};

	enum Draw_Mode {
		DRAW_TRIANGLES,
		DRAW_LINE,
		DRAW_MODE_COUNT
	};

	void cmd_buffer_bind(Command_Buffer&, const Vertex_Buffer&);
	void cmd_buffer_draw(Command_Buffer&, Draw_Mode mode, Vertex_Arena& arena);
}