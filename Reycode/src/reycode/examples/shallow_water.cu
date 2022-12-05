#include "reycode/graphics/viewer/colormap.h"
#include "reycode/graphics/rhi/window.h"
#include "reycode/graphics/rhi/rhi.h"
#include "reycode/graphics/viewer/fpv.h"
#include "reycode/reycode.h"
#include <cuda_gl_interop.h>

namespace reycode {
	constexpr uint32_t GHOST = 1;
	constexpr uint32_t TILE_WIDTH = 32;

	struct Grid_2D {
		uvec2 size;
		vec2 dx;
		vec2 dx2;
		vec2 idx;
		vec2 idx2;

		template<class T>
		INL_CGPU T& operator()(slice<T> data, uvec2 pos) {
			return data[pos.x + pos.y * size.x];
		}
	};

	struct Grid_2D_Renderer {
		RHI& rhi;
		Colormap colormap;

		Vertex_Buffer vertex_buffer;
		cudaGraphicsResource_t ibo_resource;
		cudaGraphicsResource_t vbo_resource;

		GLuint flat_shader;
	};

	static constexpr const char* vertex_shader_text =
		"#version 330\n"
		"uniform mat4 MVP;\n"
		"in vec3 v_pos;\n"
		"in vec3 v_normal;\n"
		"in vec3 v_color;\n"
		"out vec3 f_normal;\n"
		"out vec3 f_pos;\n"
		"out vec3 f_color;\n"
		"void main()\n"
		"{\n"
		"    f_normal = v_normal;\n"
		"    f_pos = v_pos;\n"
		"    f_color = v_color;\n"
		"    gl_Position = MVP * vec4(v_pos, 1.0);\n"
		"}\n";

	static constexpr const char* face_fragment_shader_text =
		"#version 330\n"
		"out vec4 fragment;\n"
		"in vec3 f_normal;\n"
		"in vec3 f_pos;\n"
		"in vec3 f_color;\n"
		"uniform vec3 dir_light;\n"
		"void main()\n"
		"{\n"
		"   vec3 diffuse = vec3(f_color);\n"
		"   vec3 color = diffuse * (0.3 + 0.7*max(vec3(0),dot(-f_normal, dir_light)));\n"
		"   fragment = vec4(color, 1.0);\n"
		"}\n";

	Grid_2D_Renderer* make_grid_2d_renderer(Arena& arena, RHI& rhi, Cuda_Error& err) {
		auto renderer = arena_push<Grid_2D_Renderer>(arena, {rhi});
		renderer->colormap = viridis_cm();

		{
			Vertex_Buffer_Desc desc = {};
			desc.vertex_buffer_size = mb(100);
			desc.index_buffer_size = mb(100);

			renderer->vertex_buffer = make_vertex_buffer(rhi, desc);
		}

		{
			renderer->flat_shader = shader_make(rhi, vertex_shader_text, face_fragment_shader_text);
		}

		err |= cudaGraphicsGLRegisterBuffer(&renderer->vbo_resource, renderer->vertex_buffer.vertices, cudaGraphicsMapFlagsWriteDiscard);
		err |= cudaGraphicsGLRegisterBuffer(&renderer->ibo_resource, renderer->vertex_buffer.indices, cudaGraphicsMapFlagsWriteDiscard);
		return renderer;
	}

	CGPU vec2 grid_to_pos(Grid_2D grid, uvec2 pos) {
		return (vec2(pos)-vec2(GHOST)) * grid.dx - vec2(0.5);
	}

	__global__ void gen_heightmap_kernel(Grid_2D grid, Colormap cm, slice<real> heights, Vertex_Arena_Mapped mapped) {
		uvec2 size = grid.size;
		uvec2 tid = uvec2(threadIdx);
		uvec2 gid = uvec2(blockDim) * uvec2(blockIdx) + tid;
		if (!(gid.x < size.x && gid.y < size.y)) return;

		vec2 pos = grid_to_pos(grid, gid);

		auto vert_id = [&](uvec2 pos) {
			return pos.x + pos.y * size.x;
		};

		real height = grid(heights, gid);

		Vertex& vertex = mapped.vertices[vert_id(gid)];
		vertex.normal = vec3(0, 1, 0);
		vertex.pos.x = pos.x;
		vertex.pos.y = height;
		vertex.pos.z = pos.y;
		vertex.color = color_map(cm, height, 0, 1);
		
		if (!(gid.x < size.x - 1 && gid.y < size.y - 1)) return;

		uint32_t index_id = 6 * (gid.x + gid.y * (size.x-1));
		mapped.indices[index_id + 0] = vert_id(gid + uvec2(0,0));
		mapped.indices[index_id + 1] = vert_id(gid + uvec2(1,0));
		mapped.indices[index_id + 2] = vert_id(gid + uvec2(1,1));
		mapped.indices[index_id + 3] = vert_id(gid + uvec2(0,0));
		mapped.indices[index_id + 4] = vert_id(gid + uvec2(1,1));
		mapped.indices[index_id + 5] = vert_id(gid + uvec2(0,1));
	}

	void grid_2d_update(Grid_2D_Renderer& renderer, Grid_2D grid, slice<real> heights, Cuda_Error& err) {
		RHI& rhi = renderer.rhi;

		uvec3 thread_dim = { 32,32,1 };
		uvec3 block_dim = ceil_div(uvec3(grid.size.x,grid.size.y,1), thread_dim);

		Map_Vertex_Buffer_Cuda_Desc desc = {renderer.vertex_buffer, renderer.vbo_resource, renderer.ibo_resource};
		
		Vertex_Arena_Mapped mapped = vertex_buffer_map_cuda(desc, err);
		DEFER(vertex_buffer_cuda_unmap(desc, err));
		gen_heightmap_kernel <<<block_dim, thread_dim >>>(grid, renderer.colormap, heights, mapped);


		mapped.arena.vertex_count = grid.size.x * grid.size.y;
		mapped.arena.index_count = 6*(grid.size.x-1) * (grid.size.y-1);

		cudaDeviceSynchronize();

		debug_view_device_values(mapped.vertices, 16);
		debug_view_device_values(mapped.indices, 9);
	}

	void grid_2d_draw(Grid_2D_Renderer& renderer, vec3 dir_light, mat4x4& mvp) {
		GLuint edge_shader = renderer.flat_shader;
		glUseProgram(edge_shader);
		glUniformMatrix4fv(glGetUniformLocation(edge_shader, "MVP"), 1, GL_TRUE, (const GLfloat*)&mvp);
		glUniform3fv(glGetUniformLocation(edge_shader, "dir_light"), 1, &dir_light.x);
		
		Command_Buffer cmd_buffer = {};
		cmd_buffer_bind(cmd_buffer, renderer.vertex_buffer);
		cmd_buffer_draw(cmd_buffer, DRAW_TRIANGLES, renderer.vertex_buffer.arena);
	}


	template<class T, class Func>
	__global__ void init_shallow_waves_kernel(Grid_2D grid, slice<T> values, Func func) {
		uvec2 gid = uvec2(blockDim) * uvec2(blockIdx) + uvec2(threadIdx);
		if (!(gid.x < grid.size.x && gid.y < grid.size.y)) return;

		vec2 pos = grid_to_pos(grid, gid);
		grid(values, gid) = func(pos);
	}

	template<class T>
	__global__ void periodic_bc(Grid_2D grid, slice<T> values) {
		uvec2 size = grid.size;
		uint32_t idx = blockDim.x*blockIdx.x + threadIdx.x;

		if (blockIdx.y == 0 && idx < size.y) {
			grid(values, { 0, idx }) = grid(values, { size.x - 2, idx });
			grid(values, { size.x - 1, idx }) = grid(values, { 1, idx });
		}
		else if (idx < size.x) {
			grid(values, { idx, 0 }) = grid(values, { idx, size.y - 2 });
			grid(values, { idx, size.y - 1 }) = grid(values, { idx, 1 });
		}
	}

	__global__ void advance_shallow_waves_kernel(Grid_2D grid, slice<real> heights1, slice<vec2> velocities1, 
															   slice<real> heights0, slice<vec2> velocities0, real dt) {
		uvec2 tid = uvec2(threadIdx);
		uvec2 gid = uvec2(TILE_WIDTH-2) * uvec2(blockIdx) + tid;
		if (!(gid.x < grid.size.x && gid.y < grid.size.y)) return;

		__shared__ real height_sm[TILE_WIDTH][TILE_WIDTH];
		__shared__ vec2 vel_sm[TILE_WIDTH][TILE_WIDTH];

		auto h = [&](int nx, int ny) -> real& {
			return height_sm[tid.y + ny][tid.x + nx]; 
		};
		auto v = [&](int nx, int ny) -> vec2& {
			return vel_sm[tid.y + ny][tid.x+nx];
		};

		h(0,0) = grid(heights0, gid);
		v(0,0) = grid(velocities0, gid);

		__syncthreads();

		if (!(GHOST <= tid.x && tid.x < TILE_WIDTH-GHOST && GHOST <= tid.y && tid.y < TILE_WIDTH - GHOST)) return;

		vec2 idx = 0.5_R * grid.idx;
		vec2 idx2 = grid.idx2;

		const real g = 9.81;
		const real b = 0;
		const real mu = 5e-4;

		real h_dt = 0;
		h_dt += -idx.x * (h(1,0)*v(1,0).x - h(-1,0)*v(-1,0).x);
		h_dt += -idx.y * (h(0,1)*v(0,1).y - h(0,-1)*v(0,-1).y);
			
		vec2 v_dt = {}; 
		v_dt += -g * idx * vec2(h(1, 0) - h(-1, 0), h(0, 1) - h(0, -1));

		v_dt += -v(0, 0).x * idx.x * (v(1,0) - v(-1,0));
		v_dt += -v(0, 0).y * idx.y * (v(0,1) - v(0,-1));

		v_dt += -b * v(0, 0);

		v_dt += mu * idx2.x * (v(-1, 0) - 2 * v(0, 0) + v(1, 0));
		v_dt += mu * idx2.y * (v(0, -1) - 2 * v(0, 0) + v(0, 1));

		grid(velocities1,gid) = v(0, 0) + dt*v_dt;
		grid(heights1,gid) = h(0, 0) + dt*h_dt;
	}

	void advance_shallow_waves(Grid_2D grid, slice<real> heights1, slice<vec2> velocities1,
											slice<real> heights0, slice<vec2> velocities0, real dt) {
		uvec2 size = grid.size;

		{
			uvec3 block_dim = { 32, 1, 1 };
			uvec3 grid_dim = { ceil_div(max(size.x,size.y),block_dim.x), 2, 1 };
			periodic_bc <<<grid_dim, block_dim>>> (grid, heights0);
			periodic_bc <<<grid_dim, block_dim>>> (grid, velocities0);
		}
		
		{
			uvec3 block_dim = { TILE_WIDTH, TILE_WIDTH, 1 };
			uvec3 grid_dim = ceil_div(uvec3(size.x - 2 * GHOST, size.y - 2 * GHOST, 1), {TILE_WIDTH-2*GHOST, TILE_WIDTH-2*GHOST, 1});
			advance_shallow_waves_kernel<<<grid_dim, block_dim>>> (grid, heights1, velocities1, heights0, velocities0, dt);
		}

		/* {
			uvec3 block_dim = { 32, 1, 1 };
			uvec3 grid_dim = { ceil_div(max(size.x,size.y),block_dim.x), 2, 1 };
			periodic_bc << <grid_dim, block_dim >> > (grid, heights1);
			periodic_bc << <grid_dim, block_dim >> > (grid, velocities1);
		}*/
	}

	void destroy_grid_2d_renderer(Grid_2D_Renderer* renderer) {
		destroy_vertex_buffer(renderer->rhi, renderer->vertex_buffer);
		cudaGraphicsUnregisterResource(renderer->vbo_resource);
		cudaGraphicsUnregisterResource(renderer->ibo_resource);
	}

	struct Shallow_Water_Sim {
		
	};

	

	int launch_shallow_water_viewer() {
		uvec2 extent = { 4096,2048 };

		Window_Desc window_desc = {};
		window_desc.title = "Shallow Water";
		window_desc.width = extent.x;
		window_desc.height = extent.y;
		window_desc.validation = true;
		window_desc.vsync = false;

		Cuda_Error err = {};

		Arena perm_arena = make_host_arena(mb(100));
		DEFER(destroy_host_arena(perm_arena));

		Arena device_arena = make_device_arena(mb(100), err);
		DEFER(destroy_device_arena(device_arena));

		Window* window = make_window(perm_arena, window_desc);
		if (!window) return -1;
		DEFER(destroy_window(window));

		RHI& rhi = window_rhi(*window);

		uint32_t n = 1024;

		Grid_2D grid = {};
		grid.size = uvec2(n+2*GHOST, n+2*GHOST);
		grid.dx = vec2(1.0) / (vec2(grid.size)-vec2(2*GHOST+1));
		grid.idx = vec2(1) / grid.dx;
		grid.idx2 = grid.idx * grid.idx;

		uint32_t cell_count = grid.size.x * grid.size.y;
		slice<real> heights0 = arena_push_array<real>(device_arena, cell_count);
		slice<real> heights1 = arena_push_array<real>(device_arena, cell_count);

		slice<vec2> velocities0 = arena_push_array<vec2>(device_arena, cell_count);
		slice<vec2> velocities1 = arena_push_array<vec2>(device_arena, cell_count);

		{
			uvec3 thread_dim = { 32,32,1 };
			uvec3 block_dim = ceil_div(uvec3(grid.size.x, grid.size.y, 1), thread_dim);

			real f = 10.0_R;
			const real A = 20;

			init_shallow_waves_kernel<<<block_dim, thread_dim >>> (grid, heights0, [=] CGPU (vec2 pos) {
				return 0.5 + 0.5*exp(-A*pow(length(pos),2));
			});

			init_shallow_waves_kernel<<<block_dim, thread_dim >>> (grid, velocities0, [=] CGPU(vec2 pos) {
				return vec2(); 
			});
		}

		Grid_2D_Renderer* renderer = make_grid_2d_renderer(perm_arena, rhi, err);
		DEFER(destroy_grid_2d_renderer(renderer));

		Window_Input input = { window };

		FPV fpv = {};
		fpv.view_pos.y = 3;
		fpv.pitch = PI / 2;

		bool playing = false;

		real t = get_time();
		while (window_is_open(*window)) {
			real dt = get_time() - t;
			t = get_time();

			window_input_poll(input);
			fpv_update(fpv, input.state, dt);
			window_input_capture_cursor(input, fpv.capture_cursor);
			if (input_key_pressed(input.state, KEY_SPACE)) playing = !playing;

			mat4x4 mvp = fpv_proj_mat(fpv, extent) * fpv_view_mat(fpv);
			vec3 dir_light = normalize(vec3(-1, -1, -1));

			uint32_t sub_timesteps = 100;

			if (playing) {
				for (uint32_t i = 0; i < sub_timesteps; i++) {
					advance_shallow_waves(grid, heights1, velocities1, heights0, velocities0, 1e-5);
					std::swap(heights0, heights1);
					std::swap(velocities0, velocities1);
				}
			}

			grid_2d_update(*renderer, grid, heights0, err);

			{
				glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				glEnable(GL_DEPTH_TEST);
				glDisable(GL_CULL_FACE);

				grid_2d_draw(*renderer, dir_light, mvp);

				window_draw(*window);
			}
		}

		return 0;
	}
}