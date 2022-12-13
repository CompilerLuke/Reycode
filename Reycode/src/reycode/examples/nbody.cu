#include "reycode/graphics/viewer/colormap.h"
#include "reycode/graphics/rhi/window.h"
#include "reycode/graphics/rhi/rhi.h"
#include "reycode/graphics/viewer/fpv.h"
#include "reycode/reycode.h"
#include <cuda_gl_interop.h>
#include <curand.h>

namespace reycode {
	struct Particle_AOS {
		vec3 position;
		vec3 velocity;
		vec3 acceleration;
		real mass;
	};

	struct AABB {
		vec2 min;
		vec2 max;
	};

	constexpr uint32_t MAX_PARTICLES_PER_QUAD = 1; // 2048;

	using quad_tree_node_handle = uint32_t;
	using quad_tree_data_handle = uint32_t;

	constexpr quad_tree_data_handle NODE_HAS_CHILDREN_FLAG = INT_MAX;

	struct Quad_Tree_Node {
		AABB aabb;
		vec3 center_of_mass;
		real mass;
		uint32_t particle_count;
		uint32_t node_count;
		quad_tree_data_handle data;
	};

	CGPU bool quad_node_is_leaf(Quad_Tree_Node node) {
		return node.particle_count > 0; 
	}

	struct N_Body_Tree {
		AABB aabb;
		slice<Quad_Tree_Node> nodes;
	};

	struct N_Body_Tree_Divide_In {
		N_Body_Tree& tree;
		uint32_t& node_counter;
		uint32_t& particle_counter;
		Arena& transient_arena;
		slice<Particle_AOS> particles;
		slice<Particle_AOS> particles_tmp;
		slice<Particle_AOS> particles_out;
		AABB aabb;
		uint32_t depth = 0;
	};

	uint32_t quad_index(vec2 min, vec2 dim_half, vec2 pos) {
		vec2 rel_pos = pos - min; // Ignore z-component
		vec2 sub_grid = vec2(rel_pos.x / dim_half.x, rel_pos.y / dim_half.y);

		const real BIAS = 1e-5;

		sub_grid.x = clamp(sub_grid.x, 0.0_R, 2 - BIAS); // Avoid floating point errors, causing point to lie just outside of quad tree node
		sub_grid.y = clamp(sub_grid.y, 0.0_R, 2 - BIAS);
		uint32_t index = 2 * (uint32_t)sub_grid.y + (uint32_t)sub_grid.x;
		assert(index < 4);
		return index;
	}

	AABB update_aabb(slice<Particle_AOS> particles) {
		AABB aabb = { vec2(FLT_MAX), vec2(-FLT_MAX) };
		for (const Particle_AOS& particle : particles) {
			vec3 pos = particle.position;

			aabb.min.x = min(aabb.min.x, pos.x);
			aabb.min.y = min(aabb.min.y, pos.y);
			aabb.max.x = max(aabb.max.x, pos.x);
			aabb.max.y = max(aabb.max.y, pos.y);
		}
		return aabb;
	}

	void build_quadtree(N_Body_Tree_Divide_In& in) {
		if (in.particles.length == 0) return;

		uint32_t id = in.node_counter++;
		Quad_Tree_Node& node = in.tree.nodes[id];
		node = {};
		slice<Particle_AOS> particles = in.particles;
		AABB aabb = in.aabb;

		node.aabb = aabb;

		if (particles.length <= MAX_PARTICLES_PER_QUAD) {
			node.particle_count = particles.length;
			node.data = in.particle_counter;
			node.node_count = 0;
			in.particle_counter += particles.length;

			real weight = 0;
			node.center_of_mass = vec3();
			node.mass = 0;

			for (uint32_t i = 0; i < particles.length; i++) {
				Particle_AOS& particle = particles[i];

				node.center_of_mass += particle.mass * particle.position;
				node.mass += particle.mass;
				if (in.particles.data != in.particles_out.data) in.particles_out[i] = in.particles[i];
			}

			node.center_of_mass = node.center_of_mass / node.mass;
			return;
		}
		
		node.data = NODE_HAS_CHILDREN_FLAG;


		vec2 dim = aabb.max - aabb.min;
		vec2 dim_half = 0.5_R * dim;

		uint32_t n = particles.length;
		slice<uint32_t> prefix_sum = arena_push_array<uint32_t>(in.transient_arena, 4 * n + 1);

		slice<Particle_AOS> particles2 = in.particles_tmp;

		memset(prefix_sum.data, 0, sizeof(uint32_t) * prefix_sum.length);

		node.center_of_mass = vec3();
		node.mass = 0;

		for (uint32_t i = 0; i < n; i++) {
			Particle_AOS particle = particles[i];
			node.center_of_mass += particle.mass * particle.position;
			node.mass += particle.mass;

			uint32_t index = quad_index(aabb.min, dim_half, particle.position.xy());
			prefix_sum[index*n + i] = 1;
		}

		node.center_of_mass = node.center_of_mass / node.mass;

		uint32_t sum = 0;
		for (uint32_t i = 0; i < prefix_sum.length; i++) {
			uint32_t incr = prefix_sum[i];
			prefix_sum[i] = sum;
			sum += incr;
		}

		slice<Particle_AOS> children[4];
		for (uint32_t k = 0; k < 4; k++) {
			uint32_t length = prefix_sum[n*(k + 1)] - prefix_sum[n*k];
			children[k] = arena_push_array<Particle_AOS>(in.transient_arena, length);
		}

		for (uint32_t i = 0; i < 4 * particles.length; i++) {
			if (prefix_sum[i] != prefix_sum[i + 1]) {
				particles2[prefix_sum[i]] = particles[i % n];
			}
		}

		for (uint32_t k = 0; k < 4; k++) {
			N_Body_Tree_Divide_In sub = in;

			uint32_t offset = prefix_sum[n * k];
			uint32_t length = prefix_sum[n * (k + 1)] - prefix_sum[n * k];

			sub.particles = subslice(particles2, offset, length);
			sub.particles_tmp = subslice(particles, offset, length);
			sub.particles_out = subslice(in.particles_out, offset, length);

			sub.aabb.min = aabb.min + vec2(real(k % 2), real(k / 2)) * dim_half;
			sub.aabb.max = sub.aabb.min + dim_half;

			sub.depth = in.depth + 1;
			build_quadtree(sub);
		};
		node.node_count = in.node_counter - id - 1;
	}


	N_Body_Tree build_quad_tree(Arena& tmp_arena, AABB aabb, slice<Particle_AOS> particles_in_gpu, slice<Particle_AOS> particles_out_gpu, slice<Quad_Tree_Node>& nodes_gpu, Cuda_Error& err) {
		err |= cudaDeviceSynchronize();

		slice<Particle_AOS> particles_cpu = arena_push_array<Particle_AOS>(tmp_arena, particles_in_gpu.length);
		err |= cudaMemcpy(particles_cpu.data, particles_in_gpu.data, (uint64_t)sizeof(Particle_AOS) * particles_in_gpu.length, cudaMemcpyDeviceToHost);

		slice<Particle_AOS> particles2_cpu = arena_push_array<Particle_AOS>(tmp_arena, particles_in_gpu.length);
		
		uint32_t max_nodes = 2*particles_in_gpu.length;
		
		N_Body_Tree tree = {};
		tree.nodes = arena_push_array<Quad_Tree_Node>(tmp_arena, max_nodes);
		tree.aabb = aabb;

		uint32_t node_counter = 0;
		uint32_t particle_counter = 0;
		N_Body_Tree_Divide_In task = { tree, node_counter, particle_counter, tmp_arena };
		task.particles = particles_cpu;
		task.particles_tmp = particles2_cpu;
		task.particles_out = particles2_cpu;
		task.aabb = tree.aabb;

		build_quadtree(task);

		tree.nodes.length = node_counter;

		err |= cudaMemcpy(particles_out_gpu.data, particles2_cpu.data, (uint64_t)sizeof(Particle_AOS) * particles_in_gpu.length, cudaMemcpyHostToDevice);
		err |= cudaMemcpy(nodes_gpu.data, tree.nodes.data, (uint64_t)sizeof(Quad_Tree_Node) * tree.nodes.length, cudaMemcpyHostToDevice);

		return tree;
	}

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

	__global__ void gen_particle_vertex_buffer_kernel(Vertex_Arena_Mapped mapped, slice<Particle_AOS> particles) {
		uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= particles.length) return;

		constexpr Vertex quad_vertices[4] = {
			{vec3(-0.5,-0.5,0), vec3(0,0,1), vec3(1,1,1) },
			{vec3(0.5,-0.5,0), vec3(0,0,1), vec3(1,1,1) },
			{vec3(0.5,0.5,0), vec3(0,0,1), vec3(1,1,1) },
			{vec3(-0.5,0.5,0), vec3(0,0,1), vec3(1,1,1) },
		};
		constexpr uint32_t quad_indices[6] = { 0,1,2,0,2,3 };

		vec3 pos = particles[i].position;

		real size = 1e-3 * particles[i].mass;

		for (uint32_t k = 0; k < 4; k++) {
			mapped.vertices[4 * i + k] = quad_vertices[k];
			mapped.vertices[4 * i + k].pos = size * quad_vertices[k].pos + pos;
		}

		for (uint32_t k = 0; k < 6; k++) mapped.indices[6 * i + k] = 4 * i + quad_indices[k];
	}

	__global__ void gen_quad_tree_buffer_kernel(Vertex_Arena_Mapped mapped, slice<Quad_Tree_Node> nodes) {
		uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= nodes.length) return;

		constexpr Vertex quad_vertices[4] = {
			{vec3(0,0,0), vec3(0,0,1), vec3(1,1,1) },
			{vec3(1,0,0), vec3(0,0,1), vec3(1,1,1)},
			{vec3(1,1,0), vec3(0,0,1), vec3(1,1,1) },
			{vec3(0,1,0), vec3(0,0,1), vec3(1,1,1) },
		};
		constexpr uint32_t quad_indices[8] = { 0,1,1,2,2,3,3,0 };

		AABB aabb = nodes[i].aabb;
		vec3 pos = vec3(aabb.min, 0);
		vec3 size = vec3(aabb.max, 0) - vec3(aabb.min, 0);

		for (uint32_t k = 0; k < 4; k++) {
			vec3 vpos = size * quad_vertices[k].pos + pos;

			Vertex& v = mapped.vertices[4 * i + k];
			v = quad_vertices[k];
			v.pos = vpos;
		}

		for (uint32_t k = 0; k < 8; k++) mapped.indices[8 * i + k] = 4 * i + quad_indices[k];
	}

	Vertex_Arena gen_particle_vertex_buffer(Vertex_Arena_Mapped& mapped_arena, Arena& tmp_arena, slice<Particle_AOS> particles) {
		uint32_t n = particles.length;

		Vertex_Arena sub_arena = vertex_arena_push(mapped_arena.arena, 4 * n, 6 * n);
		Vertex_Arena_Mapped mapped = vertex_arena_submap(mapped_arena, sub_arena);

		uint32_t block_dim = 32;
		uint32_t grid_dim = ceil_div(particles.length, block_dim);

		gen_particle_vertex_buffer_kernel<<<grid_dim, block_dim>>>(mapped_arena, particles);
		return sub_arena;
	}

	Vertex_Arena gen_quad_tree_vertex_buffer(Vertex_Arena_Mapped& mapped_arena, Arena& tmp_arena, slice<Quad_Tree_Node>& nodes) {
		uint32_t n = nodes.length;

		Vertex_Arena sub_arena = vertex_arena_push(mapped_arena.arena, 4 * n, 8 * n);
		Vertex_Arena_Mapped mapped = vertex_arena_submap(mapped_arena, sub_arena);

		uint32_t block_dim = 32;
		uint32_t grid_dim = ceil_div(nodes.length, block_dim);

		gen_quad_tree_buffer_kernel<<<grid_dim, block_dim>>> (mapped_arena, nodes);

		return sub_arena;
	}

	constexpr real G = 1e-1;

	CGPU void compute(slice<Quad_Tree_Node> nodes, real max_ratio, slice<Particle_AOS> particles, uint32_t node_id) {
		Particle_AOS& p = particles[node_id];

		vec3 position = p.position;
		vec3 result = {};

		for (uint32_t node_id = 0; node_id < nodes.length; node_id++) {
			Quad_Tree_Node node = nodes[node_id];

			vec2 size = node.aabb.max - node.aabb.min;
			vec3 center = node.center_of_mass;

			vec3 coarse = -G * node.mass / sq(center - position) * (position - center);

			real ratio = length(vec2(center.x - position.x, center.y - position.y) / size);
			if (ratio > max_ratio) {
				result += coarse;
				node_id += node.node_count;
			}
			else if (quad_node_is_leaf(node)) {
				for (uint32_t i = 0; i < node.particle_count; i++) {
					if (node.data + i == node_id) continue;
					Particle_AOS& particle = particles[node.data + i];
					vec3 center = particle.position;
					result += -G * particle.mass / (FLT_EPSILON + sq(center - position)) * (position - center);
				}
			}
		}

		p.acceleration = result;
	}

	__global__ void compute_acceleration_kernel(slice<Quad_Tree_Node> nodes, real max_ratio, slice<Particle_AOS> particles) {
		uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= particles.length) return;

		compute(nodes, max_ratio, particles, i);
	}

	__global__ void compute_position_kernel(slice<Particle_AOS> particles, slice<Particle_AOS> particles2, real dt) {
		uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= particles.length) return;

		Particle_AOS& p = particles[i];
		Particle_AOS& p_next = particles2[i];

		p_next.velocity = p.velocity + dt * p.acceleration;
		p_next.position = p.position + dt * p_next.velocity;
	}

	void advance_gravity(slice<Quad_Tree_Node> nodes, Arena& tmp_arena, slice<Particle_AOS> particles, slice<Particle_AOS> particles2, Cuda_Error& err, real dt) {
		const real max_ratio = 0.5;

		uint32_t n = particles.length;

		uint32_t block_dim = 32;
		uint32_t grid_dim = ceil_div(particles.length, block_dim);

		compute_acceleration_kernel << <grid_dim, block_dim >> > (nodes, max_ratio, particles);
		compute_position_kernel << <grid_dim, block_dim >> > (particles, particles2, dt);
	}

	__global__ void init_galaxy_kernel(slice<Particle_AOS> particles, slice<real> rngs) {
		uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= particles.length) return;

		uint32_t num_centers = 2;
		vec3 centers[2] = {
			vec3(-1,-1,0), vec3(1,1,0)
		};

		real r = rngs[i * 3 + 0]; 
		real theta = 2*PI*rngs[i * 3 + 1];

		vec3 pos = {};
		pos.x = r * cos(theta);
		pos.y = r * sin(theta);
		pos += centers[i % num_centers];

		particles[i] = {};
		particles[i].position = pos;
		particles[i].mass = 10 * rngs[i*3 + 2];
	}

	int launch_nbody_viewer() {
		uvec2 extent = { 4096,2048 };

		Window_Desc window_desc = {};
		window_desc.title = "N-Body (Lucas Gotz)";
		window_desc.width = extent.x;
		window_desc.height = extent.y;
		window_desc.validation = true;
		window_desc.vsync = false;

		Cuda_Error err = {};

		Arena perm_arena = make_host_arena(mb(500));
		DEFER(destroy_host_arena(perm_arena));

		Arena frame_arena = make_host_arena(gb(2));
		DEFER(destroy_host_arena(frame_arena));

		Arena frame_device_arena = make_device_arena(gb(1), err);
		DEFER(destroy_device_arena(frame_arena));

		Arena device_arena = make_device_arena(gb(1), err);
		DEFER(destroy_device_arena(device_arena));

		Window* window = make_window(perm_arena, window_desc);
		if (!window) return -1;
		DEFER(destroy_window(window));

		RHI& rhi = window_rhi(*window);


		Window_Input input = { window };

		GLuint shader = shader_make(rhi, vertex_shader_text, face_fragment_shader_text);
		DEFER(shader_destroy(rhi, shader));

		Vertex_Buffer vertex_buffer;
		Vertex_Arena vertex_particle_arena;
		Vertex_Arena vertex_quad_tree_arena;
		Vertex_Arena vertex_tree_arena;

		uint32_t n = 100; 
		slice<Particle_AOS> particles;
		slice<Particle_AOS> particles2;
		slice<Quad_Tree_Node> nodes;


		{
			particles = arena_push_array<Particle_AOS>(device_arena, n);
			particles2 = arena_push_array<Particle_AOS>(device_arena, n);
			nodes = arena_push_array<Quad_Tree_Node>(device_arena, 2*n);

			slice<real> rngs = arena_push_array<real>(device_arena, 3*n);

			curandGenerator_t gen;
			curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
			curandSetPseudoRandomGeneratorSeed(gen,  1234ULL);
			curandGenerateUniform(gen, rngs.data, 3*n);


			uint32_t block_dim = 32;
			uint32_t grid_dim = ceil_div(n, block_dim);
			init_galaxy_kernel << <grid_dim, block_dim >> > (particles, rngs);
			err |= cudaDeviceSynchronize();
		}

		AABB aabb = { vec2(-10.0_R), vec2(10.0_R) };

		Vertex_Buffer_Desc desc = {};
		desc.vertex_buffer_size = mb(500);
		desc.index_buffer_size = mb(500);

		vertex_buffer = make_vertex_buffer(rhi, desc);
		DEFER(destroy_vertex_buffer(rhi, vertex_buffer));

		cudaGraphicsResource_t vbo_resource;
		cudaGraphicsResource_t ibo_resource;

		err |= cudaGraphicsGLRegisterBuffer(&vbo_resource, vertex_buffer.vertices, cudaGraphicsMapFlagsWriteDiscard);
		err |= cudaGraphicsGLRegisterBuffer(&ibo_resource, vertex_buffer.indices, cudaGraphicsMapFlagsWriteDiscard);
		DEFER(cudaGraphicsUnregisterResource(vbo_resource));
		DEFER(cudaGraphicsUnregisterResource(ibo_resource));

		FPV fpv = {};
		fpv.view_pos.z = 5;
		fpv.pitch = 0;

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
			
				

			frame_arena.used = 0;
			frame_device_arena.used = 0;

			vertex_buffer.arena.vertex_count = 0;
			vertex_buffer.arena.index_count = 0;

			Map_Vertex_Buffer_Cuda_Desc desc = { vertex_buffer, vbo_resource, ibo_resource };

			Vertex_Arena_Mapped mapped = vertex_buffer_map_cuda(desc, err);
			DEFER(vertex_buffer_cuda_unmap(desc, err));

			real t0 = get_time();
			N_Body_Tree tree = build_quad_tree(frame_arena, aabb, particles, particles2, nodes, err);
			nodes.length = tree.nodes.length;
			std::swap(particles, particles2);
			printf("Build Tree %f, %i\n", 1000 * (get_time() - t0), nodes.length);

			if (playing) {
				const uint32_t sub_timesteps = 1;
				real sim_dt = 1e-4;

				for (uint32_t i = 0; i < sub_timesteps; i++) {
					real t = get_time();
					advance_gravity(nodes, frame_arena, particles, particles2, err, sim_dt);
					std::swap(particles, particles2);
					cudaDeviceSynchronize();
					printf("Advance gravity %f\n", 1000 * (get_time() - t));
				}
			}

			real t2 = get_time();
			vertex_particle_arena = gen_particle_vertex_buffer(mapped, frame_device_arena, particles);
			cudaDeviceSynchronize();
			printf("Gen Particle %f\n", 1000 * (get_time() - t2));
			vertex_quad_tree_arena = gen_quad_tree_vertex_buffer(mapped, frame_device_arena, nodes);

			{
				glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				glEnable(GL_DEPTH_TEST);
				glDisable(GL_CULL_FACE);
				glLineWidth(1.0);

				glUseProgram(shader);
				glUniformMatrix4fv(glGetUniformLocation(shader, "MVP"), 1, GL_TRUE, (const GLfloat*)&mvp);
				glUniform3fv(glGetUniformLocation(shader, "dir_light"), 1, &dir_light.x);

				Command_Buffer cmd_buffer = {};
				cmd_buffer_bind(cmd_buffer, vertex_buffer);
				cmd_buffer_draw(cmd_buffer, DRAW_LINE, vertex_quad_tree_arena);
				cmd_buffer_draw(cmd_buffer, DRAW_TRIANGLES, vertex_particle_arena);

				window_draw(*window);
			}
		}

		return 0;
	}
}