#include "reycode/graphics/viewer/colormap.h"
#include "reycode/graphics/rhi/window.h"
#include "reycode/graphics/rhi/rhi.h"
#include "reycode/graphics/viewer/fpv.h"
#include "reycode/reycode.h"
#include <cuda_gl_interop.h>
#include <curand.h>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>

#include <atomic>
#include <thread>

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

	constexpr uint32_t MAX_PARTICLES_PER_QUAD = 32; // 1024; // 2048;

	using quad_tree_node_handle = uint32_t;
	using quad_tree_data_handle = uint32_t;

	constexpr quad_tree_data_handle NODE_HAS_CHILDREN_FLAG = INT_MAX;

	using morton_code = uint64_t;
	constexpr uint32_t MORTON_BITS_TOTAL = 64;
	constexpr uint32_t MORTON_BITS_DEPTH = 4;
	constexpr uint32_t MORTON_BITS = MORTON_BITS_TOTAL - MORTON_BITS_DEPTH;

	constexpr uint32_t MAX_DEPTH = 16;
	static_assert(MAX_DEPTH <= 1 << MORTON_BITS_DEPTH, "MAX DEPTH cannot be encoded with MORTON BITS");

	struct Quad_Tree_Node {
		AABB aabb;
		vec3 center_of_mass;
		real mass;
		uint32_t particle_count;
		uint32_t node_count;
		quad_tree_data_handle data;
		morton_code morton;

		INL_CGPU operator morton_code() { return morton; }
	};

	CGPU bool quad_node_is_leaf(Quad_Tree_Node node) {
		return node.node_count == 0; 
	}

	struct N_Body_Tree {
		AABB aabb;
		slice<Quad_Tree_Node> nodes;
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

	//Source: https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits
	INL_CGPU morton_code xy2d_morton(uint64_t x, uint64_t y) {
		x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
		x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
		x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
		x = (x | (x << 2)) & 0x3333333333333333;
		x = (x | (x << 1)) & 0x5555555555555555;

		y = (y | (y << 16)) & 0x0000FFFF0000FFFF;
		y = (y | (y << 8)) & 0x00FF00FF00FF00FF;
		y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F;
		y = (y | (y << 2)) & 0x3333333333333333;
		y = (y | (y << 1)) & 0x5555555555555555;

		return x | (y << 1);
	}

	INL_CGPU uint32_t morton_1(morton_code x)
	{
		x = x & 0x5555555555555555;
		x = (x | (x >> 1)) & 0x3333333333333333;
		x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F;
		x = (x | (x >> 4)) & 0x00FF00FF00FF00FF;
		x = (x | (x >> 8)) & 0x0000FFFF0000FFFF;
		x = (x | (x >> 16)) & 0x00000000FFFFFFFF;
		return (uint32_t)x;
	}

	INL_CGPU void d2xy_morton(uint64_t d, uint64_t& x, uint64_t& y) {
		x = morton_1(d);
		y = morton_1(d >> 1);
	}

	INL_CGPU morton_code to_morton(vec2 pos, AABB aabb) {
		uint64_t x = (double)((pos.x - aabb.min.x) / (aabb.max.x - aabb.min.x)) * (1ull << MORTON_BITS_TOTAL/2);
		uint64_t y = (double)((pos.y - aabb.min.y) / (aabb.max.y - aabb.min.y)) * (1ull << MORTON_BITS_TOTAL/2);

		x << MORTON_BITS / 2;
		y << MORTON_BITS / 2;

		return xy2d_morton(x, y);
	}

	INL_CGPU vec2 from_morton(morton_code code, AABB aabb) {
		uint64_t x, y;
		d2xy_morton(code, x, y);

		x << MORTON_BITS / 2;
		y << MORTON_BITS / 2;

		return vec2(
			(double)(aabb.min.x + x * (aabb.max.x - aabb.min.x) / (1ull << MORTON_BITS_TOTAL/2)),
			(double)(aabb.min.y + y * (aabb.max.y - aabb.min.y) / (1ull << MORTON_BITS_TOTAL/2))
		);
	}

	struct vec4_highp {
		double x, y, z, w;

		CGPU vec4_highp() : x(0), y(0), z(0), w(0) {}
		CGPU vec4_highp(double x, double y, double z, double w) : x(x), y(y), z(z), w(w) {}
		CGPU vec4_highp(vec3 xyz, double w) : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {}

		CGPU vec3 xyz() { return { (real)x,(real)y,(real)z }; }
	};

	CGPU vec4_highp operator*(double a, vec4_highp b) {
		return { a * b.x,a * b.y,a * b.z,a * b.w };
	}

	CGPU vec4_highp operator+(vec4_highp a, vec4_highp b) {
		return { a.x + b.x,a.y + b.y,a.z + b.z,a.w + b.w };
	}

	CGPU vec4_highp operator-(vec4_highp a, vec4_highp b) {
		return { a.x - b.x,a.y - b.y,a.z - b.z,a.w - b.w };
	}

	N_Body_Tree build_quad_tree(Arena& tmp_arena, AABB aabb, slice<Particle_AOS> particles_gpu, slice<Quad_Tree_Node>& nodes_gpu, Cuda_Error& err) {
		uint32_t n = particles_gpu.length;
		slice<morton_code> morton_codes_particles = arena_push_array<morton_code>(tmp_arena, n);

		using counter = thrust::counting_iterator<uint32_t>;
		auto compute = thrust::device;
		bool is_host = false;

		slice<Particle_AOS> particles;
		slice<Quad_Tree_Node> nodes;

		if (is_host) {
			err |= cudaDeviceSynchronize();

			particles = arena_push_array<Particle_AOS>(tmp_arena, particles_gpu.length);
			err |= cudaMemcpy(particles.data, particles_gpu.data, (uint64_t)sizeof(Particle_AOS) * particles_gpu.length, cudaMemcpyDeviceToHost);

			nodes = arena_push_array<Quad_Tree_Node>(tmp_arena, 2*n);
		}
		else {
			particles = particles_gpu;
			nodes = nodes_gpu;
			nodes.length = 2 * n;
		}

		thrust::transform(compute, counter(0u), counter(n), morton_codes_particles.begin(),[=] CGPU(uint32_t i) {
			vec3 pos = particles[i].position;
			morton_code code = to_morton(pos.xy(), aabb);
			return code;
		});

		uint32_t node_id_offset = 0;

		slice<uint32_t> prefix_sum_morton_codes_low  = arena_push_array<uint32_t>(tmp_arena, n);
		slice<uint32_t> prefix_sum_morton_codes_high = arena_push_array<uint32_t>(tmp_arena, n);
		slice<uint32_t> prefix_sum_nodes             = arena_push_array<uint32_t>(tmp_arena, n+1);
		slice<uint32_t> node_child_counts            = arena_push_array<uint32_t>(tmp_arena, n);

		slice<vec4_highp> center_of_mass_prefix_sum = arena_push_array<vec4_highp>(tmp_arena, n);

		thrust::fill_n(compute, node_child_counts.begin(), n, 0);

		thrust::sort_by_key(compute, morton_codes_particles.begin(), morton_codes_particles.end(), particles.begin());

		for (int depth = MAX_DEPTH; depth > 0; depth--) {			
			morton_code mask_high = ((1ull << 2*(depth - 1)) - 1) << (MORTON_BITS_TOTAL - 2*(depth-1));
			morton_code mask_low = ((1ull << 2*(depth)) - 1) << (MORTON_BITS_TOTAL - 2*depth);

			auto mask_code = [](morton_code mask) {
				return[mask] CGPU(morton_code code) {
					return code & mask;
				};
			};

			auto code_low_begin = thrust::make_transform_iterator(morton_codes_particles.begin(), mask_code(mask_low));
			auto code_low_end = thrust::make_transform_iterator(morton_codes_particles.end(), mask_code(mask_low));

			auto code_high_begin = thrust::make_transform_iterator(morton_codes_particles.begin(), mask_code(mask_high));
			auto code_high_end   = thrust::make_transform_iterator(morton_codes_particles.end(), mask_code(mask_high));
			
			if (depth == MAX_DEPTH) {
				thrust::inclusive_scan_by_key(compute, code_low_begin, code_low_end, thrust::make_constant_iterator(1u), prefix_sum_morton_codes_low.begin());
			}
			else {
				std::swap(prefix_sum_morton_codes_low, prefix_sum_morton_codes_high);
			}
			thrust::inclusive_scan_by_key(compute, code_high_begin, code_high_end, thrust::make_constant_iterator(1u), prefix_sum_morton_codes_high.begin());

			auto count_nodes = [=] CGPU (uint32_t i) {
				if (i >= n) return 0u;
				if (!(i == n - 1 || prefix_sum_morton_codes_high[i + 1] == 1)) return 0u;

				uint32_t particle_count = prefix_sum_morton_codes_high[i];
				if (!(particle_count > MAX_PARTICLES_PER_QUAD)) return 0u;

				morton_code code_high = morton_codes_particles[i] & mask_high;

				uint32_t child_node_count = 0;

				int j = i;
				for (uint32_t k = 0; k < 4 && j >= 0 && (morton_codes_particles[j] & mask_high) == code_high; k++) {
					child_node_count++;
					j -= prefix_sum_morton_codes_low[j];
				}

				return child_node_count;
			};

			//thrust::copy_if(compute, thrust::make_transform_iterator(counter(0u), count_nodes), thrust::make_transform_iterator(counter(n + 1), count_nodes), prefix_sum_nodes.begin());

			thrust::exclusive_scan(compute, thrust::make_transform_iterator(counter(0u), count_nodes), thrust::make_transform_iterator(counter(n+1), count_nodes), prefix_sum_nodes.begin());

			uint32_t level_node_count = 0;
			cudaMemcpyAsync(&level_node_count, &prefix_sum_nodes[n], sizeof(uint32_t), is_host ? cudaMemcpyHostToHost : cudaMemcpyDeviceToHost);

			vec2 aabb_extent = (1.0_R/(1<<depth))*(aabb.max - aabb.min);

			thrust::for_each(compute, counter(0), counter(n), [=] CGPU (uint32_t i) mutable {
				uint32_t child_count = prefix_sum_nodes[i + 1] - prefix_sum_nodes[i];
				if (child_count == 0) return;

				uint32_t offset = node_id_offset + prefix_sum_nodes[i];				
				uint32_t total_child_node_count = 0;
				vec4_highp total_center_of_mass = {};

				int j = i;
				for (uint32_t k = 0; k < child_count; k++) {
					uint32_t child_count = node_child_counts[j];

					morton_code code = morton_codes_particles[j] & mask_low;

					Quad_Tree_Node& node = nodes[offset + k];
					node.aabb.min = from_morton(code, aabb);
					node.aabb.max = node.aabb.min + aabb_extent;
					node.node_count = child_count;
					node.particle_count = prefix_sum_morton_codes_low[j];
					node.data = j+1 - node.particle_count;
					node.morton = code | depth;

					vec4_highp center_of_mass;
					if (node.node_count == 0) {
						for (uint32_t i = 0; i < node.particle_count; i++) {
							const Particle_AOS& p = particles[node.data + i];
							center_of_mass = center_of_mass + vec4_highp(p.mass * p.position, p.mass);
						}

						center_of_mass_prefix_sum[j] = center_of_mass;
					}
					else {
						center_of_mass = center_of_mass_prefix_sum[j];
					}

					node.mass = center_of_mass.w;
					node.center_of_mass = ((1.0 / center_of_mass.w) * center_of_mass).xyz();

					j -= prefix_sum_morton_codes_low[j];
					total_child_node_count += 1 + child_count;
					total_center_of_mass = total_center_of_mass + center_of_mass;
				}

				node_child_counts[i] = total_child_node_count;
				center_of_mass_prefix_sum[i] = total_center_of_mass;
			});

			cudaDeviceSynchronize();
			node_id_offset += level_node_count;

			//std::swap(prefix_sum_morton_codes_low, prefix_sum_morton_codes_high);
		}	

		Quad_Tree_Node root = {};
		root.node_count = node_id_offset;
		root.particle_count = particles.length;
		root.aabb = aabb;
		root.morton = 0;

		cudaMemcpy(&nodes[node_id_offset], &root, sizeof(Quad_Tree_Node), is_host ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice);
		node_id_offset++;

		nodes.length = node_id_offset;
		thrust::sort(compute, nodes.begin(), nodes.end(), thrust::less<morton_code>());



		auto center_of_mass = [=] CGPU(uint32_t i) {
			Particle_AOS p = particles[i];
			return vec4_highp(p.mass * p.position, p.mass);
		};

		/*auto center_of_mass_begin = thrust::make_transform_iterator(counter(0), center_of_mass);
		auto center_of_mass_end   = thrust::make_transform_iterator(counter(n), center_of_mass);

		thrust::inclusive_scan(compute, center_of_mass_begin, center_of_mass_end, center_of_mass_prefix_sum.begin());

		thrust::for_each(compute, counter(0), counter(nodes.length), [=] CGPU(uint32_t i) mutable {
			Quad_Tree_Node& node = nodes[i];
			
			vec4_highp center = center_of_mass_prefix_sum[node.data+node.particle_count-1] - (node.data==0 ? vec4_highp() : center_of_mass_prefix_sum[node.data-1]);
			
			nodes[i].mass = center.w;
			nodes[i].center_of_mass = (1.0/center.w * center).xyz();
		});*/

		N_Body_Tree tree;
		tree.nodes = nodes;
		tree.aabb = aabb;

		if (is_host) {
			err |= cudaMemcpy(particles_gpu.data, particles_gpu.data, (uint64_t)sizeof(Particle_AOS) * particles_gpu.length, cudaMemcpyHostToDevice);
			err |= cudaMemcpy(nodes_gpu.data, tree.nodes.data, (uint64_t)sizeof(Quad_Tree_Node) * tree.nodes.length, cudaMemcpyHostToDevice);
		}



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

		real size = 5e-3 * particles[i].mass;

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

		gen_particle_vertex_buffer_kernel<<<grid_dim, block_dim>>>(mapped, particles);
		return sub_arena;
	}

	Vertex_Arena gen_quad_tree_vertex_buffer(Vertex_Arena_Mapped& mapped_arena, Arena& tmp_arena, slice<Quad_Tree_Node>& nodes) {
		uint32_t n = nodes.length;

		Vertex_Arena sub_arena = vertex_arena_push(mapped_arena.arena, 4 * n, 8 * n);
		Vertex_Arena_Mapped mapped = vertex_arena_submap(mapped_arena, sub_arena);

		uint32_t block_dim = 32;
		uint32_t grid_dim = ceil_div(nodes.length, block_dim);

		gen_quad_tree_buffer_kernel<<<grid_dim, block_dim>>> (mapped, nodes);

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

		p_next = p;
		p_next.velocity = p.velocity + dt * p.acceleration;
		p_next.position = p.position + dt * p_next.velocity;
	}

	void advance_gravity(slice<Quad_Tree_Node> nodes, Arena& tmp_arena, slice<Particle_AOS> particles, slice<Particle_AOS> particles2, Cuda_Error& err, real dt) {
		const real max_ratio = 2.0;

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
			vec3(-1,0,0), vec3(1,0,0)
		};

		real r = rngs[i * 3 + 0]; 
		real theta = 2*PI*rngs[i * 3 + 1];

		vec3 pos = {};
		pos.x = r * cos(theta);
		pos.y = r * sin(theta);
		pos += centers[i % num_centers];

		particles[i] = {};
		particles[i].position = pos;
		particles[i].mass = 0.2 * (rngs[i*3 + 2] + 0.1);
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

		uint32_t n = 1e6;
		slice<Particle_AOS> particles;
		slice<Particle_AOS> particles2;
		slice<Quad_Tree_Node> nodes;


		particles = arena_push_array<Particle_AOS>(device_arena, n);
		particles2 = arena_push_array<Particle_AOS>(device_arena, n);
		nodes = arena_push_array<Quad_Tree_Node>(device_arena, 2 * n);


		if(true) {
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
		else {
			Particle_AOS particles_cpu[4] = {};
			particles_cpu[0].position = vec3(0.5, 0.5, 0);
			particles_cpu[1].position = vec3(-0.5, 0.5, 0);
			particles_cpu[2].position = vec3(0.5, -0.5, 0);
			particles_cpu[3].position = vec3(-0.5, -0.5, 0);

			for (uint32_t i = 0; i < 4; i++) particles_cpu[i].mass = 100;
		
			cudaMemcpy(particles.data, particles_cpu, 4 * sizeof(Particle_AOS), cudaMemcpyHostToDevice);
			cudaMemcpy(particles2.data, particles_cpu, 4 * sizeof(Particle_AOS), cudaMemcpyHostToDevice);
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
			N_Body_Tree tree = build_quad_tree(frame_device_arena, aabb, particles,  nodes, err);
			nodes.length = tree.nodes.length;
			printf("Build Tree %f, %i\n", 1000 * (get_time() - t0), nodes.length);

			if (playing) {
				const uint32_t sub_timesteps = 1;
				real sim_dt = 1e-4;

				for (uint32_t i = 0; i < sub_timesteps; i++) {
					real t = get_time();
					advance_gravity(nodes, frame_arena, particles, particles2, err, sim_dt);
					
					//debug_view_device_values(particles, 100);
					//debug_view_device_values(particles2, 100);
					
					std::swap(particles, particles2);
					//cudaDeviceSynchronize();
					printf("Advance gravity %f\n", 1000 * (get_time() - t));
				}
			}

			real t2 = get_time();
			
			//vertex_quad_tree_arena = gen_quad_tree_vertex_buffer(mapped, frame_device_arena, nodes);
			//cudaDeviceSynchronize();
			
			vertex_particle_arena = gen_particle_vertex_buffer(mapped, frame_device_arena, particles);
			cudaDeviceSynchronize();
			printf("Gen Particle %f\n", 1000 * (get_time() - t2));
			

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