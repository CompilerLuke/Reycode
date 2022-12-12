#include "reycode/reycode.h"
#include "reycode/mesh/mesh.h"

namespace reycode {
    __global__ void make_coarse_grid_kernel(slice<vec3> particles, uvec3 dims, uvec3 stride, vec3 tile_size) {
        uvec3 tid = threadIdx;
        uvec3 gid = uvec3(blockIdx) * uvec3(blockDim) + tid;

        if (!(gid.x < dims.x && gid.y < dims.y && gid.z < dims.z)) return;

        particles[gid.x * stride.x + gid.y * stride.y + gid.z * stride.z] = (vec3(gid) - 0.5_R * vec3(dims) + vec3(0.5_R)) * tile_size;
    }

    Mesh_Level make_coarse_grid(cudaStream_t stream, Arena& arena, uvec3 dims, vec3 tile_size) {
        uvec3 block_dim = uvec3(TILE_WIDTH_INNER, TILE_WIDTH_INNER, TILE_WIDTH_INNER);
        uvec3 grid_dim = ceil_div(dims, block_dim);
        uvec3 stride = uvec3(1, dims.x, dims.x * dims.y);

        Mesh_Level level = {};
        level.tile_size = tile_size;
        level.tile_count = dims.x * dims.y * dims.z;
        level.particles = arena_push_array<vec3>(arena, level.tile_count);

        make_coarse_grid_kernel << <grid_dim, block_dim, 0, stream>> > (level.particles, dims, stride, tile_size);

        return level;
    }

}