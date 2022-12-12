#pragma once

#include <reycode/reycode.h>
#include <driver_types.h>

namespace reycode {
    constexpr uint32_t TILE_HALO_DEPTH = 1;
    constexpr uint32_t TILE_WIDTH_INNER = 8;
    constexpr uint32_t TILE_WIDTH_OUTER = TILE_WIDTH_INNER + 2*TILE_HALO_DEPTH;
    constexpr uint32_t TILE_INNER_CELLS = TILE_WIDTH_INNER * TILE_WIDTH_INNER * TILE_WIDTH_INNER;
    constexpr uint32_t TILE_OUTER_CELLS = 6 * TILE_HALO_DEPTH * TILE_WIDTH_INNER * TILE_WIDTH_INNER;
    constexpr uint32_t TILE_CELLS = TILE_INNER_CELLS + TILE_OUTER_CELLS;
    constexpr real TILE_INNER_RATIO = real(TILE_INNER_CELLS) / TILE_OUTER_CELLS;

    struct Mesh_Level {
        vec3 tile_size;
        slice<vec3> particles;
        uint32_t tile_count;

        template<class T>
        INL_CGPU T& operator()(slice<T> data, uint32_t tile_id, uvec3 tile_pos) {
            uint32_t idx = tile_id* TILE_CELLS + tile_pos.x + tile_pos.y * TILE_WIDTH_INNER + tile_pos.z * TILE_WIDTH_INNER * TILE_WIDTH_INNER;
            return data[idx];
        }
    };

    struct Adaptive_Mesh {
        slice<Mesh_Level> levels;
    };

    INL_CGPU uint32_t mesh_cell_count(Mesh_Level level) {
        return level.tile_count * TILE_CELLS;
    }

	Mesh_Level make_coarse_grid(cudaStream_t stream, Arena& arena, uvec3 dims, vec3 tile_size);
}