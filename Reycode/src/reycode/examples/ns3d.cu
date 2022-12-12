#include "reycode/reycode.h"
#include "reycode/graphics/rhi/window.h"
#include "reycode/graphics/viewer/fpv.h"
#include "reycode/mesh/mesh.h"
#include "reycode/graphics/rhi/rhi.h"
#include "reycode/graphics/viewer/cross_section.h"

#define GLAD_GL_IMPLEMENTATION
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include <stdio.h>
#include <stdint.h>

namespace reycode {
    __global__ void fixed_value_bc_values_kernel(Mesh_Level level, real* data) {

    }

    __global__ void init_scalar_field(Mesh_Level level, slice<real> data) {
        uint32_t tile_id = blockIdx.x;
        uvec3 cell = uvec3(threadIdx);

        if (!(tile_id < level.tile_count && max(cell) < TILE_INNER_CELLS)) return;

        vec3 pos = level.particles[tile_id];
        real value = cos(pos.x)* cos(pos.y)* cos(pos.z);
        level(data, tile_id, cell) = value;
    }
    
    void solve_laplace_eq(cudaStream_t stream, Adaptive_Mesh& mesh) {
        Mesh_Level level = mesh.levels[0];


    }

    int launch_ns3d_viewer() {
        RHI* rhi = nullptr;

        uint32_t frame_width = 4096;
        uint32_t frame_height = 2160;

        Window_Desc window_desc = {};
        window_desc.width = frame_width;
        window_desc.height = frame_height;
        window_desc.title = "Reycode";
        window_desc.vsync = false;
        window_desc.validation = true;

        Cuda_Error err;

        Arena host_perm_arena = make_host_arena(mb(10));
        DEFER(destroy_host_arena(host_perm_arena));

        Arena device_perm_arena = make_device_arena(mb(100), err);
        DEFER(destroy_device_arena(device_perm_arena));

        Arena device_frame_arena = make_device_arena(mb(100), err);
        DEFER(destroy_device_arena(device_frame_arena));
        //vertex_buffer_upload(vbuffer, 4, 6, vertices, indices);

        Window* window = make_window(host_perm_arena, window_desc);
        DEFER(destroy_window(window));

        Window_Input input = { window };
        
        FPV fpv = {};
        fpv.view_pos.z = 5;

        if (!window) {
            return 1;
        }

        cudaStream_t stream = nullptr;



        uvec3 dims = uvec3(20, 20, 20);
        vec3 tile_size = vec3(5.0) / vec3(dims);

        uint32_t mesh_levels = 1;
        Adaptive_Mesh mesh = {};
        mesh.levels = arena_push_array<Mesh_Level>(host_perm_arena, mesh_levels);

        Mesh_Level& level = mesh.levels[0];
        level = make_coarse_grid(stream, device_perm_arena, dims, tile_size);
       
        slice<real> pressure = arena_push_array<real>(device_perm_arena, mesh_cell_count(level));

        {
            uvec3 grid_size = {level.tile_count, 1, 1};
            uvec3 block_size = uvec3(TILE_WIDTH_INNER, TILE_WIDTH_INNER, TILE_WIDTH_INNER);

            init_scalar_field<<<grid_size, block_size, 0, stream>>>(level, pressure);
        }

        err |= cudaDeviceSynchronize();

        Cross_Section_Renderer* renderer = cross_section_make(host_perm_arena, *rhi);
        DEFER(cross_section_destroy(renderer));

        real prev_t = glfwGetTime();
        
        real slice_move_speed = 1.0;
        vec3 cut_pos = vec3(0);

        while (window_is_open(*window)) {
            arena_reset(device_frame_arena, 0);

            real t = glfwGetTime();
            real dt = t - prev_t;

            window_input_poll(input);
            fpv_update(fpv, input.state, dt);


            if (input_key_down(input.state, KEY_ARROW_UP)) cut_pos += vec3(0, 1, 0) * slice_move_speed * dt;
            if (input_key_down(input.state, KEY_ARROW_DOWN)) cut_pos += vec3(0, -1, 0) * slice_move_speed * dt;

            vec3 fdir = fpv.forward_dir;
            vec3 side_dir;
            if (fabs(fdir.z) > fabs(fdir.x)) side_dir = (fdir.z < 0 ? 1 : -1) * vec3(1, 0, 0);
            else side_dir = (fdir.x > 0 ? 1 : -1) * vec3(0, 0, 1);

            if (input_key_down(input.state, KEY_ARROW_RIGHT)) cut_pos += side_dir * slice_move_speed * dt;
            if (input_key_down(input.state, KEY_ARROW_LEFT)) cut_pos -= side_dir * slice_move_speed * dt;
            

            window_input_capture_cursor(input, fpv.capture_cursor);
        
            mat4x4 projection = fpv_proj_mat(fpv, {frame_width, frame_height});

            mat4x4 view = fpv_view_mat(fpv);
            mat4x4 mvp = projection * view;

            vec3 dir_light = normalize(vec3(-1, -1, -1));
            
            cross_section_update(*renderer, device_frame_arena, pressure, cut_pos, mesh, stream);
            
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glEnable(GL_DEPTH_TEST);
            
            cross_section_render(*renderer, dir_light, mvp);

            window_draw(*window);
            
            prev_t = t;
        }

        return -1;
    }
}
