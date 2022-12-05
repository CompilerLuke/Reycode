#include "reycode/reycode.h"
#include "reycode/mesh/mesh.h"
#include "reycode/graphics/rhi/rhi.h"
#include "reycode/graphics/viewer/cross_section.h"
#include "reycode/graphics/viewer/colormap.h"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_scan.h>
#include <thrust/scan.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace reycode {
    struct Cull_Mesh_In {
        Arena& tmp_device_arena;
        Mesh_Level& level;
        cudaStream_t stream;
        vec3 cut_center;
    };

    struct Cull_Mesh_Out {
        uint32_t visible_tile_count;
        slice<uint32_t> ids;
    };

    void cull_grid(const Cull_Mesh_In& in, Cull_Mesh_Out& out) {
        Arena& tmp_device_arena = in.tmp_device_arena;
        size_t arena_watermark = tmp_device_arena.used;
        DEFER(arena_reset(tmp_device_arena, arena_watermark));

        Mesh_Level& level = in.level;
        slice<vec3> positions = in.level.positions;
        vec3 cut_center = in.cut_center;
        vec3 tile_size = level.tile_size;

        slice<uint32_t> offsets = arena_push_array<uint32_t>(tmp_device_arena, level.tile_count + 1);

        auto cull_kernel = [=] CGPU(uint32_t gid) {
            if (gid >= level.tile_count) return false;

            vec3 pos = positions[gid];
            vec3 cut_pos = pos - cut_center;
            bool visible = !(cut_pos.x > 0 && cut_pos.y > 0 && cut_pos.z > 0)
                && !(cut_pos.x < -tile_size.x || cut_pos.y < -tile_size.y || cut_pos.z < -tile_size.z);
            return visible;
        };

        auto compact_kernel = [=] CGPU(uint32_t gid) mutable {
            if (offsets[gid] != offsets[gid + 1]) out.ids[offsets[gid]] = gid;
        };

        thrust::counting_iterator<uint32_t> first(0);
        thrust::counting_iterator<uint32_t> last = first + level.tile_count;

        thrust::transform(thrust::device, first, first + level.tile_count + 1, offsets.begin(), cull_kernel); // we want the total sum as well
        thrust::exclusive_scan(thrust::device, offsets.begin(), offsets.begin() + level.tile_count + 1, offsets.begin());
        cudaDeviceSynchronize();
        cudaMemcpy(&out.visible_tile_count, offsets.begin() + level.tile_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        //out.visible_tile_count = 60;

        thrust::for_each(thrust::device, first, first + level.tile_count, compact_kernel);
    }

    uint32_t quad_indices[6] = { 0,1,2,0,2,3 };
    Vertex quad_vertices[4] = { vec3(0, 0, 1), vec3(1, 0, 1), vec3(1,1,1), vec3(0,1,1) };

    constexpr uint32_t VERTS_PER_CUBE = 6 * (TILE_WIDTH_INNER + 1) * (TILE_WIDTH_INNER + 1);
    constexpr uint32_t INDICES_PER_CUBE = 6 * 3 * 2 * TILE_WIDTH_INNER * TILE_WIDTH_INNER;

    constexpr uint32_t EDGE_VERTS_PER_CUBE = 8;// (TILE_WIDTH + 1)* (TILE_WIDTH + 1);
    constexpr uint32_t EDGE_INDICES_PER_CUBE = 24;// 3 * 2 * ((TILE_WIDTH + 1) * (TILE_WIDTH + 1) / 4);

    /*
    __global__ void gen_cross_section_mesh(
        slice<Vertex> face_vertices, slice<uint32_t> face_indices,
        slice<Vertex> edge_vertices, slice<uint32_t> edge_indices,
        slice<real> data,
        slice<uint32_t> tile_ids,
        Mesh_Level level,
        vec3 cut_center,
        Colormap cm
    ) {
        Vertex cube_edge_vertices[8] = {
            {{-0.5,-0.5,0.5}},
            {{0.5,-0.5,0.5}},
            {{0.5,-0.5,-0.5}},
            {{-0.5,-0.5,-0.5}},

            {{-0.5,0.5,0.5}},
            {{0.5,0.5,0.5}},
            {{0.5,0.5,-0.5}},
            {{-0.5,0.5,-0.5}}
        };

        uint32_t cube_edge_indices[24] = {
            0,1,
            1,2,
            2,3,
            0,3,

            0,4,
            1,5,
            2,6,
            3,7,

            4,5,
            5,6,
            6,7,
            7,4
        };

        for (int i = 0; i < 24; i++) edge_indices[gid * EDGE_INDICES_PER_CUBE + i] = gid * EDGE_VERTS_PER_CUBE + cube_edge_indices[i];
        for (int i = 0; i < 8; i++) {
            Vertex vert = cube_edge_vertices[i];
            uint32_t vert_id = gid * EDGE_VERTS_PER_CUBE + i;
            edge_vertices[vert_id] = vert;
            edge_vertices[vert_id].pos = pos + vert.pos * level.tile_size * vec3(1 + 1e-3);
        }
    }
    */

    __global__ void gen_cross_section_mesh(
        slice<Vertex> face_vertices, slice<uint32_t> face_indices,
        slice<Vertex> edge_vertices, slice<uint32_t> edge_indices, 
        slice<real> data,
        slice<uint32_t> tile_ids, 
        Mesh_Level level, 
        vec3 cut_center,
        Colormap cm) 
    {
        uint32_t tid = threadIdx.x;
        uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
        if (gid >= level.tile_count) return;

        uint32_t tile_id = tile_ids[gid];

        vec3 pos = level.positions[tile_id];
        vec3 cut_pos = pos - cut_center;

        uint32_t vert_gid = gid * VERTS_PER_CUBE;
        uint32_t index_gid = gid * INDICES_PER_CUBE;
        
        Vertex cube_edge_vertices[8] = {
            {{-0.5,-0.5,0.5}},
            {{0.5,-0.5,0.5}},
            {{0.5,-0.5,-0.5}},
            {{-0.5,-0.5,-0.5}},

            {{-0.5,0.5,0.5}},
            {{0.5,0.5,0.5}},
            {{0.5,0.5,-0.5}},
            {{-0.5,0.5,-0.5}}
        };

        /*
        for (int axis = 0; axis < 3; axis++) {
            for (int dir = -1; dir <= 1; dir += 2) {
                ivec3 normal = {};
                normal[axis] = dir;

                ivec3 tangent;
                tangent[(axis + 1) % 3] = dir;

                ivec3 bitangent;
                bitangent[(axis + 2) % 3] = dir;

                for (int i = 0; i < TILE_WIDTH_INNER; i++) {
                    for (int j = 0; j < TILE_WIDTH_INNER; j++) {
                        auto get_index = [=](int i, int j) {
                            return vert_gid + i * (TILE_WIDTH_INNER + 1) + j;
                        };

                        face_indices[index_gid++] = get_index(i, j);
                        face_indices[index_gid++] = get_index(i, j + 1);
                        face_indices[index_gid++] = get_index(i + 1, j + 1);

                        face_indices[index_gid++] = get_index(i, j);
                        face_indices[index_gid++] = get_index(i + 1, j + 1);
                        face_indices[index_gid++] = get_index(i + 1, j);
                    }
                }

                for (int i = 0; i < TILE_WIDTH_INNER+1; i++) {
                    for (int j = 0; j < TILE_WIDTH_INNER + 1; j++) {
                        real u = 2 * (real(i) / TILE_WIDTH_INNER) - 1.0;
                        real v = 2 * (real(j) / TILE_WIDTH_INNER) - 1.0;

                        //vec3 offset = 0.5 * (normal + bitangent + u * tangent * v);

                        //vec3 tile_pos_f = vec3(TILE_HALO_DEPTH) + TILE_INNER_CELLS * (offset + vec3(0.5));
                        //uvec3 tile_pos = { uint32_t(tile_pos_f.x), uint32_t(tile_pos_f.y), uint32_t(tile_pos_f.z) };

                        uint32_t ci = (ci + ci + 1) / 2;

                        uint32_t W = TILE_WIDTH_INNER / 2;
                        uvec3 cell = {}; // uvec3((ivec3(W) + W * normal) + (i - W) * bitangent + (j - W) * tangent);

                        Vertex vert = {};
                        vert.pos = pos + 0.5 * (vec3(normal) + vec3(bitangent) * u + vec3(tangent) * v) * level.tile_size;
                        vert.normal = vec3(normal);
                        vert.color = color_map(cm, level(data, tile_id, cell), -1, 1);
                        //cosf(pos.x)* cosf(pos.y)* cosf(pos.z), -1, 1);
                            //color_map(cm, data[get_cell(level, tile_id, tile_pos)]);
                        // 0.5 * vert.pos + vec3(0.5);

                        face_vertices[vert_gid++] = vert;
                    }
                }
            }
        }
        */


        uint32_t cube_edge_indices[24] = {
            0,1,
            1,2,
            2,3,
            0,3,

            0,4,
            1,5,
            2,6,
            3,7,

            4,5,
            5,6,
            6,7,
            7,4
        };

        for (int i = 0; i < 24; i++) edge_indices[gid * EDGE_INDICES_PER_CUBE + i] = gid * EDGE_VERTS_PER_CUBE + cube_edge_indices[i];
        for (int i = 0; i < 8; i++) {
            Vertex vert = cube_edge_vertices[i];
            uint32_t vert_id = gid * EDGE_VERTS_PER_CUBE + i;
            edge_vertices[vert_id] = vert;
            edge_vertices[vert_id].pos = pos + vert.pos * level.tile_size * vec3(1 + 1e-3);
        }
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

    static constexpr const char* edge_fragment_shader_text =
        "#version 330\n"
        "out vec4 fragment;\n"
        "in vec3 f_normal;\n"
        "uniform vec3 dir_light;\n"
        "void main()\n"
        "{\n"
        "   fragment = vec4(vec3(0), 1.0);\n"
        "}\n";

    struct Cross_Section_Renderer {
        RHI& rhi;
        GLuint face_shader;
        GLuint edge_shader;
        
        Vertex_Buffer vbuffer;
        Vertex_Arena faces;
        Vertex_Arena edges;

        cudaGraphicsResource_t ibo_resource;
        cudaGraphicsResource_t vbo_resource;

        uint32_t visible_tiles;

        Colormap colormap;
    };

    Cross_Section_Renderer* cross_section_make(Arena& arena, RHI& rhi) {
        Cross_Section_Renderer* renderer = arena_push<Cross_Section_Renderer>(arena, {rhi});
        renderer->face_shader = shader_make(rhi, vertex_shader_text, face_fragment_shader_text);
        renderer->edge_shader = shader_make(rhi, vertex_shader_text, edge_fragment_shader_text);

        {
            Vertex_Buffer_Desc vbuffer_desc = {};
            vbuffer_desc.index_buffer_size = mb(1);//gb(1);
            vbuffer_desc.vertex_buffer_size = mb(1);//gb(1);
            renderer->vbuffer = make_vertex_buffer(rhi, vbuffer_desc);
            
            Vertex_Arena& varena = renderer->vbuffer.arena;
            renderer->faces = vertex_arena_push(varena, varena.vertex_count / 2, varena.index_count / 2);
            renderer->edges = vertex_arena_push(varena, varena.vertex_count / 2, varena.index_count / 2);
        }

        {
            Cuda_Error err;
            err |= cudaGraphicsGLRegisterBuffer(&renderer->vbo_resource, renderer->vbuffer.vertices, cudaGraphicsMapFlagsWriteDiscard);
            err |= cudaGraphicsGLRegisterBuffer(&renderer->ibo_resource, renderer->vbuffer.indices, cudaGraphicsMapFlagsWriteDiscard);
        }

        renderer->colormap = viridis_cm();

        return renderer;
    }

    void cross_section_destroy(Cross_Section_Renderer* renderer) {
        RHI& rhi = renderer->rhi;
        shader_destroy(rhi, renderer->face_shader);
        shader_destroy(rhi, renderer->edge_shader);
        destroy_vertex_buffer(rhi, renderer->vbuffer);
        cudaGraphicsUnregisterResource(renderer->vbo_resource);
        cudaGraphicsUnregisterResource(renderer->ibo_resource);
    }

    __global__ void test_shader(Vertex* vertices, uint32_t* indices) {
        indices[0] = 0;
        indices[1] = 1;
        indices[2] = 2;
        vertices[0].pos = vec3(0, 0, 0);
        vertices[1].pos = vec3(1, 0, 0);
        vertices[2].pos = vec3(1, 1, 0);
    }

    void cross_section_update(Cross_Section_Renderer& renderer, Arena& tmp_gpu_arena, slice<real> data, vec3 cut_pos, Adaptive_Mesh& mesh, cudaStream_t stream) {
        Cuda_Error err;

        Colormap& colormap = renderer.colormap;
        Mesh_Level& level = mesh.levels[0];

        Map_Vertex_Buffer_Cuda_Desc map_desc = { renderer.vbuffer, renderer.vbo_resource, renderer.ibo_resource};
        Vertex_Arena_Mapped mapped = vertex_buffer_map_cuda(map_desc, err);
        DEFER(vertex_buffer_cuda_unmap(map_desc, err));

        Vertex_Arena_Mapped faces = vertex_arena_submap(mapped, renderer.faces);
        Vertex_Arena_Mapped edges = vertex_arena_submap(mapped, renderer.edges);
        
        slice<uint32_t> block_offsets = arena_push_array<uint32_t>(tmp_gpu_arena, level.tile_count + 1);
        slice<uint32_t> tile_ids = arena_push_array<uint32_t>(tmp_gpu_arena, level.tile_count);

        uint32_t& visible_tiles = renderer.visible_tiles;
        
        Cull_Mesh_In cull_in = {tmp_gpu_arena, level};
        cull_in.stream = stream;
        cull_in.cut_center = cut_pos;

        Cull_Mesh_Out cull_out = {};
        cull_out.ids = tile_ids;
        
        cull_grid(cull_in, cull_out);
        renderer.visible_tiles = cull_out.visible_tile_count;

        cudaStreamSynchronize(stream);

        uint32_t ids[64] = {};
        cudaMemcpy(ids, tile_ids.data, sizeof(ids), cudaMemcpyDeviceToHost);

        gen_cross_section_mesh<<<ceil_div(visible_tiles, 32), 32, 0, stream >>>(faces.vertices, faces.indices, edges.vertices, edges.indices, data, tile_ids, level, cut_pos, colormap);

        Vertex debug_vertices[64] = {};
        uint32_t debug_indices[64] = {};
        cudaMemcpy(debug_vertices, faces.vertices.data, sizeof(debug_vertices), cudaMemcpyDeviceToHost);
        cudaMemcpy(debug_indices, faces.indices.data, sizeof(debug_indices), cudaMemcpyDeviceToHost);

        cudaStreamSynchronize(stream);
    }

    void cross_section_render(Cross_Section_Renderer& renderer, vec3 dir_light, mat4x4& mvp) {
        uint32_t visible_tiles = renderer.visible_tiles;
        const uint32_t line_width = 5;

        Command_Buffer cmd_buffer = {};
        
        glBindVertexArray(renderer.vbuffer.vao);
        
        GLuint face_shader = renderer.face_shader;
        glUseProgram(face_shader);
        glUniformMatrix4fv(glGetUniformLocation(face_shader, "MVP"), 1, GL_TRUE, (const GLfloat*)&mvp);
        glUniform3fv(glGetUniformLocation(face_shader, "dir_light"), 1, &dir_light.x);

        renderer.faces.vertex_count = visible_tiles * VERTS_PER_CUBE;
        renderer.faces.index_count = visible_tiles * INDICES_PER_CUBE;
        cmd_buffer_draw(cmd_buffer, DRAW_TRIANGLES, renderer.faces);

        GLuint edge_shader = renderer.edge_shader;
        glUseProgram(edge_shader);
        glLineWidth(line_width);
        glUniformMatrix4fv(glGetUniformLocation(edge_shader, "MVP"), 1, GL_TRUE, (const GLfloat*)&mvp);
        glUniform3fv(glGetUniformLocation(edge_shader, "dir_light"), 1, &dir_light.x);
        
        renderer.edges.vertex_count = visible_tiles * EDGE_VERTS_PER_CUBE;
        renderer.edges.index_count = visible_tiles * EDGE_INDICES_PER_CUBE;
        cmd_buffer_draw(cmd_buffer, DRAW_LINE, renderer.edges);
    }
}