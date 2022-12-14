#pragma once

#include "reycode/reycode.h"
#include <glad/glad.h>
#include <cuda_runtime.h>

namespace reycode {
    struct RHI;

    struct Vertex {
        vec3 pos;
        vec3 normal;
        vec3 color;
    };

    struct Vertex_Buffer_Desc {
        size_t vertex_buffer_size = 0;
        size_t index_buffer_size = 0;
    };

    struct Vertex_Arena {
        uint32_t vertex_offset = 0;
        uint32_t vertex_count = 0;
        uint32_t vertex_capacity = 0;

        uint32_t index_offset = 0;
        uint32_t index_count = 0;
        uint32_t index_capacity = 0;
    };

    struct Vertex_Buffer {
        GLuint vao = 0;
        GLuint vertices = 0;
        GLuint indices = 0;

        Vertex_Arena arena;

        INLINE operator Vertex_Arena() { return arena; }
    };

    struct Vertex_Arena_Mapped {
        Vertex_Arena& arena;
        slice<Vertex> vertices;
        slice<uint32_t> indices;
    };

    struct Map_Vertex_Buffer_Cuda_Desc {
        Vertex_Buffer& vbuffer;
        cudaGraphicsResource_t vbo_resource;
        cudaGraphicsResource_t ibo_resource;
    };

    Vertex_Buffer make_vertex_buffer(RHI& rhi, const Vertex_Buffer_Desc& desc);
    void vertex_buffer_upload(RHI& rhi, Vertex_Buffer& buffer, slice<Vertex> vertices, slice<uint32_t> indices);
    void destroy_vertex_buffer(RHI& rhi, Vertex_Buffer& buffer);

    Vertex_Arena_Mapped vertex_arena_submap(Vertex_Arena_Mapped& master, Vertex_Arena& child);
    Vertex_Arena vertex_arena_push(Vertex_Arena& arena, uint32_t vertices, uint32_t indices);

    void vertex_buffer_cuda_unmap(Map_Vertex_Buffer_Cuda_Desc& desc, Cuda_Error err);
    Vertex_Arena_Mapped vertex_buffer_map_cuda(Map_Vertex_Buffer_Cuda_Desc& desc, Cuda_Error err);
}