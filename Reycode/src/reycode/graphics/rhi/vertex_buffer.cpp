#pragma once

#include <glad/glad.h>
#include "reycode/graphics/rhi/vertex_buffer.h"
#include <cuda_runtime.h>

namespace reycode {
    Vertex_Buffer make_vertex_buffer(RHI& rhi, const Vertex_Buffer_Desc& desc) {
        Vertex_Buffer buffer = {};
        glGenVertexArrays(1, &buffer.vao);
        glBindVertexArray(buffer.vao);

        glGenBuffers(1, &buffer.indices);
        glGenBuffers(1, &buffer.vertices);

        glBindBuffer(GL_ARRAY_BUFFER, buffer.vertices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer.indices);

        const int VPOS_LOC = 0;
        const int VNORMAL_LOC = 1;
        const int VCOLOR_LOC = 2;

        glEnableVertexAttribArray(VPOS_LOC);
        glVertexAttribPointer(VPOS_LOC, 3, GL_FLOAT, GL_FALSE,
            sizeof(Vertex), (void*)offsetof(Vertex, pos));
        glEnableVertexAttribArray(VNORMAL_LOC);
        glVertexAttribPointer(VNORMAL_LOC, 3, GL_FLOAT, GL_FALSE,
            sizeof(Vertex), (void*)offsetof(Vertex, normal));
        glEnableVertexAttribArray(VCOLOR_LOC);
        glVertexAttribPointer(VCOLOR_LOC, 3, GL_FLOAT, GL_FALSE,
            sizeof(Vertex), (void*)offsetof(Vertex, color));

        glBufferData(GL_ELEMENT_ARRAY_BUFFER, desc.index_buffer_size, 0, GL_STATIC_DRAW);
        glBufferData(GL_ARRAY_BUFFER, desc.vertex_buffer_size, 0, GL_STATIC_DRAW);

        glBindVertexArray(0);

        buffer.arena.vertex_capacity = (uint32_t)(desc.vertex_buffer_size / sizeof(Vertex));
        buffer.arena.index_capacity = (uint32_t)(desc.index_buffer_size / sizeof(uint32_t));

        return buffer;
    }

    void vertex_buffer_upload(RHI& rhi, Vertex_Buffer& buffer, slice<Vertex> vertices, slice<uint32_t> indices) {
        glBindVertexArray(buffer.vao);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * indices.length, indices.data, GL_STATIC_DRAW);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.length, vertices.data, GL_STATIC_DRAW);
    }

    void destroy_vertex_buffer(RHI& rhi, Vertex_Buffer& buffer) {
        glDeleteBuffers(1, &buffer.indices);
        glDeleteBuffers(1, &buffer.vertices);
        glDeleteVertexArrays(1, &buffer.vao);
    }

    Vertex_Arena vertex_arena_push(Vertex_Arena& arena, uint32_t vertices, uint32_t indices) {
        Vertex_Arena sub = {};
        sub.vertex_offset = arena.vertex_offset + arena.vertex_count;
        sub.vertex_count = vertices;
        sub.vertex_capacity = vertices;
        sub.index_offset = arena.index_offset + arena.index_count;
        sub.index_count = indices;
        sub.index_capacity = indices;

        arena.vertex_count += vertices;
        arena.index_count += indices;

        assert(arena.vertex_count < arena.vertex_capacity);
        assert(arena.index_count < arena.index_capacity);

        return sub;
    }

    Vertex_Arena_Mapped vertex_arena_submap(Vertex_Arena_Mapped& master, Vertex_Arena& child) {
        uint32_t vertex_offset = child.vertex_offset - master.arena.vertex_offset;
        uint32_t index_offset = child.index_offset - master.arena.index_offset;

        Vertex_Arena_Mapped result = { child };
        result.vertices = subslice(master.vertices, vertex_offset, child.vertex_count);
        result.indices = subslice(master.indices, index_offset, child.index_count);
        return result;
    }

    Vertex_Arena_Mapped vertex_buffer_map_cuda(Map_Vertex_Buffer_Cuda_Desc& desc, Cuda_Error err) {
        Vertex_Arena_Mapped mapped = {desc.vbuffer.arena};
        cudaGraphicsResource_t resources[2] = { desc.ibo_resource, desc.vbo_resource };
        err |= cudaGraphicsMapResources(2, resources, 0);

        size_t vertices_bytes, indices_bytes;
        err |= cudaGraphicsResourceGetMappedPointer((void**)(&mapped.vertices.data), &vertices_bytes, desc.vbo_resource);
        err |= cudaGraphicsResourceGetMappedPointer((void**)(&mapped.indices.data), &indices_bytes, desc.ibo_resource);

        mapped.vertices.length = (uint32_t)(vertices_bytes / sizeof(Vertex));
        mapped.indices.length = (uint32_t)(indices_bytes / sizeof(uint32_t));
    
        return mapped;
    }

    void vertex_buffer_cuda_unmap(Map_Vertex_Buffer_Cuda_Desc& desc, Cuda_Error err) {
        cudaGraphicsResource_t resources[2] = { desc.ibo_resource, desc.vbo_resource };
        err |= cudaGraphicsUnmapResources(2, resources, 0);
    }
}