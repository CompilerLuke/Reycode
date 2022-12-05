#include <glad/glad.h>
#include "reycode/graphics/rhi/shader.h"
#include <stdio.h>

namespace reycode {
    GLuint shader_make(RHI& rhi, const char* vertex_shader_text, const char* fragment_shader_text) {
        const GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);
        glCompileShader(vertex_shader);

        int  success;
        char infoLog[512];
        glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);

        if (!success) {
            glGetShaderInfoLog(vertex_shader, 512, NULL, infoLog);
            printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s", infoLog);
            return 0;
        }

        const GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);
        glCompileShader(fragment_shader);

        glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);

        if (!success) {
            glGetShaderInfoLog(fragment_shader, 512, NULL, infoLog);
            printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s", infoLog);
            return 0;
        }

        GLuint program = glCreateProgram();
        glAttachShader(program, vertex_shader);
        glAttachShader(program, fragment_shader);
        glLinkProgram(program);
        return program;
    }

    void shader_destroy(RHI& rhi, GLuint shader) {
        glDeleteProgram(shader);
    }
}