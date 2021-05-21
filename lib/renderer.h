/*
 * Derived from GLFW Examples, with the following license:
 *
 * Copyright (c) 2002-2006 Marcus Geelnard
 * Copyright (c) 2006-2019 Camilla Löwy
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgment in the product documentation would
 * be appreciated but is not required.
 *
 * 2. Altered source versions must be plainly marked as such, and must not
 * be misrepresented as being the original software.
 *
 * 3. This notice may not be removed or altered from any source
 * distribution.
 *
 */

#include "glad/gl.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "linmath.h"

#include <cassert>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <streambuf>

#include "grain_types.h"

namespace grain {

class OpenGLRenderer {
public:
    template<typename F>
    static void start(F compute_buffer_func, EventData& event_data, int w, int h) {
        // todo make neater

        /////////////////////////////
        // GLFW Setup
        GLFWwindow* window;
        GLuint vertex_buffer, vertex_shader, fragment_shader, program;
        GLint mvp_location, vpos_location, vuv_location;

        glfwSetErrorCallback(glfw_error_callback);

        if (!glfwInit()) {
            exit(EXIT_FAILURE);
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

        window = glfwCreateWindow(800, 800, "Simple example", NULL, NULL);
        if (!window) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwSetKeyCallback(window, keyboard_callback);

        glfwMakeContextCurrent(window);
        gladLoadGL(glfwGetProcAddress);
        glfwSwapInterval(1);

        /////////////////////////////
        // OpenGL Setup

        // NOTE: OpenGL error checks have been omitted for brevity
        const auto vertex_shader_text = load_text_file("../shaders/shader.vert");
        const auto fragment_shader_text = load_text_file("../shaders/shader.frag");

        // During init, enable debug output
        glEnable(GL_DEBUG_OUTPUT);
        glDebugMessageCallback(gl_error_callback, 0);

        glGenBuffers(1, &vertex_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        const char *vertex_sources[] = { vertex_shader_text.c_str() };
        glShaderSource(vertex_shader, 1, vertex_sources, NULL);
        glCompileShader(vertex_shader);

        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        const char *fragment_sources[] = { fragment_shader_text.c_str() };
        glShaderSource(fragment_shader, 1, fragment_sources, NULL);
        glCompileShader(fragment_shader);

        program = glCreateProgram();
        glAttachShader(program, vertex_shader);
        glAttachShader(program, fragment_shader);
        glLinkProgram(program);

        mvp_location = glGetUniformLocation(program, "MVP");
        vpos_location = glGetAttribLocation(program, "vPos");
        vuv_location = glGetAttribLocation(program, "vUV");

        glEnableVertexAttribArray(vpos_location);
        glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE,
                              sizeof(vertices[0]), (void*) 0);
        glEnableVertexAttribArray(vuv_location);
        glVertexAttribPointer(vuv_location, 2, GL_FLOAT, GL_FALSE,
                              sizeof(vertices[0]), (void*) (sizeof(float) * 2));

        /////////////////////////////
        // Main loop

        while (!glfwWindowShouldClose(window)) {
            float ratio;
            int width, height;
            mat4x4 m, p, mvp;

            glfwGetFramebufferSize(window, &width, &height);
            ratio = width / (float) height;

            glViewport(0, 0, width, height);
            glClear(GL_COLOR_BUFFER_BIT);

            mat4x4_identity(m);
            mat4x4_translate(p, 0.0, 0.0, 5.0);
            // mat4x4_rotate_Z(m, m, (float) glfwGetTime());
            mat4x4_ortho(p, -ratio, ratio, -1.f, 1.f, 1.f, -1.f);
            mat4x4_mul(mvp, p, m);

            glUseProgram(program);
            glUniformMatrix4fv(mvp_location, 1, GL_FALSE, (const GLfloat*) mvp);
            glDrawArrays(GL_TRIANGLES, 0, 6);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        glfwDestroyWindow(window);

        glfwTerminate();
    }

private:
    static constexpr struct {
        float x, y;
        float u, v;
    } vertices[6] = {
            {-1.0f, -1.0f, 0.f, 0.f,},
            {1.0f,  -1.0f, 0.f, 1.f,},
            {-1.f,  1.0f,  0.f, 0.f,},

            {1.0f,  -1.0f, 0.f, 0.f,},
            {1.0f,  1.0f,  0.f, 1.f,},
            {-1.f,  1.0f,  0.f, 0.f,},
    };

    static std::string load_text_file(const std::string &filename) {
        std::ifstream file(filename);
        assert(file.is_open());
        std::string str((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
        return str;
    }

    static void glfw_error_callback(int error, const char* description) {
        // todo use fmt
        fprintf(stderr, "Error: %s\n", description);
    }

    static void GLAPIENTRY gl_error_callback(GLenum source, GLenum type, GLuint id,
                                             GLenum severity, GLsizei length, const GLchar *message,
                                             const void *userParam) {
        // todo use fmt
        fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
                (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
                type, severity, message);
    }

    static void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
};

}
