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
#include <chrono>
#include <fmt/format.h>

#include "grain_types.h"
#include "gpu_image.h"

namespace grain {

class OpenGLRenderer {
public:
    OpenGLRenderer() {
        OpenGLRenderer::print_usage();

        /////////////////////////////
        // GLFW Setup
        GLuint vertex_buffer, vertex_shader, fragment_shader;
        GLint vpos_location, vuv_location;

        glfwSetErrorCallback(glfw_error_callback);

        if (!glfwInit()) {
            exit(EXIT_FAILURE);
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        m_window = glfwCreateWindow(800, 800, "grain", nullptr, nullptr);
        if (!m_window) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(m_window);
        gladLoadGL(glfwGetProcAddress);
        glfwSwapInterval(1);

        /////////////////////////////
        // OpenGL Setup
        const auto vertex_shader_text = load_text_file("../shaders/shader.vert");
        const auto fragment_shader_text = load_text_file("../shaders/shader.frag");

        // During init, enable debug output
        glEnable(GL_DEBUG_OUTPUT);
        glDebugMessageCallback(gl_error_callback, nullptr);

        glGenBuffers(1, &vertex_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        const char *vertex_sources[] = { vertex_shader_text.c_str() };
        glShaderSource(vertex_shader, 1, vertex_sources, nullptr);
        glCompileShader(vertex_shader);
        check_shader(vertex_shader);


        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        const char *fragment_sources[] = { fragment_shader_text.c_str() };
        glShaderSource(fragment_shader, 1, fragment_sources, nullptr);
        glCompileShader(fragment_shader);
        check_shader(fragment_shader);

        m_program = glCreateProgram();
        glAttachShader(m_program, vertex_shader);
        glAttachShader(m_program, fragment_shader);
        glLinkProgram(m_program);

        m_loc_mvp = glGetUniformLocation(m_program, "uMVP");

        // from https://learnopengl.com/code_viewer_gh.php?code=src/1.getting_started/2.3.hello_triangle_exercise1/hello_triangle_exercise1.cpp
        unsigned int VBO;
        glGenVertexArrays(1, &m_VAO);
        glGenBuffers(1, &VBO);
        // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
        glBindVertexArray(m_VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                              4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                              4 * sizeof(float), (void*) (sizeof(float) * 2));
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindVertexArray(0);

        /////////////////////////////
        // Texture stuff

        // Create one OpenGL texture
        glGenTextures(1, &m_main_tex_id);

        // "Bind" the newly created texture : all future texture functions will modify this texture
        glBindTexture(GL_TEXTURE_2D, m_main_tex_id);

        // Give the image to OpenGL
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
        // GLint main_tex_location = glGetUniformLocation(m_program, "uMainTex");
        // glUniform1i(main_tex_location, textureID);
    }

    ~OpenGLRenderer() {
        // todo what else to clean up?
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    template<typename F>
    void start(F compute_buffer_func, EventData& event_data, int w, int h, bool verbose=false) {
        glfwSetWindowUserPointer(m_window, &event_data);
        glfwSetKeyCallback(m_window, keyboard_callback);
        glfwSetMouseButtonCallback(m_window, mouse_button_callback);

        /////////////////////////////
        // Main loop
        while (!glfwWindowShouldClose(m_window)) {
            using namespace std::chrono;
            const auto start_time = system_clock::now();

            int window_width, window_height;
            glfwGetFramebufferSize(m_window, &window_width, &window_height);
            const float ratio = window_width / (float) window_height;

            double xpos, ypos;
            glfwGetCursorPos(m_window, &xpos, &ypos);
            event_data.mouse_x = xpos / window_width;
            event_data.mouse_y = ypos / window_height;
            event_data.mouse_x = std::clamp(event_data.mouse_x, 0.0f, 1.0f);
            event_data.mouse_y = std::clamp(event_data.mouse_y, 0.0f, 1.0f);

            glViewport(0, 0, window_width, window_height);
            glClear(GL_COLOR_BUFFER_BIT);

            mat4x4 m, p, mvp;
            mat4x4_identity(m);
            mat4x4_translate(p, 0.0, 0.0, 5.0);
            // mat4x4_rotate_Z(m, m, (float) glfwGetTime());
            mat4x4_ortho(p, -ratio, ratio, -1.f, 1.f, 1.f, -1.f);
            mat4x4_mul(mvp, p, m);

            glUseProgram(m_program);
            glBindVertexArray(m_VAO);

            glBindTexture(GL_TEXTURE_2D, m_main_tex_id);

            const auto data = compute_buffer_func();
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, w, h,
                         0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, data);

            glUniformMatrix4fv(m_loc_mvp, 1, GL_FALSE, (const GLfloat*) mvp);
            glDrawArrays(GL_TRIANGLES, 0, 6);

            glfwSwapBuffers(m_window);
            glfwPollEvents();

            const auto end_time = system_clock::now();
            const double elapsed_seconds = duration_cast<duration<double>>(
                    end_time - start_time).count();
            if(verbose) {
                fmt::print("             [total_time: {:.6}ms] \n", elapsed_seconds*1000.0);
            }
        }
    }

private:
    GLFWwindow* m_window;
    int m_loc_mvp;
    uint m_program, m_VAO, m_main_tex_id;

    static constexpr struct {
        float x, y;
        float u, v;
    } vertices[6] = {
            {-1.0f, -1.0f, 0.f, 0.f,},
            {1.0f,  -1.0f, 1.f, 0.f,},
            {-1.f,  1.0f,  0.f, 1.f,},

            {1.0f,  -1.0f, 1.f, 0.f,},
            {1.0f,  1.0f,  1.f, 1.f,},
            {-1.f,  1.0f,  0.f, 1.f,},
    };

    static void print_usage() {
        std::cerr << "R:     Reset\n"
                  << "S:     Screenshot(overwrites screenshot.png in current dir)\n"
                  << "Q/E:   Previous/Next Brush\n"
                  << "Space: Toggle pause\n"
                  << "Esc:   Close\n";
    }

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

    static void check_shader(GLuint shader) {
        GLint isCompiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
        if (isCompiled == GL_FALSE) {
            GLint maxLength = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

            // The maxLength includes the NULL character
            std::vector<GLchar> errorLog(maxLength);
            glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);

            for(auto ch : errorLog) {
                std::cerr << ch;
            }
            std::cerr << '\n';

            // Provide the infolog in whatever manor you deem best.
            // Exit with failure.
            glDeleteShader(shader); // Don't leak the shader.
            throw std::runtime_error("Bad shader");
        }
    }

    static void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if(action != GLFW_PRESS) {
            return;
        }

        EventData& event_data = *((EventData*)glfwGetWindowUserPointer(window));
        switch(key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_R:
                event_data.reset = true;
                break;
            case GLFW_KEY_S:
                event_data.screenshot = true;
                break;
            case GLFW_KEY_Q:
                event_data.prev_brush = true;
                break;
            case GLFW_KEY_E:
                event_data.next_brush = true;
                break;
            case GLFW_KEY_SPACE:
                event_data.paused = !event_data.paused;
                break;
        }
    }

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            EventData& event_data = *((EventData*)glfwGetWindowUserPointer(window));
            event_data.mouse_pressed = action == GLFW_PRESS;
        }
    }

};

}
