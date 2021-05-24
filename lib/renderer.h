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
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl2.h>

#include "grain_types.h"
#include "gpu_image.h"

namespace grain {

class OpenGLRenderer {
public:
    OpenGLRenderer(int width, int height) {
        OpenGLRenderer::print_usage();

        glfwSetErrorCallback(glfw_error_callback);

        if (!glfwInit()) {
            exit(EXIT_FAILURE);
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

        m_window = glfwCreateWindow(width, height, "grain", nullptr, nullptr);
        if (!m_window) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(m_window);
        gladLoadGL(glfwGetProcAddress);
        glfwSwapInterval(1);

        // Setup Imgui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(m_window, true);
        ImGui_ImplOpenGL2_Init();

        // During init, enable debug output
        glEnable(GL_DEBUG_OUTPUT);
        glDebugMessageCallback(gl_error_callback, nullptr);

        /////////////////////////////
        // Texture stuff
        glGenTextures(1, &m_main_tex_id);
        glBindTexture(GL_TEXTURE_2D, m_main_tex_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    ~OpenGLRenderer() {
        ImGui_ImplOpenGL2_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        // todo what else to clean up?
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    void start(GrainSim& grain_sim, EventData& event_data, int w, int h, bool verbose=false) {
        glfwSetWindowUserPointer(m_window, &event_data);
        glfwSetKeyCallback(m_window, keyboard_callback);
        glfwSetMouseButtonCallback(m_window, mouse_button_callback);

        // todo really need to get the image to work for non-square lol.
        assert(w == h);
        grain::GPUImage<uint32_t> display_image(w);

        /////////////////////////////
        // Main loop
        while (!glfwWindowShouldClose(m_window)) {
            Timer timer;

            glfwPollEvents();

            int window_width, window_height;
            glfwGetFramebufferSize(m_window, &window_width, &window_height);

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL2_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            // todo this may lag behind by one frame?
            const auto stats = grain_sim.stats();
            imgui_overlay(stats);

            // get mouse position
            double xpos, ypos;
            glfwGetCursorPos(m_window, &xpos, &ypos);
            event_data.mouse_x = xpos / window_width;
            event_data.mouse_y = ypos / window_height;
            event_data.mouse_x = std::clamp(event_data.mouse_x, 0.0f, 1.0f);
            event_data.mouse_y = std::clamp(event_data.mouse_y, 0.0f, 1.0f);

            // Clear screen
            ImGui::Render();
            glViewport(0, 0, window_width, window_height);
            glClear(GL_COLOR_BUFFER_BIT);

            // Setup texture
            glBindTexture(GL_TEXTURE_2D, m_main_tex_id);

            //TODO use PBO so that cuda and opengl share memory
            grain_sim.update(event_data);
            grain_sim.as_color_image(display_image);
            const uint32_t* data = display_image.data();

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h,
                         0, GL_BGRA, GL_UNSIGNED_BYTE, data);
            glEnable(GL_TEXTURE_2D);

            // Setup a full screen quad
            glBegin(GL_QUADS);
            glTexCoord2f(0, 1);
            glVertex2f(-1, -1);
            glTexCoord2f(1, 1);
            glVertex2f(1, -1);
            glTexCoord2f(1, 0);
            glVertex2f(1, 1);
            glTexCoord2f(0, 0);
            glVertex2f(-1, 1);
            glEnd();

            glDisable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, 0);
            glDrawArrays(GL_TRIANGLES, 0, 6);

            // Draw ImGUI stuff
            ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(m_window);

            if(verbose) {
                const double t = timer.elapsed();
                fmt::print("             [total_time: {:6g}ms / {:6g}its/s] \n",
                           t*1000.0, 1.0 / t);
            }
        }
    }

private:
    GLFWwindow* m_window;
    uint m_main_tex_id;

    void imgui_overlay(const grain::Stats& stats) {
        ImGui::Begin("Stats");
        const double sim_framerate = 1.0 / stats.last_update_time;
        ImGui::Text("Render: %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::Text("Sim:    %.3f ms/frame (%.1f FPS)", 1000.0f / sim_framerate, sim_framerate);
        ImGui::End();
    }

    static void print_usage() {
        std::cerr << "R:     Reset\n"
                  << "S:     Screenshot(overwrites screenshot.png in current dir)\n"
                  << "Q/E:   Previous/Next Brush\n"
                  << "Space: Toggle pause\n"
                  << "Esc:   Close\n";
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
