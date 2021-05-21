#version 330 core

layout (location=0) in vec2 inPos;
layout (location=1) in vec2 inUV;

out vec2 uv;

uniform mat4 uMVP;

void main()
{
    gl_Position = uMVP * vec4(inPos, 0.0, 1.0);
    uv = inUV;
}
