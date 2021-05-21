#version 330 core

in vec2 uv;
out vec4 FragColor;

uniform usampler2D uMainTex;

const uint COLORS[11] = uint[11](
    0xffff00ffu, // Undefined
    0xffb3c6b7u, // Blank
    0xffe4d08cu, // Sand
    0xff0086ccu, // Water
    0xfff55545u, // Lava
    0xff765d59u, // Smoke
    0xff333333u, // Stone
    0xff00ff00u, // Debug0
    0xff00cc00u, // Debug1
    0xff00aa00u, // Debug2
    0xff005500u  // Debug3
);

void main()
{
    uint raw = texture(uMainTex, vec2(uv.x, 1.0 - uv.y)).r;
    raw &= 0x1fu;
    uint ucolor = COLORS[raw];
    float r = float((ucolor & 0x00ff0000u) >> 16) / 255.0;
    float g = float((ucolor & 0x0000ff00u) >>  8) / 255.0;
    float b = float((ucolor & 0x000000ffu) >>  0) / 255.0;

    FragColor = vec4(r, g, b, 1.0);
}
