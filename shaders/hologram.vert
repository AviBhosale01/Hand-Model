#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 v_Pos;

void main() {
    v_Pos = aPos;
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
}
