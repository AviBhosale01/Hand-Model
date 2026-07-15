#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uNormalMatrix;

out vec3 v_WorldPos;
out vec3 v_Normal;

void main() {
    v_WorldPos = vec3(uModel * vec4(aPos, 1.0));
    v_Normal = normalize(mat3(uNormalMatrix) * aNormal);
    
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
}
