#version 330 core
layout (location = 0) in vec4 aVal; // [x, y, z, life]

uniform mat4 uView;
uniform mat4 uProjection;

out float v_Life;

void main() {
    v_Life = aVal.w;
    
    // Render size scales down as particle reaches end of life (scaled by 12.0)
    gl_PointSize = 12.0 * v_Life;
    
    gl_Position = uProjection * uView * vec4(aVal.xyz, 1.0);
}
