#version 330 core
out vec4 FragColor;

in vec2 v_TexCoord;

uniform sampler2D uTexture;

void main() {
    vec3 color = texture(uTexture, v_TexCoord).rgb;
    
    // Calculate simple vignette
    vec2 uv = v_TexCoord - 0.5;
    float dist = length(uv);
    float vignette = smoothstep(0.8, 0.4, dist);
    
    // Darken background slightly and blend with vignette
    color = color * (vignette * 0.7 + 0.1);
    
    FragColor = vec4(color, 1.0);
}
