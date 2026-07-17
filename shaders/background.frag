#version 330 core
out vec4 FragColor;

in vec2 v_TexCoord;

uniform sampler2D uTexture;

void main() {
    vec3 color = texture(uTexture, v_TexCoord).rgb;
    
    // Calculate a soft, high-quality vignette that dims only the outer edges
    // Keeping the center at 100% camera brightness for superior image quality
    vec2 uv = v_TexCoord - 0.5;
    float dist = length(uv);
    float vignette = smoothstep(0.8, 0.55, dist);
    
    color = color * (vignette * 0.35 + 0.65);
    
    FragColor = vec4(color, 1.0);
}
