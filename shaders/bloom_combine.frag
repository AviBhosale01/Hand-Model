#version 330 core
out vec4 FragColor;

in vec2 v_TexCoord;

uniform sampler2D scene;
uniform sampler2D bloom_blur;
uniform float bloom_intensity;

void main() {
    vec3 scene_color = texture(scene, v_TexCoord).rgb;
    vec3 bloom_color = texture(bloom_blur, v_TexCoord).rgb;
    
    // Additive blend
    vec3 result = scene_color + bloom_color * bloom_intensity;
    
    // 1. Reinhard Tone Mapping to fit HDR intensities into SDR space
    vec3 mapped = result / (result + vec3(1.0));
    
    // 2. Gamma correction (2.2)
    mapped = pow(mapped, vec3(1.0 / 2.2));
    
    FragColor = vec4(mapped, 1.0);
}
