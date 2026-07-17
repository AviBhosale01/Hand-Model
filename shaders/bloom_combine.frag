#version 330 core
out vec4 FragColor;

in vec2 v_TexCoord;

uniform sampler2D scene;
uniform sampler2D bloom_blur;
uniform float bloom_intensity;

void main() {
    vec4 scene_sample = texture(scene, v_TexCoord);
    vec3 scene_color = scene_sample.rgb;
    float scene_alpha = scene_sample.a;
    
    vec3 bloom_color = texture(bloom_blur, v_TexCoord).rgb;
    
    // Combine the HDR 3D scene and the blurred glow
    vec3 result = scene_color + bloom_color * bloom_intensity;
    
    // Tone mapping (Reinhard) & Gamma correction (2.2)
    vec3 mapped = result / (result + vec3(1.0));
    mapped = pow(mapped, vec3(1.0 / 2.2));
    
    // Set the transparency alpha dynamically based on 3D geometry presence & glow intensity
    // This allows it to blend on top of the webcam feed without modifying the camera frame directly
    float alpha = clamp(scene_alpha + length(bloom_color) * bloom_intensity, 0.0, 1.0);
    
    FragColor = vec4(mapped, alpha);
}
