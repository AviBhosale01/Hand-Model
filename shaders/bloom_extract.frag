#version 330 core
out vec4 FragColor;

in vec2 v_TexCoord;

uniform sampler2D scene;
uniform float threshold;

void main() {
    vec4 color = texture(scene, v_TexCoord);
    
    // Calculate luminance using standard weights
    float luminance = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    
    if (luminance > threshold) {
        // Keep bright colors (scaling them up slightly for bloom effect)
        FragColor = vec4(color.rgb, 1.0);
    } else {
        // Output transparent black for dim areas
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
