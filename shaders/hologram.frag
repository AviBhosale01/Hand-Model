#version 330 core
out vec4 FragColor;

in vec3 v_Pos;

uniform vec3 uGlowColor;
uniform float uOpacity;
uniform float uTime;
uniform float uGlowIntensity;

void main() {
    // Generate pulse factor (pulsing between 0.7 and 1.3)
    float pulse = 1.0 + 0.3 * sin(uTime * 5.0);
    
    // Calculate a color that pulses and glows slightly brighter at edges
    vec3 final_color = uGlowColor * uGlowIntensity * pulse;
    
    // Output glowing neon cyan border lines with additive opacity
    FragColor = vec4(final_color, uOpacity);
}
