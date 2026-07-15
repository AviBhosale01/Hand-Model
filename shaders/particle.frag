#version 330 core
out vec4 FragColor;

in float v_Life;

uniform vec3 uGlowColor;
uniform float uOpacity;

void main() {
    // Generate circular point sprite shape
    vec2 circ_coord = gl_PointCoord - vec2(0.5);
    float dist = length(circ_coord);
    
    // Discard pixels outside point boundary radius (0.5)
    if (dist > 0.5) {
        discard;
    }
    
    // Soft edges decay function
    float intensity = smoothstep(0.5, 0.0, dist);
    
    // Particle color fades out towards the end of its life (v_Life)
    vec3 color = uGlowColor * intensity * 2.0;
    float alpha = intensity * v_Life * uOpacity;
    
    FragColor = vec4(color, alpha);
}
