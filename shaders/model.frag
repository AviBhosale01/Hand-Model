#version 330 core
out vec4 FragColor;

in vec3 v_WorldPos;
in vec3 v_Normal;
in vec3 v_Color;

uniform vec3 uGlowColor;
uniform float uOpacity;
uniform float uTime;
uniform vec3 uViewPos;

void main() {
    vec3 N = normalize(v_Normal);
    vec3 V = normalize(uViewPos - v_WorldPos);
    
    // Light vector: soft directional light from top-right-front
    vec3 L = normalize(vec3(0.5, 0.8, 0.5));
    
    // Diffuse shading
    float diff = max(dot(N, L), 0.0);
    
    // Specular shading (Blinn-Phong)
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.3;
    
    // Soft ambient term
    float ambient = 0.3;
    
    // Shaded original color
    vec3 shaded_color = v_Color * (diff + ambient) + vec3(spec);
    
    // Subtly blend with a fresnel outer rim glow to match the holographic style
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    vec3 glow = uGlowColor * fresnel * 0.35;
    
    vec3 final_color = shaded_color + glow;
    
    // Output color with opacity animation (for gesture summon transitions)
    FragColor = vec4(final_color, uOpacity);
}
