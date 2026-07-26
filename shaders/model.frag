#version 330 core
out vec4 FragColor;

in vec3 v_WorldPos;
in vec3 v_Normal;
in vec3 v_Color;

uniform vec3 uGlowColor;
uniform float uOpacity;
uniform float uTime;
uniform vec3 uViewPos;
uniform int uSolidMode; // 0 = Hologram Glow, 1 = Solid Original Colors

void main() {
    vec3 N = normalize(v_Normal);
    vec3 V = normalize(uViewPos - v_WorldPos);
    
    // Light vector: soft directional light from top-right-front
    vec3 L = normalize(vec3(0.5, 0.8, 0.5));
    
    // Diffuse shading
    float diff = max(dot(N, L), 0.0);
    
    // Specular shading (Blinn-Phong)
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.4;
    
    if (uSolidMode == 1) {
        // ── Solid Opaque Original Colors Mode ────────────────────────────
        float ambient = 0.45;
        vec3 col = v_Color * (diff * 0.7 + ambient) + vec3(spec);
        FragColor = vec4(col, 1.0); // 100% solid, fully opaque
    } else {
        // ── Holographic Cyan Glow Mode ────────────────────────────────────
        float ambient = 0.3;
        vec3 shaded_color = v_Color * (diff + ambient) + vec3(spec);
        
        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
        vec3 glow = uGlowColor * fresnel * 0.35;
        vec3 final_color = shaded_color + glow;
        
        FragColor = vec4(final_color, uOpacity);
    }
}
