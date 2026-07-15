#version 330 core
out vec4 FragColor;

in vec3 v_WorldPos;
in vec3 v_Normal;

uniform vec3 uGlowColor;
uniform float uOpacity;
uniform float uTime;
uniform vec3 uViewPos;

void main() {
    // 1. Fresnel Edge Glow (glowing boundaries)
    vec3 N = normalize(v_Normal);
    vec3 V = normalize(uViewPos - v_WorldPos);
    float dotProduct = dot(N, V);
    
    // Invert and shape Fresnel term (brighter at glance angles)
    float fresnel = pow(1.0 - abs(dotProduct), 3.0);
    
    // 2. Scanning line effect
    // Compute coordinate relative scanlines
    float scanline = sin(v_WorldPos.y * 80.0 - uTime * 6.0) * 0.5 + 0.5;
    
    // Add a faster micro scanline pattern for digital texture
    float micro_scan = sin(v_WorldPos.y * 300.0) * 0.2 + 0.8;
    
    // Combined neon emission texture
    float emission = (fresnel * 1.5) + (scanline * 0.4) * micro_scan;
    
    // 3. Base holographic lighting
    // Soft diffuse fallback to provide shape definition
    float diffuse = max(dot(N, vec3(0.0, 1.0, 0.0)), 0.0) * 0.1;
    
    vec3 final_color = uGlowColor * (emission + diffuse + 0.2);
    
    // Add subtle time-based brightness oscillation
    float pulse = 0.95 + 0.05 * sin(uTime * 8.0);
    final_color *= pulse;
    
    // Alpha blend: Fresnel stays opaque, center model is transparent
    float alpha = clamp(emission * uOpacity, 0.1 * uOpacity, 0.85 * uOpacity);
    
    FragColor = vec4(final_color, alpha);
}
