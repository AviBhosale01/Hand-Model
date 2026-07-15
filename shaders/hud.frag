#version 330 core
out vec4 FragColor;

in vec2 v_TexCoord;

uniform sampler2D uFontAtlas;
uniform vec3 uTextColor;

void main() {
    // Sample intensity (Pillow generated red channel atlas)
    float alpha = texture(uFontAtlas, v_TexCoord).r;
    
    if (alpha < 0.1) {
        discard;
    }
    
    FragColor = vec4(uTextColor, alpha);
}
