#version 450

layout(location=0) in vec3 v_color;
layout(location=0) out vec4 f_color;

float gold_noise(in vec2 xy, in float seed){
	float PHI = 1.61803398874989484820459;  // Φ = Golden Ratio   
	return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}

void main() {
    f_color = vec4(v_color, 1.0);
}