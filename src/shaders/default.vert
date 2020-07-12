#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 f_color;

layout(set=0, binding=0, std140) uniform Uniforms {
	mat4 cam_proj;
	int time;
};

void main() {

	f_color = color * (time % 60) / 60.0;
	gl_Position = cam_proj * vec4(position.xy, 0.0, 1.0);

}