#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;

layout(location = 0) out vec3 f_color;
layout(location = 1) out vec3 m_coord;
layout(location = 2) out vec3 m_norm;

layout(set=0, binding=0)
uniform Uniforms {
	int time;
	vec3 cam_pos;
	mat4 cam_proj;
	vec3 light_pos;
	vec3 light_col;
	vec2 win_size;
};

layout(set=1, binding=0)
buffer Instances {
	mat4 models[];
};

void main() {

	mat4 model_matrix = models[gl_InstanceIndex];
	mat3 normal_matrix = mat3(transpose(inverse(model_matrix)));

	m_norm = normal_matrix * normal;

	vec4 model_space = model_matrix * vec4(position, 1.0);

	m_coord = model_space.xyz;

	gl_Position = cam_proj * model_space;

	f_color = color;

}