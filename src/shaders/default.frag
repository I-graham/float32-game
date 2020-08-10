#version 450

layout(location = 0) in vec3 m_coord;
layout(location = 1) in vec4 f_color;
layout(location = 2) in vec3 m_norm;

layout(location=0) out vec4 out_color;

layout(set=0, binding=0)
uniform Uniforms {
	int time;
	vec3 cam_pos;
	mat4 cam_proj;
	vec3 light_pos;
	vec3 light_col;
	vec2 win_size;
};

vec3 ambient_lerp(vec3 normal) {
	return vec3(0.5, 0.8, 1.0)*(normal.y)+vec3(0.0, 0.3, 0.4)*(1.0-normal.y);
}

void main() {

	vec3 normal = normalize(m_norm);
	vec3 light_dir = normalize(light_pos - m_coord);

	float ambient_strength = 0.1;
	vec3 ambient_color = ambient_lerp(normal) * ambient_strength;

	float diffuse_strength = max(dot(normal, light_dir), 0.0);
	vec3 diffuse_color = light_col * diffuse_strength;

	vec3 view_dir = normalize(cam_pos - m_coord);
	vec3 half_dir = normalize(view_dir + light_dir);

	vec3 reflect_dir = reflect(-light_dir, normal);

	float specular_strength = pow(max(dot(normal, reflect_dir), 0.0), 32);
	vec3 specular_color = specular_strength * light_col;

	vec3 result = (ambient_color + diffuse_color + specular_color) * f_color.xyz;

	out_color = vec4(result, f_color.w);
}