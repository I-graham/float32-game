#version 450

layout(location = 0) in vec3 f_color;
layout(location = 1) in vec3 m_coord;
layout(location = 2) in vec3 m_norm;

layout(location=0) out vec4 out_color;

layout(set=0, binding=0)
uniform Uniforms {
	int time;
	vec3 cam_pos;
	mat4 cam_proj;
	vec3 light_pos;
	vec3 light_col;
};

int b(int N, int B) { return N>>B & 1; }
int T[] = int[](0x15,0x38,0x32,0x2c,0x0d,0x13,0x07,0x2a);
int A[] = int[](0,0,0);

int b(int i, int j, int k, int B) { return T[b(i,B)<<2 | b(j,B)<<1 | b(k,B)]; }

int shuffle(int i, int j, int k) {
    return b(i,j,k,0) + b(j,k,i,1) + b(k,i,j,2) + b(i,j,k,3) +
        b(j,k,i,4) + b(k,i,j,5) + b(i,j,k,6) + b(j,k,i,7) ;
}

float K(int a, vec3 uvw, vec3 ijk)
{
    float s = float(A[0]+A[1]+A[2])/6.0;
    float x = uvw.x - float(A[0]) + s,
        y = uvw.y - float(A[1]) + s,
        z = uvw.z - float(A[2]) + s,
        t = 0.6 - x * x - y * y - z * z;
    int h = shuffle(int(ijk.x) + A[0], int(ijk.y) + A[1], int(ijk.z) + A[2]);
    A[a]++;
    if (t < 0.0)
        return 0.0;
    int b5 = h>>5 & 1, b4 = h>>4 & 1, b3 = h>>3 & 1, b2= h>>2 & 1, b = h & 3;
    float p = b==1?x:b==2?y:z, q = b==1?y:b==2?z:x, r = b==1?z:b==2?x:y;
    p = (b5==b3 ? -p : p); q = (b5==b4 ? -q : q); r = (b5!=(b4^b3) ? -r : r);
    t *= t;
    return 8.0 * t * t * (p + (b==0 ? q+r : b2==0 ? q : r));
}

float noise(float x, float y, float z)
{
    float s = (x + y + z) / 3.0;  
    vec3 ijk = vec3(int(floor(x+s)), int(floor(y+s)), int(floor(z+s)));
    s = float(ijk.x + ijk.y + ijk.z) / 6.0;
    vec3 uvw = vec3(x - float(ijk.x) + s, y - float(ijk.y) + s, z - float(ijk.z) + s);
    A[0] = A[1] = A[2] = 0;
    int hi = uvw.x >= uvw.z ? uvw.x >= uvw.y ? 0 : 1 : uvw.y >= uvw.z ? 1 : 2;
    int lo = uvw.x <  uvw.z ? uvw.x <  uvw.y ? 0 : 1 : uvw.y <  uvw.z ? 1 : 2;
    return K(hi, uvw, ijk) + K(3 - hi - lo, uvw, ijk) + K(lo, uvw, ijk) + K(0, uvw, ijk);
}

float gold_noise(in vec2 xy, in float seed){
	const float PHI = 1.61803398874989484820459;  // Φ = Golden Ratio   
	return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}

void main() {

	float ambient_strength = 0.1;
	vec3 ambient_color = light_col * ambient_strength;

	vec3 normal = normalize(m_norm);
	vec3 light_dir = normalize(light_pos - m_coord);

	float diffuse_strength = max(dot(normal, light_dir), 0.0);
	vec3 diffuse_color = light_col * diffuse_strength;

	vec3 view_dir = normalize(cam_pos - m_coord);
	vec3 half_dir = normalize(view_dir + light_dir);

	vec3 reflect_dir = reflect(-light_dir, normal);

	float specular_strength = pow(max(dot(normal, reflect_dir), 0.0), 32);
	vec3 specular_color = specular_strength * light_col;

	vec3 result = (ambient_color + diffuse_color + specular_color) * f_color.xyz;
	
	out_color = vec4(result, 1.0);
}