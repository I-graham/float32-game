vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

float simplex_noise(vec2 v){
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
		   -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
  + i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
	dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

//perlin
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

float perlin_noise(float x, float y, float z)
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