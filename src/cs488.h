// =======================================
// CS488/688 base code
// (written by Toshiya Hachisuka)
// =======================================
#pragma once
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX


// OpenGL
#define GLEW_STATIC

#include <GL/glew.h>
#include <GLFW/glfw3.h>


// image loader and writer
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"


// linear algebra 
#include "linalg.h"
class TriangleMesh;
using namespace linalg::aliases;


// animated GIF writer
#include "gif.h"


// misc
#include <iostream>
#include <vector>
#include <cfloat>
#include <random>



// main window
static GLFWwindow* globalGLFWindow;


// window size and resolution
// (do not make it too large - will be slow!)
constexpr int globalWidth = 512;
constexpr int globalHeight = 384;


// degree and radian
constexpr float PI = 3.14159265358979f;
constexpr float DegToRad = PI / 180.0f;
constexpr float RadToDeg = 180.0f / PI;


// for ray tracing
constexpr float Epsilon = 5e-5f;


// amount the camera moves with a mouse and a keyboard
constexpr float ANGFACT = 0.2f;
constexpr float SCLFACT = 0.1f;


// fixed camera parameters
constexpr float globalAspectRatio = float(globalWidth / float(globalHeight));
constexpr float globalFOV = 45.0f; // vertical field of view
constexpr float globalDepthMin = Epsilon; // for rasterization
constexpr float globalDepthMax = 100.0f; // for rasterization
constexpr float globalFilmSize = 0.032f; //for ray tracing
const float globalDistanceToFilm = globalFilmSize / (2.0f * tan(globalFOV * DegToRad * 0.5f)); // for ray tracing


// particle system related
bool globalEnableParticles = false;
constexpr float deltaT = 0.002f;
constexpr float3 globalGravity = float3(0.0f, -9.8f, 0.0f);
constexpr int globalNumParticles = 300;


// dynamic camera parameters
float3 globalEye = float3(-1.5f, 0.5f, -2.5f);
float3 globalLookat = float3(0.5f, -0.4f, 0.0f);
float3 globalUp = normalize(float3(0.0f, 1.0f, 0.0f));
float3 globalViewDir; // should always be normalize(globalLookat - globalEye)
float3 globalRight; // should always be normalize(cross(globalViewDir, globalUp));
bool globalShowRaytraceProgress = false; // for ray tracing


// mouse event
static bool mouseLeftPressed;
static double m_mouseX = 0.0;
static double m_mouseY = 0.0;


// rendering algorithm
enum enumRenderType {
	RENDER_RASTERIZE,
	RENDER_RAYTRACE,
	RENDER_IMAGE,
	RENDER_PATHTRACE
};
enumRenderType globalRenderType = RENDER_IMAGE;
int globalFrameCount = 0;
static bool globalRecording = false;
static GifWriter globalGIFfile;
constexpr int globalGIFdelay = 1;


// OpenGL related data (do not modify it if it is working)
static GLuint GLFrameBufferTexture;
static GLuint FSDraw;
static const std::string FSDrawSource = R"(
    #version 120

    uniform sampler2D input_tex;
    uniform vec4 BufInfo;

    void main()
    {
        gl_FragColor = texture2D(input_tex, gl_FragCoord.st * BufInfo.zw);
    }
)";
static const char* PFSDrawSource = FSDrawSource.c_str();


// fast random number generator based pcg32_fast
#include <stdint.h>
namespace PCG32 {
	static uint64_t mcg_state = 0xcafef00dd15ea5e5u;	// must be odd
	static uint64_t const multiplier = 6364136223846793005u;
	uint32_t pcg32_fast(void) {
		uint64_t x = mcg_state;
		const unsigned count = (unsigned)(x >> 61);
		mcg_state = x * multiplier;
		x ^= x >> 22;
		return (uint32_t)(x >> (22 + count));
	}
	float rand() {
		return float(double(pcg32_fast()) / 4294967296.0);
	}
}

// image with a depth buffer
// (depth buffer is not always needed, but hey, we have a few GB of memory, so it won't be an issue...)
class Image {
public:
	std::vector<float3> pixels;
	std::vector<float> depths;
	int width = 0, height = 0;

	static float toneMapping(const float r) {
		// you may want to implement better tone mapping
		return std::max(std::min(1.0f, r), 0.0f);
	}

	static float gammaCorrection(const float r, const float gamma = 1.0f) {
		// assumes r is within 0 to 1
		// gamma is typically 2.2, but the default is 1.0 to make it linear
		return pow(r, 1.0f / gamma);
	}

	void resize(const int newWdith, const int newHeight) {
		this->pixels.resize(newWdith * newHeight);
		this->depths.resize(newWdith * newHeight);
		this->width = newWdith;
		this->height = newHeight;
	}

	void clear() {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				this->pixel(i, j) = float3(0.0f);
				this->depth(i, j) = FLT_MAX;
			}
		}
	}

	Image(int _width = 0, int _height = 0) {
		this->resize(_width, _height);
		this->clear();
	}

	bool valid(const int i, const int j) const {
		return (i >= 0) && (i < this->width) && (j >= 0) && (j < this->height);
	}

	float& depth(const int i, const int j) {
		return this->depths[i + j * width];
	}

	float3& pixel(const int i, const int j) {
		// optionally can check with "valid", but it will be slow
		return this->pixels[i + j * width];
	}

	void load(const char* fileName) {
		int comp, w, h;
		float* buf = stbi_loadf(fileName, &w, &h, &comp, 3);
		if (!buf) {
			std::cerr << "Unable to load: " << fileName << std::endl;
			return;
		}

		this->resize(w, h);
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				this->pixels[i + j * width] = float3(buf[k], buf[k + 1], buf[k + 2]);
				k += 3;
			}
		}
		delete[] buf;
		printf("Loaded \"%s\".\n", fileName);
	}
	void save(const char* fileName) {
		unsigned char* buf = new unsigned char[width * height * 3];
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).x)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).y)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).z)));
			}
		}
		stbi_write_png(fileName, width, height, 3, buf, width * 3);
		delete[] buf;
		printf("Saved \"%s\".\n", fileName);
	}
};

// main image buffer to be displayed
Image FrameBuffer(globalWidth, globalHeight);

// you may want to use the following later for progressive ray tracing
Image AccumulationBuffer(globalWidth, globalHeight);
unsigned int sampleCount = 10;



// keyboard events (you do not need to modify it unless you want to)
void keyFunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) {
			case GLFW_KEY_R: {
				if (globalRenderType == RENDER_RAYTRACE) {
					printf("(Switched to rasterization)\n");
					glfwSetWindowTitle(window, "Rasterization mode");
					globalRenderType = RENDER_RASTERIZE;
				} else if (globalRenderType == RENDER_RASTERIZE) {
					printf("(Switched to ray tracing)\n");
					AccumulationBuffer.clear();
					sampleCount = 0;
					glfwSetWindowTitle(window, "Ray tracing mode");
					globalRenderType = RENDER_RAYTRACE;
				}
			break;}

			case GLFW_KEY_ESCAPE: {
				glfwSetWindowShouldClose(window, GL_TRUE);
			break;}

			case GLFW_KEY_I: {
				char fileName[1024];
				sprintf(fileName, "output%d.png", int(1000.0 * PCG32::rand()));
				FrameBuffer.save(fileName);
			break;}

			case GLFW_KEY_F: {
				if (!globalRecording) {
					char fileName[1024];
					sprintf(fileName, "output%d.gif", int(1000.0 * PCG32::rand()));
					printf("Saving \"%s\"...\n", fileName);
					GifBegin(&globalGIFfile, fileName, globalWidth, globalHeight, globalGIFdelay);
					globalRecording = true;
					printf("(Recording started)\n");
				} else {
					GifEnd(&globalGIFfile);
					globalRecording = false;
					printf("(Recording done)\n");
				}
			break;}

			case GLFW_KEY_W: {
				globalEye += SCLFACT * globalViewDir;
				globalLookat += SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_S: {
				globalEye -= SCLFACT * globalViewDir;
				globalLookat -= SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_Q: {
				globalEye += SCLFACT * globalUp;
				globalLookat += SCLFACT * globalUp;
			break;}

			case GLFW_KEY_Z: {
				globalEye -= SCLFACT * globalUp;
				globalLookat -= SCLFACT * globalUp;
			break;}

			case GLFW_KEY_A: {
				globalEye -= SCLFACT * globalRight;
				globalLookat -= SCLFACT * globalRight;
			break;}

			case GLFW_KEY_D: {
				globalEye += SCLFACT * globalRight;
				globalLookat += SCLFACT * globalRight;
			break;}

			default: break;
		}
	}
}



// mouse button events (you do not need to modify it unless you want to)
void mouseButtonFunc(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			mouseLeftPressed = true;
		} else if (action == GLFW_RELEASE) {
			mouseLeftPressed = false;
			if (globalRenderType == RENDER_RAYTRACE) {
				AccumulationBuffer.clear();
				sampleCount = 0;
			}
		}
	}
}



// mouse button events (you do not need to modify it unless you want to)
void cursorPosFunc(GLFWwindow* window, double mouse_x, double mouse_y) {
	if (mouseLeftPressed) {
		const float xfact = -ANGFACT * float(mouse_y - m_mouseY);
		const float yfact = -ANGFACT * float(mouse_x - m_mouseX);
		float3 v = globalViewDir;

		// local function in C++...
		struct {
			float3 operator()(float theta, const float3& v, const float3& w) {
				const float c = cosf(theta);
				const float s = sinf(theta);

				const float3 v0 = dot(v, w) * w;
				const float3 v1 = v - v0;
				const float3 v2 = cross(w, v1);

				return v0 + c * v1 + s * v2;
			}
		} rotateVector;

		v = rotateVector(xfact * DegToRad, v, globalRight);
		v = rotateVector(yfact * DegToRad, v, globalUp);
		globalViewDir = v;
		globalLookat = globalEye + globalViewDir;
		globalRight = cross(globalViewDir, globalUp);

		m_mouseX = mouse_x;
		m_mouseY = mouse_y;

		if (globalRenderType == RENDER_RAYTRACE) {
			AccumulationBuffer.clear();
			sampleCount = 0;
		}
	} else {
		m_mouseX = mouse_x;
		m_mouseY = mouse_y;
	}
}




class PointLightSource {
public:
	float3 position, wattage;
};



class Ray {
public:
	float3 o, d;
	Ray() : o(), d(float3(0.0f, 0.0f, 1.0f)) {}
	Ray(const float3& o, const float3& d) : o(o), d(d) {}
};



// uber material
// "type" will tell the actual type
// ====== implement it in A1, if you want ======
enum enumMaterialType {
	MAT_LAMBERTIAN,
	MAT_METAL,
	MAT_GLASS,
	MAT_COOKTORR,
};
class Material {
public:
	std::string name;

	enumMaterialType type = MAT_LAMBERTIAN;
	float eta = 1.0f;
	float glossiness = 1.0f;

	float3 Ka = float3(0.0f);
	float3 Kd = float3(0.9f);
	float3 Ks = float3(0.0f);
	float Ns = 0.0;

	float3 albedo = float3(0.1f, 0.5f, 0.9f); // surface colour
	float3 emission;
	float metallic = 0.3f; // metallic coefficient
	float roughness = 0.3f; // surface roughness
	float ior; // index of refraction

	// support 8-bit texture
	bool isTextured = false;
	unsigned char* texture = nullptr;
	int textureWidth = 0;
	int textureHeight = 0;
	int d;


	Material() {};
	virtual ~Material() {};

	// --- cook torrance helper functions ---

	// brdf = diffuse + specular
	// lambertian diffuse = kd * albedo / pi
	// specular = ks * (fresnel * normal distribution * geometry) / (4 * (inner product of normal and incident ray) * (inner product of normal and view ray))
	// using Schlick for fresnel term, GGX for normal distribution, and Smith's geometry term which uses schlick-ggx geometry fn

	float3 fresnelSchlick(float cosTheta, float3 f0) const { // cosTheta = inner product of view direction and half vector
		return f0 + (1.0f - f0) * pow(1.0f - cosTheta, 5.0f);
	}

	float ggx(float3 n, float3 h, float alpha) const { // alpha = roughness^2, n = normal, h = half vector
		float NoH = fmax(dot(n,h), 0.0f);
		float a2 = alpha * alpha;
		float den = pow(NoH, 2.0f) * (a2 - 1)  + 1;
		float num = a2  * (NoH > 0.0f ? 1.0f : 0.0f);
		return num / ( PI * den * den );
	}

	float smithG(float3 n, float3 v, float3 l, float alpha) const { // alpha = roughness^2, n = normal, v = view direction, l = incident direction
		float NoV = fmax(dot(n, v), 0.0f); // cosine of angle between normal and view ray
		float NoL = fmax(dot(n, l), 0.0f); // cosine of angle between normal and incident ray
		float k = alpha / 2; // roughness
		float g1 = NoV / (NoV * (1-k) + k); // schlick ggx geometry terms
		float g2 = NoL / (NoL * (1-k) + k);
		return g1 * g2;
	}

	float3 microfacetBRDF (float3 n, float3 v, float3 l, float alpha, float3 albedo, float metallic) const {
		float3 h = normalize(v + l); // half vector between light and view ray
		float3 f0 = (1.0f - metallic) * float3(0.04f) + metallic * albedo; // surface reflection at 0 incidence

		float3 f = fresnelSchlick(fmax(dot(v, h), 0.0f), f0);
		float d = ggx(n, h, alpha);
		float g = smithG(n, v, l, alpha);
		float3 spec = f * d * g / (4 * fmax(dot(n, l), 0.0f) * fmax(dot(n, v), 0.0f) + 0.0001f);

		float3 diffuse = (albedo * (1 - metallic) * (1 - f)) / PI; // albedo is 0 when metallic is 1, and follow energy conservation principle according to fresnel term

		return diffuse + spec;
	}

	// --- end of cook torrance helper functions ---

	void setReflectance(const float3& c) {
		if (type == MAT_LAMBERTIAN) {
			Kd = c;
		} else if (type == MAT_METAL) {
			Ks = c;
		} else if (type == MAT_GLASS) {
			Ks = c;
		}
	}

	float3 fetchTexture(const float2& tex) const {
		// repeating
		int x = int(tex.x * textureWidth) % textureWidth;
		int y = int(tex.y * textureHeight) % textureHeight;
		if (x < 0) x += textureWidth;
		if (y < 0) y += textureHeight;

		int pix = (x + y * textureWidth) * 3;
		const unsigned char r = texture[pix + 0];
		const unsigned char g = texture[pix + 1];
		const unsigned char b = texture[pix + 2];
		return float3(r, g, b) / 255.0f;
	}

	float3 BRDF(const float3& wi, const float3& wo, const float3& n) const {
		float3 brdfValue = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// BRDF
			brdfValue = Kd / PI;
		} else if (type == MAT_METAL || type == MAT_GLASS) {
			//float alpha = roughness * roughness;
			//return microfacetBRDF(n, wo, wi, alpha, albedo, metallic);
		}
		return brdfValue;
	};

	float PDF(const float3& wGiven, const float3& normal) const {
		// probability density function for a given direction and a given sample
		// it has to be consistent with the sampler
		return dot(normal, wGiven) / PI;
	}

	float3 uniformSampleHemisphere(const float &r1, const float &r2) const {
		// cos(theta) = r1 = y
		// cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = sqrtf(1 - cos^2(theta))
		float sinTheta = sqrtf(1 - r1 * r1);
		float phi = 2 * M_PI * r2;
		float x = sinTheta * cosf(phi);
		float z = sinTheta * sinf(phi);
		return float3(x, r1, z);
	}

	float3 sampleGGX(float3 N, float roughness, float2 xi) const {
		float a = roughness * roughness;

		float phi = 2.0f * PI * xi.x;
		float cosTheta = sqrtf((1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y));
		float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

		// spherical coords to cartesian coords
		float3 H = float3(
			sinTheta * cosf(phi),
			sinTheta * sinf(phi),
			cosTheta
		);

		// transform to world space
		float3 up = fabs(N.y) < 0.999f ? float3(0, 1, 0) : float3(1, 0, 0);
		float3 tangent = normalize(cross(up, N));
		float3 bitangent = cross(N, tangent);
		float3 sample = normalize(tangent * H.x + bitangent * H.y + N * H.z);

		return sample;
	}


	float3 sampler(const float3& wGiven) const {
		// sample a vector and record its probability density as pdfValue
		float3 smp = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			/*float r1 = PCG32::rand();
			float r2 = PCG32::rand();
			float3 sample = uniformSampleHemisphere(r1, r2);
			float pdfValue = dot(smp, float3(0, 1, 0)) / PI;
			return sample;*/
		} else if (type == MAT_METAL) {
			/*float r1 = PCG32::rand();
			float r2 = PCG32::rand();
			float alpha = roughness * roughness;
			float3 N = float3(0, 1, 0); // local frame, adjust if using world normals
			float3 V = wGiven;         // assumed normalized

			smp = sampleGGX(N, roughness, float2(r1, r2));
			float pdfValue = PDF(wGiven, smp);
			return smp;*/
		} else if (type == MAT_GLASS) {
			// empty
		}

		float pdfValue = PDF(wGiven, smp);
		return smp;
	}
};

float clamp(float x, float lowerlimit = 0.0f, float upperlimit = 1.0f) {
	if (x < lowerlimit) return lowerlimit;
	if (x > upperlimit) return upperlimit;
	return x;
}
float smoothstep (float edge0, float edge1, float x) {
	// Scale, and clamp x to 0..1 range
	x = clamp((x - edge0) / (edge1 - edge0));

	return x * x * (3.0f - 2.0f * x);
}


// noisy background
class Environment {
public:

	float2 fract(float2 x) {
		return x - floor(x);
	}

	float lerp1(float a, float b, float t) {
		return a + t * (b - a);
	}


	float noise (float2 p) {
		float2 i = floor(p);
		float2 f = fract(p);

		// Four corners in 2D of a tile
		float a = PCG32::rand();
		float b = PCG32::rand();
		float c = PCG32::rand();
		float d = PCG32::rand();

		float2 u = f * f * (float2(3.0f) - float2(2.0f) * f);


		return lerp1(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
	}

	// Fractal Brownian Motion noise function
	float fbm(float2 p) {
		float value = 0.0f;
		float amplitude = 0.5f;

		for (int i = 0; i < 6; i++) {
			value += amplitude * noise(p);
			amplitude *= 0.5f;
			p *= 2.0f;
		}

		return value;
	}

	float3 blurryEnv(const float3& direction) {
		// sky colour
		float3 skyColor = float3(0.5f, 0.7f, 1.0f);
		float3 horizonColor = float3(0.8f, 0.85f, 0.9f);

		// ground colour
		float3 groundColor = float3(0.2f, 0.3f, 0.1f);

		// y component for sky-ground blend
		float y = direction.y;

		// noise for cloudiness and blur
		float2 sphereCoords = float2(
			atan2(direction.z, direction.x),
			acos(y)
		);

		float cloudNoise = fbm(sphereCoords * 4.0f) * 0.5f + 0.5f;

		// blending sky and ground
		float blend = smoothstep(-0.1f, 0.1f, y);
		float3 baseColor = lerp(groundColor, skyColor, blend);

		float cloudInfluence = smoothstep(-0.2f, 0.5f, y);
		float3 cloudColor = float3(1.0f, 1.0f, 1.0f);
		baseColor = lerp(baseColor, cloudColor, cloudNoise * cloudInfluence * 0.5f);

		//  atmospheric haze
		float haze = pow(1.0f - abs(y), 2.0f);
		baseColor = lerp(baseColor, horizonColor, haze * 0.6f);

		return baseColor;
	}

};



class HitInfo {
public:
	float t; // distance
	float3 P; // location
	float3 N; // shading normal vector
	float2 T; // texture coordinate
	const Material* material; // const pointer to the material of the intersected object
};



// axis-aligned bounding box
class AABB {
private:
	float3 minp, maxp, size;

public:
	float3 get_minp() const { return minp; };
	float3 get_maxp() const { return maxp; };
	float3 get_size() const { return size; };


	AABB() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	void reset() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	int getLargestAxis() const {
		if ((size.x > size.y) && (size.x > size.z)) {
			return 0;
		} else if (size.y > size.z) {
			return 1;
		} else {
			return 2;
		}
	}

	void fit(const float3& v) {
		if (minp.x > v.x) minp.x = v.x;
		if (minp.y > v.y) minp.y = v.y;
		if (minp.z > v.z) minp.z = v.z;

		if (maxp.x < v.x) maxp.x = v.x;
		if (maxp.y < v.y) maxp.y = v.y;
		if (maxp.z < v.z) maxp.z = v.z;

		size = maxp - minp;
	}

	float area() const {
		return (2.0f * (size.x * size.y + size.y * size.z + size.z * size.x));
	}


	bool intersect(HitInfo& minHit, const Ray& ray) const {
		// set minHit.t as the distance to the intersection point
		// return true/false if the ray hits or not
		float tx1 = (minp.x - ray.o.x) / ray.d.x;
		float ty1 = (minp.y - ray.o.y) / ray.d.y;
		float tz1 = (minp.z - ray.o.z) / ray.d.z;

		float tx2 = (maxp.x - ray.o.x) / ray.d.x;
		float ty2 = (maxp.y - ray.o.y) / ray.d.y;
		float tz2 = (maxp.z - ray.o.z) / ray.d.z;

		if (tx1 > tx2) {
			const float temp = tx1;
			tx1 = tx2;
			tx2 = temp;
		}

		if (ty1 > ty2) {
			const float temp = ty1;
			ty1 = ty2;
			ty2 = temp;
		}

		if (tz1 > tz2) {
			const float temp = tz1;
			tz1 = tz2;
			tz2 = temp;
		}

		float t1 = tx1; if (t1 < ty1) t1 = ty1; if (t1 < tz1) t1 = tz1;
		float t2 = tx2; if (t2 > ty2) t2 = ty2; if (t2 > tz2) t2 = tz2;

		if (t1 > t2) return false;
		if ((t1 < 0.0) && (t2 < 0.0)) return false;

		minHit.t = t1;
		return true;
	}
};




// triangle
struct Triangle {
	float3 positions[3];
	float3 normals[3];
	float2 texcoords[3];
	int idMaterial = 0;
	AABB bbox;
	float3 center;
};



// triangle mesh
static float3 shade(const HitInfo& hit, const float3& viewDir, const int level = 0);
class TriangleMesh {
public:
	std::vector<Triangle> triangles;
	std::vector<Material> materials;
	AABB bbox;

	void transform(const float4x4& m) {
		// ====== implement it if you want =====
		// matrix transformation of an object	
		// m is a matrix that transforms an object
		// implement proper transformation for positions and normals
		// (hint: you will need to have float4 versions of p and n)
		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			for (int k = 0; k <= 2; k++) {
				const float3 &p = this->triangles[i].positions[k];
				const float3 &n = this->triangles[i].normals[k];
				// not doing anything right now
			}
		}
	}


	bool raytraceTriangle(HitInfo& result, const Ray& ray, const Triangle& tri, float tMin, float tMax) const {
		// ====== implement it in A1 ======
		// ray-triangle intersection
		// fill in "result" when there is an intersection
		// return true/false if there is an intersection or not

		// sides of triangle
		float3 sideA = tri.positions[1] - tri.positions[0];
		float3 sideB = tri.positions[2] - tri.positions[0];

		float3 P = cross(ray.d, sideB);
		float det = dot(sideA, P);

		// checking to see if plane and ray are parallel, which means no or infinite intersections with the plane
		if (det < 1e-8f || fabs(det) < 1e-8f) return false;

		// Cramer's Rule

		float3 T = ray.o - tri.positions[0]; // distance from ray's origin to point A of triangle

		float u = dot(P, T);
		if (u < 0.0f || u > det) return false;

		float3 Q = cross(T, sideA);

		float v = dot(Q, ray.d);
		if (v < 0.0f || u + v > det) return false;

		float t = dot(Q, sideB);
		if (t < tMin * det || t > tMax * det) return false;

		// otherwise there is an intersection
		float invDet = 1.0f / det;
		u *= invDet;
		v *= invDet;
		t *= invDet;

		result.t = t;

		float3 intersection = ray.o + t * ray.d;
		result.P = intersection;

		float w = 1 - u - v;

		result.N = linalg::normalize(tri.normals[0] * u + tri.normals[1] * v + tri.normals[2] * w);
		result.T = tri.texcoords[0] * u + tri.texcoords[1] * v + tri.texcoords[2] * w;
		result.material = &materials[tri.idMaterial];

		return true;
	}


	// some precalculation for bounding boxes (you do not need to change it)
	void preCalc() {
		bbox.reset();
		for (int i = 0, _n = (int)triangles.size(); i < _n; i++) {
			this->triangles[i].bbox.reset();
			this->triangles[i].bbox.fit(this->triangles[i].positions[0]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[1]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[2]);

			this->triangles[i].center = (this->triangles[i].positions[0] + this->triangles[i].positions[1] + this->triangles[i].positions[2]) * (1.0f / 3.0f);

			this->bbox.fit(this->triangles[i].positions[0]);
			this->bbox.fit(this->triangles[i].positions[1]);
			this->bbox.fit(this->triangles[i].positions[2]);
		}
	}


	// load .obj file (you do not need to modify it unless you want to change something)
	bool load(const char* filename, const float4x4& ctm = linalg::identity) {
		int nVertices = 0;
		float* vertices;
		float* normals;
		float* texcoords;
		int nIndices;
		int* indices;
		int* matid = nullptr;

		printf("Loading \"%s\"...\n", filename);
		ParseOBJ(filename, nVertices, &vertices, &normals, &texcoords, nIndices, &indices, &matid);
		if (nVertices == 0) return false;
		this->triangles.resize(nIndices / 3);

		if (matid != nullptr) {
			for (unsigned int i = 0; i < materials.size(); i++) {
				// convert .mlt data into BSDF definitions
				// you may change the followings in the final project if you want
				materials[i].type = MAT_LAMBERTIAN;
				if (materials[i].Ns == 100.0f) {
					materials[i].type = MAT_METAL;
					materials[i].metallic = 1.0f;
				}
				if (materials[i].name.compare(0, 5, "glass", 0, 5) == 0)
				{
					materials[i].type = MAT_GLASS;
					materials[i].eta = 1.5f;
				}
			}
		} else {
			// use default Lambertian
			this->materials.resize(1);
		}

		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			const int v0 = indices[i * 3 + 0];
			const int v1 = indices[i * 3 + 1];
			const int v2 = indices[i * 3 + 2];

			this->triangles[i].positions[0] = float3(vertices[v0 * 3 + 0], vertices[v0 * 3 + 1], vertices[v0 * 3 + 2]);
			this->triangles[i].positions[1] = float3(vertices[v1 * 3 + 0], vertices[v1 * 3 + 1], vertices[v1 * 3 + 2]);
			this->triangles[i].positions[2] = float3(vertices[v2 * 3 + 0], vertices[v2 * 3 + 1], vertices[v2 * 3 + 2]);

			if (normals != nullptr) {
				this->triangles[i].normals[0] = float3(normals[v0 * 3 + 0], normals[v0 * 3 + 1], normals[v0 * 3 + 2]);
				this->triangles[i].normals[1] = float3(normals[v1 * 3 + 0], normals[v1 * 3 + 1], normals[v1 * 3 + 2]);
				this->triangles[i].normals[2] = float3(normals[v2 * 3 + 0], normals[v2 * 3 + 1], normals[v2 * 3 + 2]);
			} else {
				// no normal data, calculate the normal for a polygon
				const float3 e0 = this->triangles[i].positions[1] - this->triangles[i].positions[0];
				const float3 e1 = this->triangles[i].positions[2] - this->triangles[i].positions[0];
				const float3 n = normalize(cross(e0, e1));

				this->triangles[i].normals[0] = n;
				this->triangles[i].normals[1] = n;
				this->triangles[i].normals[2] = n;
			}

			// material id
			this->triangles[i].idMaterial = 0;
			if (matid != nullptr) {
				// read texture coordinates
				if ((texcoords != nullptr) && materials[matid[i]].isTextured) {
					this->triangles[i].texcoords[0] = float2(texcoords[v0 * 2 + 0], texcoords[v0 * 2 + 1]);
					this->triangles[i].texcoords[1] = float2(texcoords[v1 * 2 + 0], texcoords[v1 * 2 + 1]);
					this->triangles[i].texcoords[2] = float2(texcoords[v2 * 2 + 0], texcoords[v2 * 2 + 1]);
				} else {
					this->triangles[i].texcoords[0] = float2(0.0f);
					this->triangles[i].texcoords[1] = float2(0.0f);
					this->triangles[i].texcoords[2] = float2(0.0f);
				}
				this->triangles[i].idMaterial = matid[i];
			} else {
				this->triangles[i].texcoords[0] = float2(0.0f);
				this->triangles[i].texcoords[1] = float2(0.0f);
				this->triangles[i].texcoords[2] = float2(0.0f);
			}
		}
		printf("Loaded \"%s\" with %d triangles.\n", filename, int(triangles.size()));

		delete[] vertices;
		delete[] normals;
		delete[] texcoords;
		delete[] indices;
		delete[] matid;

		return true;
	}

	~TriangleMesh() {
		materials.clear();
		triangles.clear();
	}


	bool bruteforceIntersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) {
		// bruteforce ray tracing (for debugging)
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		for (int i = 0; i < triangles.size(); ++i) {
			if (raytraceTriangle(tempMinHit, ray, triangles[i], tMin, tMax)) {
				if (tempMinHit.t < result.t) {
					hit = true;
					result = tempMinHit;
				}
			}
		}

		return hit;
	}

	void createSingleTriangle() {
		triangles.resize(1);
		materials.resize(1);

		triangles[0].idMaterial = 0;

		triangles[0].positions[0] = float3(-0.5f, -0.5f, 0.0f);
		triangles[0].positions[1] = float3(0.5f, -0.5f, 0.0f);
		triangles[0].positions[2] = float3(0.0f, 0.5f, 0.0f);

		const float3 e0 = this->triangles[0].positions[1] - this->triangles[0].positions[0];
		const float3 e1 = this->triangles[0].positions[2] - this->triangles[0].positions[0];
		const float3 n = normalize(cross(e0, e1));

		triangles[0].normals[0] = n;
		triangles[0].normals[1] = n;
		triangles[0].normals[2] = n;

		triangles[0].texcoords[0] = float2(0.0f, 0.0f);
		triangles[0].texcoords[1] = float2(0.0f, 1.0f);
		triangles[0].texcoords[2] = float2(1.0f, 0.0f);
	}


private:
	// === you do not need to modify the followings in this class ===
	void loadTexture(const char* fname, const int i) {
		int comp;
		materials[i].texture = stbi_load(fname, &materials[i].textureWidth, &materials[i].textureHeight, &comp, 3);
		if (!materials[i].texture) {
			std::cerr << "Unable to load texture: " << fname << std::endl;
			return;
		}
	}

	std::string GetBaseDir(const std::string& filepath) {
		if (filepath.find_last_of("/\\") != std::string::npos) return filepath.substr(0, filepath.find_last_of("/\\"));
		return "";
	}
	std::string base_dir;

	void LoadMTL(const std::string fileName) {
		FILE* fp = fopen(fileName.c_str(), "r");

		Material mtl;
		mtl.texture = nullptr;
		char line[81];
		while (fgets(line, 80, fp) != nullptr) {
			float r, g, b, s;
			std::string lineStr;
			lineStr = line;
			int i = int(materials.size());

			if (lineStr.compare(0, 6, "newmtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				mtl.name = lineStr;
				mtl.isTextured = false;
			} else if (lineStr.compare(0, 2, "Ka", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ka = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Kd", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Kd = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ks", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ks = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ns", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f\n", &s);
				mtl.Ns = s;
				mtl.texture = nullptr;
				materials.push_back(mtl);
			} else if (lineStr.compare(0, 6, "map_Kd", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				materials[i - 1].isTextured = true;
				loadTexture((base_dir + lineStr).c_str(), i - 1);
			}
		}

		fclose(fp);
	}

	void ParseOBJ(const char* fileName, int& nVertices, float** vertices, float** normals, float** texcoords, int& nIndices, int** indices, int** materialids) {
		// local function in C++...
		struct {
			void operator()(char* word, int* vindex, int* tindex, int* nindex) {
				const char* null = " ";
				char* ptr;
				const char* tp;
				const char* np;

				// by default, the texture and normal pointers are set to the null string
				tp = null;
				np = null;

				// replace slashes with null characters and cause tp and np to point
				// to character immediately following the first or second slash
				for (ptr = word; *ptr != '\0'; ptr++) {
					if (*ptr == '/') {
						if (tp == null) {
							tp = ptr + 1;
						} else {
							np = ptr + 1;
						}

						*ptr = '\0';
					}
				}

				*vindex = atoi(word);
				*tindex = atoi(tp);
				*nindex = atoi(np);
			}
		} get_indices;

		base_dir = GetBaseDir(fileName);
		#ifdef _WIN32
			base_dir += "\\";
		#else
			base_dir += "/";
		#endif

		FILE* fp = fopen(fileName, "r");
		int nv = 0, nn = 0, nf = 0, nt = 0;
		char line[81];
		if (!fp) {
			printf("Cannot open \"%s\" for reading\n", fileName);
			return;
		}

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (lineStr.compare(0, 6, "mtllib", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				LoadMTL(base_dir + lineStr);
			}

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					nn++;
				} else if (line[1] == 't') {
					nt++;
				} else {
					nv++;
				}
			} else if (line[0] == 'f') {
				nf++;
			}
		}
		fseek(fp, 0, 0);

		float* n = new float[3 * (nn > nf ? nn : nf)];
		float* v = new float[3 * nv];
		float* t = new float[2 * nt];

		int* vInd = new int[3 * nf];
		int* nInd = new int[3 * nf];
		int* tInd = new int[3 * nf];
		int* mInd = new int[nf];

		int nvertices = 0;
		int nnormals = 0;
		int ntexcoords = 0;
		int nindices = 0;
		int ntriangles = 0;
		bool noNormals = false;
		bool noTexCoords = false;
		bool noMaterials = true;
		int cmaterial = 0;

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					float x, y, z;
					sscanf(&line[2], "%f %f %f\n", &x, &y, &z);
					float l = sqrt(x * x + y * y + z * z);
					x = x / l;
					y = y / l;
					z = z / l;
					n[nnormals] = x;
					nnormals++;
					n[nnormals] = y;
					nnormals++;
					n[nnormals] = z;
					nnormals++;
				} else if (line[1] == 't') {
					float u, v;
					sscanf(&line[2], "%f %f\n", &u, &v);
					t[ntexcoords] = u;
					ntexcoords++;
					t[ntexcoords] = v;
					ntexcoords++;
				} else {
					float x, y, z;
					sscanf(&line[1], "%f %f %f\n", &x, &y, &z);
					v[nvertices] = x;
					nvertices++;
					v[nvertices] = y;
					nvertices++;
					v[nvertices] = z;
					nvertices++;
				}
			}
			if (lineStr.compare(0, 6, "usemtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				if (materials.size() != 0) {
					for (unsigned int i = 0; i < materials.size(); i++) {
						if (lineStr.compare(materials[i].name) == 0) {
							cmaterial = i;
							noMaterials = false;
							break;
						}
					}
				}

			} else if (line[0] == 'f') {
				char s1[32], s2[32], s3[32];
				int vI, tI, nI;
				sscanf(&line[1], "%s %s %s\n", s1, s2, s3);

				mInd[ntriangles] = cmaterial;

				// indices for first vertex
				get_indices(s1, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for second vertex
				get_indices(s2, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for third vertex
				get_indices(s3, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				ntriangles++;
			}
		}

		*vertices = new float[ntriangles * 9];
		if (!noNormals) {
			*normals = new float[ntriangles * 9];
		} else {
			*normals = 0;
		}

		if (!noTexCoords) {
			*texcoords = new float[ntriangles * 6];
		} else {
			*texcoords = 0;
		}

		if (!noMaterials) {
			*materialids = new int[ntriangles];
		} else {
			*materialids = 0;
		}

		*indices = new int[ntriangles * 3];
		nVertices = ntriangles * 3;
		nIndices = ntriangles * 3;

		for (int i = 0; i < ntriangles; i++) {
			if (!noMaterials) {
				(*materialids)[i] = mInd[i];
			}

			(*indices)[3 * i] = 3 * i;
			(*indices)[3 * i + 1] = 3 * i + 1;
			(*indices)[3 * i + 2] = 3 * i + 2;

			(*vertices)[9 * i] = v[3 * vInd[3 * i]];
			(*vertices)[9 * i + 1] = v[3 * vInd[3 * i] + 1];
			(*vertices)[9 * i + 2] = v[3 * vInd[3 * i] + 2];

			(*vertices)[9 * i + 3] = v[3 * vInd[3 * i + 1]];
			(*vertices)[9 * i + 4] = v[3 * vInd[3 * i + 1] + 1];
			(*vertices)[9 * i + 5] = v[3 * vInd[3 * i + 1] + 2];

			(*vertices)[9 * i + 6] = v[3 * vInd[3 * i + 2]];
			(*vertices)[9 * i + 7] = v[3 * vInd[3 * i + 2] + 1];
			(*vertices)[9 * i + 8] = v[3 * vInd[3 * i + 2] + 2];

			if (!noNormals) {
				(*normals)[9 * i] = n[3 * nInd[3 * i]];
				(*normals)[9 * i + 1] = n[3 * nInd[3 * i] + 1];
				(*normals)[9 * i + 2] = n[3 * nInd[3 * i] + 2];

				(*normals)[9 * i + 3] = n[3 * nInd[3 * i + 1]];
				(*normals)[9 * i + 4] = n[3 * nInd[3 * i + 1] + 1];
				(*normals)[9 * i + 5] = n[3 * nInd[3 * i + 1] + 2];

				(*normals)[9 * i + 6] = n[3 * nInd[3 * i + 2]];
				(*normals)[9 * i + 7] = n[3 * nInd[3 * i + 2] + 1];
				(*normals)[9 * i + 8] = n[3 * nInd[3 * i + 2] + 2];
			}

			if (!noTexCoords) {
				(*texcoords)[6 * i] = t[2 * tInd[3 * i]];
				(*texcoords)[6 * i + 1] = t[2 * tInd[3 * i] + 1];

				(*texcoords)[6 * i + 2] = t[2 * tInd[3 * i + 1]];
				(*texcoords)[6 * i + 3] = t[2 * tInd[3 * i + 1] + 1];

				(*texcoords)[6 * i + 4] = t[2 * tInd[3 * i + 2]];
				(*texcoords)[6 * i + 5] = t[2 * tInd[3 * i + 2] + 1];
			}

		}
		fclose(fp);

		delete[] n;
		delete[] v;
		delete[] t;
		delete[] nInd;
		delete[] vInd;
		delete[] tInd;
		delete[] mInd;
	}
};



// BVH node (for A1 extra)
class BVHNode {
public:
	bool isLeaf;
	int idLeft, idRight;
	int triListNum;
	int* triList;
	AABB bbox;
};


// ====== implement it in A1 extra ======
// fill in the missing parts
class BVH {
public:
	const TriangleMesh* triangleMesh = nullptr;
	BVHNode* node = nullptr;

	const float costBBox = 1.0f;
	const float costTri = 1.0f;

	int leafNum = 0;
	int nodeNum = 0;

	BVH() {}
	void build(const TriangleMesh* mesh);

	bool intersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		// bvh
		if (this->node[0].bbox.intersect(tempMinHit, ray)) {
			hit = traverse(result, ray, 0, tMin, tMax);
		}
		if (result.t != FLT_MAX) hit = true;

		return hit;
	}
	bool traverse(HitInfo& result, const Ray& ray, int node_id, float tMin, float tMax) const;

private:
	void sortAxis(int* obj_index, const char axis, const int li, const int ri) const;
	int splitBVH(int* obj_index, const int obj_num, const AABB& bbox);

};


// sort bounding boxes (in case you want to build SAH-BVH)
void BVH::sortAxis(int* obj_index, const char axis, const int li, const int ri) const {
	int i, j;
	float pivot;
	int temp;

	i = li;
	j = ri;

	pivot = triangleMesh->triangles[obj_index[(li + ri) / 2]].center[axis];

	while (true) {
		while (triangleMesh->triangles[obj_index[i]].center[axis] < pivot) {
			++i;
		}

		while (triangleMesh->triangles[obj_index[j]].center[axis] > pivot) {
			--j;
		}

		if (i >= j) break;

		temp = obj_index[i];
		obj_index[i] = obj_index[j];
		obj_index[j] = temp;

		++i;
		--j;
	}

	if (li < (i - 1)) sortAxis(obj_index, axis, li, i - 1);
	if ((j + 1) < ri) sortAxis(obj_index, axis, j + 1, ri);
}


//#define SAHBVH // use this in once you have SAH-BVH
int BVH::splitBVH(int* obj_index, const int obj_num, const AABB& bbox) {
	// ====== extend it in A1 extra ======
#ifndef SAHBVH
	int bestAxis, bestIndex;
	AABB bboxL, bboxR, bestbboxL, bestbboxR;
	int* sorted_obj_index  = new int[obj_num];

	// split along the largest axis
	bestAxis = bbox.getLargestAxis();

	// sorting along the axis
	this->sortAxis(obj_index, bestAxis, 0, obj_num - 1);
	for (int i = 0; i < obj_num; ++i) {
		sorted_obj_index[i] = obj_index[i];
	}

	// split in the middle
	bestIndex = obj_num / 2 - 1;

	bboxL.reset();
	for (int i = 0; i <= bestIndex; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxL.fit(tri.positions[0]);
		bboxL.fit(tri.positions[1]);
		bboxL.fit(tri.positions[2]);
	}

	bboxR.reset();
	for (int i = bestIndex + 1; i < obj_num; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxR.fit(tri.positions[0]);
		bboxR.fit(tri.positions[1]);
		bboxR.fit(tri.positions[2]);
	}

	bestbboxL = bboxL;
	bestbboxR = bboxR;
#else
	// implelement SAH-BVH here
#endif

	if (obj_num <= 4) {
		delete[] sorted_obj_index;

		this->nodeNum++;
		this->node[this->nodeNum - 1].bbox = bbox;
		this->node[this->nodeNum - 1].isLeaf = true;
		this->node[this->nodeNum - 1].triListNum = obj_num;
		this->node[this->nodeNum - 1].triList = new int[obj_num];
		for (int i = 0; i < obj_num; i++) {
			this->node[this->nodeNum - 1].triList[i] = obj_index[i];
		}
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->leafNum++;

		return temp_id;
	} else {
		// split obj_index into two 
		int* obj_indexL = new int[bestIndex + 1];
		int* obj_indexR = new int[obj_num - (bestIndex + 1)];
		for (int i = 0; i <= bestIndex; ++i) {
			obj_indexL[i] = sorted_obj_index[i];
		}
		for (int i = bestIndex + 1; i < obj_num; ++i) {
			obj_indexR[i - (bestIndex + 1)] = sorted_obj_index[i];
		}
		delete[] sorted_obj_index;
		int obj_numL = bestIndex + 1;
		int obj_numR = obj_num - (bestIndex + 1);

		// recursive call to build a tree
		this->nodeNum++;
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->node[temp_id].bbox = bbox;
		this->node[temp_id].isLeaf = false;
		this->node[temp_id].idLeft = splitBVH(obj_indexL, obj_numL, bestbboxL);
		this->node[temp_id].idRight = splitBVH(obj_indexR, obj_numR, bestbboxR);

		delete[] obj_indexL;
		delete[] obj_indexR;

		return temp_id;
	}
}


// you may keep this part as-is
void BVH::build(const TriangleMesh* mesh) {
	triangleMesh = mesh;

	// construct the bounding volume hierarchy
	const int obj_num = (int)(triangleMesh->triangles.size());
	int* obj_index = new int[obj_num];
	for (int i = 0; i < obj_num; ++i) {
		obj_index[i] = i;
	}
	this->nodeNum = 0;
	this->node = new BVHNode[obj_num * 2];
	this->leafNum = 0;

	// calculate a scene bounding box
	AABB bbox;
	for (int i = 0; i < obj_num; i++) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];

		bbox.fit(tri.positions[0]);
		bbox.fit(tri.positions[1]);
		bbox.fit(tri.positions[2]);
	}

	// ---------- buliding BVH ----------
	printf("Building BVH...\n");
	splitBVH(obj_index, obj_num, bbox);
	printf("Done.\n");

	delete[] obj_index;
}


// you may keep this part as-is
bool BVH::traverse(HitInfo& minHit, const Ray& ray, int node_id, float tMin, float tMax) const {
	bool hit = false;
	HitInfo tempMinHit, tempMinHitL, tempMinHitR;
	bool hit1, hit2;

	if (this->node[node_id].isLeaf) {
		for (int i = 0; i < (this->node[node_id].triListNum); ++i) {
			if (triangleMesh->raytraceTriangle(tempMinHit, ray, triangleMesh->triangles[this->node[node_id].triList[i]], tMin, tMax)) {
				hit = true;
				if (tempMinHit.t < minHit.t) minHit = tempMinHit;
			}
		}
	} else {
		hit1 = this->node[this->node[node_id].idLeft].bbox.intersect(tempMinHitL, ray);
		hit2 = this->node[this->node[node_id].idRight].bbox.intersect(tempMinHitR, ray);

		hit1 = hit1 && (tempMinHitL.t < minHit.t);
		hit2 = hit2 && (tempMinHitR.t < minHit.t);

		if (hit1 && hit2) {
			if (tempMinHitL.t < tempMinHitR.t) {
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
			} else {
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
			}
		} else if (hit1) {
			hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
		} else if (hit2) {
			hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
		}
	}

	return hit;
}

class VolumetricEffect {
public:
	float density;
	float3 scattering;
	float3 absorption;
	float phase_g;        // phase function asymmetry parameter

	VolumetricEffect(float d, float3 s, float3 a, float g)
		: density(d), scattering(s), absorption(a), phase_g(g) {}

	// henyey-greenstein phase function - used to check amount of incoming light scattered along the direction of ray
	float phase(float cosTheta) const {
		float g = phase_g;
		float den = 1.0f + g * g + 2.0f * g * cosTheta;
		return (1.0f - g * g) / (4.0f * PI * den * sqrt(den));
	}
};

// scene definition
class Scene {

public:
	std::vector<TriangleMesh*> objects;
	std::vector<PointLightSource*> pointLightSources;
	std::vector<BVH> bvhs;
	std::vector<VolumetricEffect*> volumes;
	int DEPTH_MAX = 3;

	float rng = PCG32::rand();
	std::uniform_real_distribution<float> distribution{0, 1};


	void addObject(TriangleMesh* pObj) {
		objects.push_back(pObj);
	}
	void addLight(PointLightSource* pObj) {
		pointLightSources.push_back(pObj);
	}

	void preCalc() {
		bvhs.resize(objects.size());
		for (int i = 0; i < objects.size(); i++) {
			objects[i]->preCalc();
			bvhs[i].build(objects[i]);
		}
	}


    // volume scattering
    float3 evaluateVolume(const Ray& ray, float tMin, float tMax) const {
        float3 accumTransmittance(1.0f);
        float3 accumulatedLight(0.0f);

        const int numSamples = 10;
        float stepSize = (tMax - tMin) / float(numSamples);

        for (int i = 0; i < numSamples; i++) {
            float t = tMin + (i + PCG32::rand()) * stepSize;
            float3 samplePos = ray.o + ray.d * t;

            for (const auto* volume : volumes) {
                float3 extinction = volume->absorption + volume->scattering; // extinction = loss of radiance from absorption and out-scattering
                float3 transmittance = exp(-extinction * stepSize * volume->density); // beer lambert law = e^(-extinction * stepSize * density)

                for (const auto* light : pointLightSources) {
                    float3 lightDir = normalize(light->position - samplePos);
                    float lightDist = length(light->position - samplePos);

                   Ray shadowRay(samplePos, lightDir);
                    HitInfo shadowHit;
                    bool blocked = intersect(shadowHit, shadowRay, 0.001f, lightDist);

                    if (!blocked) {
                        float cosTheta = dot(ray.d, lightDir);
                        float phase = volume->phase(cosTheta);
                        float3 lightContrib = light->wattage / (4.0f * PI * lightDist * lightDist);

                        accumulatedLight += accumTransmittance * volume->scattering * phase *
                                          lightContrib * stepSize * volume->density;
                    }
                }

                accumTransmittance *= transmittance;
            }
        }

        return accumulatedLight;
    }



	// ray-scene intersection
	bool intersect(HitInfo& minHit, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		minHit.t = FLT_MAX;

		for (int i = 0, i_n = (int)objects.size(); i < i_n; i++) {
			//if (objects[i]->bruteforceIntersect(tempMinHit, ray, tMin, tMax)) { // for debugging
			if (bvhs[i].intersect(tempMinHit, ray, tMin, tMax)) {
				if (tempMinHit.t < minHit.t) {
					hit = true;
					minHit = tempMinHit;
				}
			}
		}
		return hit;
	}

	// eye ray generation (given to you for A1)
	Ray eyeRay(int x, int y) const {
		// compute the camera coordinate system 
		const float3 wDir = normalize(float3(-globalViewDir));
		const float3 uDir = normalize(cross(globalUp, wDir));
		const float3 vDir = cross(wDir, uDir);

		// compute the pixel location in the world coordinate system using the camera coordinate system
		// trace a ray through the center of each pixel
		const float imPlaneUPos = (x + 0.5f) / float(globalWidth) - 0.5f;
		const float imPlaneVPos = (y + 0.5f) / float(globalHeight) - 0.5f;

		const float3 pixelPos = globalEye + float(globalAspectRatio * globalFilmSize * imPlaneUPos) * uDir + float(globalFilmSize * imPlaneVPos) * vDir - globalDistanceToFilm * wDir;

		return Ray(globalEye, normalize(pixelPos - globalEye));
	}


	float3 trace(int depth, const Ray &ray) {
		HitInfo hitInfo;
		Environment environment;
    	float3 colour(0.0f);

		if (intersect(hitInfo, ray)) {


			const Material* mat = hitInfo.material;

			if (mat->emission > float3(0.0f)) // light source
			{
				return mat->emission;
			}

			float3 normal = hitInfo.N;
			float3 viewDir = -ray.d;
			float alpha = pow(mat->roughness, 2.0f);
			float3 f0 = lerp(float3(0.04f), mat->albedo, mat->metallic)
;

			for (const auto* light : pointLightSources) { // direct lighting
				// direction to light
				float3 lightDir = normalize(light->position - hitInfo.P);
				float lightDist = length(light->position - hitInfo.P);

				float3 h = normalize(lightDir + viewDir); // half vector between light and view direction

				Ray shadowRay(hitInfo.P, lightDir);
				HitInfo shadowHit;
				bool inShadow = intersect(shadowHit, shadowRay, 0.001f, lightDist);

				if (!inShadow) { // outgoing ray = integral over brdf * Li * cos(theta)
					// basic brdf
					/*float cosTheta = fmax(0.0f, dot(hitInfo.N, lightDir));
					float3 brdf = mat->Kd;
					float3 Li = light->wattage / (4.0f * PI * lightDist * lightDist);
					colour += brdf * Li * cosTheta ;*/

					// cook torrance BRDF
					float NoL = dot(normal, lightDir);
					float NoV = dot(normal, viewDir);
					float NoH = dot(normal, h);
					float VoH = dot(viewDir, h);

					float3 f = mat->fresnelSchlick(VoH, f0);
					float d = mat->ggx(normal, h, alpha);
					float g = mat->smithG(normal, viewDir, lightDir, alpha);
					float3 specular = f * d * g / (4.0f * NoL * NoV + 0.001f);

					float3 diffuse = (mat->Kd * (1 - mat->metallic) * (1 - f)) / PI;

					float3 brdf = diffuse + specular;
					float3 Li = light->wattage / (4.0f * PI * lightDist * lightDist); // inverse square law
					float cosTheta = fmax(0.0f, NoL);

					colour += brdf * Li * cosTheta; // rendering equation*/

				}
			}

			// indirect lighting
			if (depth < DEPTH_MAX) {
				float r1 = PCG32::rand();
				float r2 = PCG32::rand();

            float probabilitySpecular = 0.5f;

            float3 bounceDir;
			float3 w = normal;
			float3 u = normalize(cross(fabs(w.x) > 0.1f ? float3(0, 1, 0) : float3(1, 0, 0), w));
			float3 v = cross(w, u);
				float phi = 2.0f * PI * r1;

            if (PCG32::rand() < probabilitySpecular) {
				// ggx importance sampling for specular surface
            	// cosTheta = sqrt((1 - epsilon)/ epsilon * (alpha^2 - 1) + 1
                float cosTheta = sqrt((1.0f - r2) / (1.0f + (pow(alpha, 2.0f) - 1.0f) * r2));
                float sinTheta = sqrt(1.0f - cosTheta * cosTheta); // just using sin^2 + cos^2 = 1

            	// spherical to cartesian coords
                bounceDir = normalize(
                    u * cos(phi) * sinTheta +
                    v * sin(phi) * sinTheta +
                    w * cosTheta
                );
            } else {
            	// cosine sampling for diffuse surfaces
                float cosTheta = sqrt(1.0f - r2);
                float sinTheta = sqrt(r2);

                bounceDir = normalize(
                    u * cos(phi) * sinTheta +
                    v * sin(phi) * sinTheta +
                    w * cosTheta
                );
            }

            Ray bounceRay(hitInfo.P, bounceDir);
            float3 indirectLight = trace(depth + 1, bounceRay);

            // BRDF for indirect lighting
            float3 H = normalize(bounceDir + viewDir);
            float NoL = fmax(dot(normal, bounceDir), 0.0f);
				float NoV = fmax(dot(normal, viewDir), 0.0f);

            float D = mat->ggx(normal, H, alpha);
            float G = mat->smithG(normal, viewDir, bounceDir, alpha);
            float3 F = mat->fresnelSchlick(fmax(dot(viewDir, H), 0.0f), f0);
				float3 specular = F * D * G / (4.0f * NoL * NoV + 0.001f);
				float3 diffuse = (mat->Kd * (1 - mat->metallic) * (1 - F)) / PI;


            float3 brdf = diffuse + specular;

            colour += indirectLight * brdf * NoL * 2.0f * PI;
				colour += evaluateVolume(ray, 0.0f, hitInfo.t);

        }

			return colour;


		}
		return environment.blurryEnv(ray.d);

	}

	// want to approximate the result of integral of hemsphere of directions above the point P using monte carlo integration
	// pick N random directions over hemisphere and find average amount of light emitted


	void PathTracer() {
		FrameBuffer.clear();


		// rendering equation: outgoing radiance = emission + integral over all directions of incoming radiance * BRDF
		for (int i = 0; i < globalWidth; i++)
		{
			for (int j = 0; j < globalHeight; j++)
			{
				float3 colour = float3(0.0f);
				for (int k = 0; k < sampleCount; k++)
				{
					Ray ray = eyeRay(i, j);
					colour += trace(0, ray);

				}
				colour /= float(sampleCount);
				colour = float3(sqrt(colour.x), sqrt(colour.y), sqrt(colour.z)); // gamma correction of 0.5 (without this the output looks darker)

				FrameBuffer.pixel(i, j) = colour;
			}
			// show intermediate process
			if (globalShowRaytraceProgress) {
				constexpr int scanlineNum = 64;
				if ((i % scanlineNum) == (scanlineNum - 1)) {
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0]);
					glRecti(1, 1, -1, -1);
					glfwSwapBuffers(globalGLFWindow);
					printf("Rendering Progress: %.3f%%\r", i / float(globalHeight - 1) * 100.0f);
					fflush(stdout);
				}
			}
		}

	}

	void Raytrace() const {
		FrameBuffer.clear();

		// loop over all pixels in the image
		for (int j = 0; j < globalHeight; ++j) {
			for (int i = 0; i < globalWidth; ++i) {
				const Ray ray = eyeRay(i, j);
				HitInfo hitInfo;
				if (intersect(hitInfo, ray)) {
					FrameBuffer.pixel(i, j) = shade(hitInfo, -ray.d);
				} else {
					FrameBuffer.pixel(i, j) = float3(0.0f);
				}
			}

			// show intermediate process
			if (globalShowRaytraceProgress) {
				constexpr int scanlineNum = 64;
				if ((j % scanlineNum) == (scanlineNum - 1)) {
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0]);
					glRecti(1, 1, -1, -1);
					glfwSwapBuffers(globalGLFWindow);
					printf("Rendering Progress: %.3f%%\r", j / float(globalHeight - 1) * 100.0f);
					fflush(stdout);
				}
			}
		}
	}


};
static Scene globalScene;

Image envMap;

float3 env(const float3& dir) {
	envMap.load("./media/test.png"); // image

	float u = (atan2f(dir.z, dir.x) + PI) / (2.0f * PI);
	float v = acosf(dir.y) / PI;

	int x = int(u * envMap.width);
	int y = int(v * envMap.height);

	return envMap.pixels[y * envMap.width + x];
}


// ====== implement it in A1 ======
// fill in the missing parts
static float3 shade(const HitInfo& hit, const float3& viewDir, const int level) {
	if (hit.material->type == MAT_LAMBERTIAN) {
		// you may want to add shadow ray tracing here in A1
		float3 L = float3(0.0f);
		float3 brdf, irradiance;

		// loop over all of the point light sources
		for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
			float3 l = globalScene.pointLightSources[i]->position - hit.P;

			// the inverse-squared falloff
			const float falloff = length2(l);

			// normalize the light direction
			l /= sqrtf(falloff);

			Ray shadow; // creating shadow ray
			shadow.o = hit.P + l*1e-4f;
			shadow.d = l;

			HitInfo shadowHit;
			if (globalScene.intersect(shadowHit, shadow)) {
				if (shadowHit.t < sqrtf(falloff)) {
					continue; // if intersection is too close and in shadow then ignore
				}
			}

			// get the irradiance
			irradiance = float(std::max(0.0f, dot(hit.N, l)) / (4.0 * PI * falloff)) * globalScene.pointLightSources[i]->wattage;
			brdf = hit.material->BRDF(l, viewDir, hit.N);

			if (hit.material->isTextured) {
				brdf *= hit.material->fetchTexture(hit.T);
			}
			 return brdf * PI; //debug output

			L += irradiance * brdf;
		}
		return L;
	} else if (hit.material->type == MAT_METAL) {
		if (level < 5) {
			float3 r = -2 * dot(-viewDir, hit.N)* hit.N - viewDir; // reflected direction
			Ray reflection;
			reflection.o = hit.P + r * 1e-8f;
			reflection.d = r;

			HitInfo reflectionHit;
			if (globalScene.intersect(reflectionHit, reflection)) {
				float3 reflectedSurface = shade(reflectionHit, -r, level + 1); // recursively reflect rays
				return hit.material->Ks * reflectedSurface;
			}
			else return env(reflection.d); // if there are no hits then return the background
		}
		else return float3(0.0f); // if max depth is hit
	} else if (hit.material->type == MAT_GLASS) {
		if (level < 5) {
			float3 normal = hit.N;
			float cosTheta = dot(normal, -viewDir); // cos(theta) = normal dot incidence array
			float airRefraction = 1.0f;
			float glassRefraction = hit.material->eta;

			float eta = cosTheta < 0? airRefraction / glassRefraction : glassRefraction / airRefraction;

			if(cosTheta >= 0) { // if the dot product is positive, then the incident ray is inside the object and is leaving it so normal needs to be flipped to find refraction direction
				normal *= -1;
			} else { // if the dot product is negative, then incident ray is outside object and will enter it, so cosTheta will become positive
				cosTheta *= -1;
			}

			float c = 1 - pow(eta, 2) * (1 - pow(cosTheta, 2)); // (see slide 64 in 5_shading) will use this term to check for total internal reflection

			if (c < 0.0f) { // total internal reflection
				float3 r = -2 * dot(-viewDir, hit.N)* hit.N - viewDir; // reflected direction
				Ray reflection;
				reflection.o = hit.P + r * 1e-6f;
				reflection.d = r;

				HitInfo reflectionHit;
				if (globalScene.intersect(reflectionHit, reflection)) {
					float3 reflectedSurface = shade(reflectionHit, -r, level + 1); // recursively reflect rays
					return reflectedSurface;
				}
				else return env(reflection.d); // if there are no hits then return the background
			} else {
				float3 r = eta * (-1 * viewDir - cosTheta * normal) - sqrtf(c) * normal; // refracted direction
				Ray refraction;
				refraction.o = hit.P + r * 1e-6f;
				refraction.d = r;

				HitInfo refractionHit;
				if (globalScene.intersect(refractionHit, refraction)) {
					float3 refractedSurface = shade(refractionHit, -r, level + 1); // recursively refract rays
					return hit.material->Ka * refractedSurface;
				}
				else return env(refraction.d);
			}
			// return float3(0.0f); // if there are no hits then return the background
		}
		else return float3(0.0f); // if max depth is hit
	} else {
		// something went wrong - make it apparent that it is an error
		return float3(100.0f, 0.0f, 100.0f);
	}

}



// OpenGL initialization (you will not use any OpenGL/Vulkan/DirectX... APIs to render 3D objects!)
// you probably do not need to modify this in A0 to A3.
class OpenGLInit {
public:
	OpenGLInit() {
		// initialize GLFW
		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW." << std::endl;
			exit(-1);
		}

		// create a window
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		globalGLFWindow = glfwCreateWindow(globalWidth, globalHeight, "Welcome to CS488/688!", NULL, NULL);
		if (globalGLFWindow == NULL) {
			std::cerr << "Failed to open GLFW window." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// make OpenGL context for the window
		glfwMakeContextCurrent(globalGLFWindow);

		// initialize GLEW
		glewExperimental = true;
		if (glewInit() != GLEW_OK) {
			std::cerr << "Failed to initialize GLEW." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// set callback functions for events
		glfwSetKeyCallback(globalGLFWindow, keyFunc);
		glfwSetMouseButtonCallback(globalGLFWindow, mouseButtonFunc);
		glfwSetCursorPosCallback(globalGLFWindow, cursorPosFunc);

		// create shader
		FSDraw = glCreateProgram();
		GLuint s = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(s, 1, &PFSDrawSource, 0);
		glCompileShader(s);
		glAttachShader(FSDraw, s);
		glLinkProgram(FSDraw);

		// create texture
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &GLFrameBufferTexture);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, globalWidth, globalHeight, 0, GL_LUMINANCE, GL_FLOAT, 0);

		// initialize some OpenGL state (will not change)
		glDisable(GL_DEPTH_TEST);

		glUseProgram(FSDraw);
		glUniform1i(glGetUniformLocation(FSDraw, "input_tex"), 0);

		GLint dims[4];
		glGetIntegerv(GL_VIEWPORT, dims);
		const float BufInfo[4] = { float(dims[2]), float(dims[3]), 1.0f / float(dims[2]), 1.0f / float(dims[3]) };
		glUniform4fv(glGetUniformLocation(FSDraw, "BufInfo"), 1, BufInfo);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	virtual ~OpenGLInit() {
		glfwTerminate();
	}
};



// main window
// you probably do not need to modify this in A0 to A3.
class CS488Window {
public:
	// put this first to make sure that the glInit's constructor is called before the one for CS488Window
	OpenGLInit glInit;

	CS488Window() {}
	virtual ~CS488Window() {}

	void(*process)() = NULL;

	void start() const {

		globalScene.preCalc();

		// main loop
		while (glfwWindowShouldClose(globalGLFWindow) == GL_FALSE) {
			glfwPollEvents();
			globalViewDir = normalize(globalLookat - globalEye);
			globalRight = normalize(cross(globalViewDir, globalUp));


			if (globalRenderType == RENDER_RASTERIZE) {
				// globalScene.Rasterize();
			} else if (globalRenderType == RENDER_RAYTRACE) {
				 globalScene.Raytrace();
			} else if (globalRenderType == RENDER_PATHTRACE) {
				 globalScene.PathTracer();
			} else if (globalRenderType == RENDER_IMAGE) {
				if (process) process();
			}

			if (globalRecording) {
				unsigned char* buf = new unsigned char[FrameBuffer.width * FrameBuffer.height * 4];
				int k = 0;
				for (int j = FrameBuffer.height - 1; j >= 0; j--) {
					for (int i = 0; i < FrameBuffer.width; i++) {
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).x));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).y));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).z));
						buf[k++] = 255;
					}
				}
				GifWriteFrame(&globalGIFfile, buf, globalWidth, globalHeight, globalGIFdelay);
				delete[] buf;
			}

			// drawing the frame buffer via OpenGL (you don't need to touch this)
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0][0]);
			glRecti(1, 1, -1, -1);
			glfwSwapBuffers(globalGLFWindow);
			globalFrameCount++;
			PCG32::rand();
		}
	}
};





/* failed animation
 *
 in path tracer:
static float lastTime = 0.0f;
		float currentTime = glfwGetTime();
		float deltaTime = currentTime - lastTime;
		lastTime = currentTime;

		// Update wing animation if we have wings
		if (wings) {
			wings->update();
			TriangleMesh* newFrame = &wings->getCurrentFrame();
			updateObject(currentWingFrame, newFrame);
			currentWingFrame = newFrame;


			// Clear buffers when wing frame changes
			if (wings->frameChanged) {
				FrameBuffer.clear();
				AccumulationBuffer.clear();
				sampleCount = 0;
			}
		}

class AnimatedWings {
public:
	std::vector<TriangleMesh> frames;
	int currentFrameIndex = 0;
	float frameTime = 0.0f;
	float frameDuration = 1.0f / 15.0f;  // For 24 fps animation
	float deltaTime = 1.0f;
	bool frameChanged = false;


	void loadAnimation(const std::string& baseFilename, int numFrames) {
		frames.resize(numFrames);
		for(int i = 0; i < numFrames; i++) {
			std::string number = std::to_string(i);
			while (number.length() < 4) number = "0" + number;
			std::string filename = baseFilename + number + ".obj";
			if (!frames[i].load(filename.c_str())) {
				printf("Failed to load wing frame: %s\n", filename.c_str());
			}
		}
	}

	void update() {
		frameChanged = false;
		frameTime += deltaTime;

		if (frameTime >= frameDuration) {
			frameTime -= frameDuration;
			int previousFrame = currentFrameIndex;
			currentFrameIndex = (currentFrameIndex + 1) % frames.size();

			if (previousFrame != currentFrameIndex) {
				frameChanged = true;
			}
		}

	}

	TriangleMesh& getCurrentFrame() {
		return frames[currentFrameIndex];
	}
};

in scene class:
void setWings(AnimatedWings* wings) {
		this->wings = wings;
		if (wings) {
			this->currentWingFrame = &wings->getCurrentFrame();
		}

	}

void updateObject(TriangleMesh* oldObj, TriangleMesh* newObj) {
		for (size_t i = 0; i < objects.size(); i++) {
			if (objects[i] == oldObj) {
				objects[i] = newObj;
				break;
			}
		}
	}



*/