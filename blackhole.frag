#version 330
out vec4 fragColor;
uniform vec2 iResolution;
uniform vec3 camPos;
uniform vec3 camDir;
uniform sampler2D panoTex;
uniform float iTime;

const float SchwarzschildRadius = 1.0;
const int MAX_STEPS = 2024;
const float MAX_DISTANCE = 1e6;
const float DiskRadius = 12.0;
const float DiskSpeed = 3.0;

float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f*f*(3.0-2.0*f);
    float n = mix(mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
                      mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
                  mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
                      mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
    return n;
}

vec3 samplePanorama(vec3 dir) {
    float theta = atan(dir.z, dir.x);
    if (theta < 0.0) theta += 2.0 * 3.14159265359;
    float phi = asin(dir.y);
    vec2 uv = vec2(theta / (2.0 * 3.14159265359), 0.5 - phi / 3.14159265359);
    return texture(panoTex, uv).rgb;
}

vec3 rayDirection(vec2 uv, vec3 forward, vec3 right, vec3 up) {
    return normalize(forward + uv.x * right + uv.y * up);
}

vec3 kerrBending(vec3 pos, vec3 rd, float spin) {
    vec3 toCenter = -pos / length(pos);
    float r = length(pos);
    vec3 spinAxis = vec3(0.0, 1.0, 0.0);
    vec3 tangential = cross(spinAxis, rd);
    float gravBend = 0.1 / (r*r);
    float frameDragFactor = spin * 0.05 / (r*r);
    return normalize(rd + gravBend * toCenter + tangential * frameDragFactor);
}

// Compute disk density factor, with smooth drop-off and adjustable padding beyond Schwarzschild radius
float computeDiskFactor(vec3 pos, float DiskRadius, float schwarzschildRadius, float padding) {
    float radial = length(pos.xz);
    float innerRadius = schwarzschildRadius + padding;
    if (radial <= innerRadius) {
        return 0.0; // No disk material within padded region
    }
    // Smoothstep transition to avoid harsh cutoff near padded region
    float innerSmooth = smoothstep(innerRadius, innerRadius + 1, radial);
    float radialFalloff = exp(-pow(radial / DiskRadius, 2.0));
    float verticalFalloff = exp(-pow(pos.y * 10.0, 2.0));
    return innerSmooth * radialFalloff * verticalFalloff;
}

vec4 tonemap(vec4 color) { return color / (color + 1.0); }

vec3 kelvinToRGB(float T) {
    // Clamp to practical range
    T = clamp(T, 1000.0, 25000.0);

    // Normalize temperature for polynomial fit
    float t = T;
    float t2 = t * t;
    float t3 = t2 * t;

    // Approximation of Planckian locus in CIE 1960 UCS space
    float x;
    if (T <= 4000.0) {
        x = -0.2661239 * (1e9 / t3) - 0.2343580 * (1e6 / t2)
            + 0.8776956 * (1e3 / t) + 0.179910;
    } else {
        x = -3.0258469 * (1e9 / t3) + 2.1070379 * (1e6 / t2)
            + 0.2226347 * (1e3 / t) + 0.240390;
    }

    float y = -3.000 * x * x + 2.870 * x - 0.275;

    // Convert xyY to XYZ, with Y = 1.0
    float Y = 1.0;
    float X = (Y / y) * x;
    float Z = (Y / y) * (1.0 - x - y);

    // Convert XYZ to linear sRGB (D65)
    vec3 rgb;
    rgb.r =  3.2406 * X - 1.5372 * Y - 0.4986 * Z;
    rgb.g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
    rgb.b =  0.0557 * X - 0.2040 * Y + 1.0570 * Z;

    // Clamp and gamma correct
    rgb = max(rgb, vec3(0.0));
    rgb = pow(rgb, vec3(1.0 / 2.2));

    return rgb;
}

// Units: G = M = c = 1. r is Boyer–Lindquist radius (approx via |pos|).
// Spin parameter a in [0, 1).
float safeInv(float x){ return 1.0 / max(1e-6, x); }
float clampBeta(float b){ return clamp(b, -0.999999, 0.999999); }

// Kerr horizons (not used directly in math below but handy for guards)
float rPlus(float a){ return 1.0 + sqrt(max(0.0, 1.0 - a*a)); }

// Prograde/retrograde orbital angular velocity (equatorial circular orbits)
float kerrOmega(float r, float a, bool prograde) {
    float s = prograde ? +1.0 : -1.0;
    return safeInv(pow(r, 1.5) + s * a);
}

// Frame-dragging angular velocity ω. For shader speed, a standard
// far-to-moderate r approximation works well:
float frameDragOmega(float r, float a) {
    // ω ≈ 2 a / r^3 (equatorial, moderate r). Stable and cheap.
    return 2.0 * a * safeInv(r*r*r);
}

// "Lapse-like" gravitational factor at the emitter (equatorial shortcut).
// α_emit ≈ sqrt(1 - 2/r + a^2/r^2) = sqrt(Δ)/r  in equatorial plane.
// For a distant/static camera, α_cam ≈ 1.
float kerrGravFactor(float r, float a, float r_cam) {
    float emit = sqrt(max(1e-6, 1.0 - 2.0/r + (a*a)/(r*r)));
    float cam  = sqrt(max(1e-6, 1.0 - 2.0/r_cam + (a*a)/(r_cam*r_cam)));
    return emit / cam; // ≈ emit if r_cam >> 1
}

// Tangential (azimuthal) unit vector in the disk plane from position.
vec3 azimuthalTangent(vec3 pos) {
    // In x–z plane: +phi points roughly ( -z, 0, +x )
    vec2 xz = pos.xz;
    if (dot(xz, xz) < 1e-12) return vec3(0.0); // avoid NaN at r~0
    vec2 t = normalize(vec2(-xz.y, xz.x));
    return normalize(vec3(t.x, 0.0, t.y));
}

// Approximate linear speed (β) of disk matter relative to LNRF/ZAMO.
// β ≈ (Ω - ω) * A/(r^2 sqrt(Δ))  collapses to the shader-friendly form below.
// This compact surrogate behaves well and captures the main trends.
float betaLNRF(float r, float a, float Omega) {
    float denom = sqrt(max(1e-6, 1.0 - 2.0/r + (a*a)/(r*r))); // ~ lapse
    float beta  = (r * Omega - a) * safeInv(max(1e-6, r) * denom);
    return clampBeta(beta);
}

// Special-relativistic Doppler factor for local emitter velocity v (β) and
// photon direction toward the camera n (unit).
float dopplerSR(float beta, float costh) {
    float gamma = 1.0 / sqrt(1.0 - beta*beta);
    return 1.0 / (gamma * (1.0 - beta * costh)); // ν_obs = D * ν_emit
}

vec3 ACESFilm(vec3 x) {
    // ACES approximation by Narkowicz
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x*(a*x + b)) / (x*(c*x + d) + e), 0.0, 1.0);
}

// Uniforms you likely already have or can add
const float a = -0.9;              // Kerr spin 0..0.999 (positive = prograde)
const float beamingPower = 16.0;   // 3.0 = physical; >3 for artistic boost (e.g., 4.5)

void main(){
    vec2 uv = (gl_FragCoord.xy / iResolution) * 2.0 - 1.0;
    uv.x *= iResolution.x / iResolution.y;

    vec3 forward = normalize(camDir);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, forward);
    vec3 rd = rayDirection(uv, forward, right, up);

    vec3 pos = camPos;
    vec3 col = vec3(0.0);
    float alpha = 1.0;
    bool insideHorizon = false;

    // Normalized path accumulation (log-space)
    float log_g_path = 0.0;

    float r_cam = length(camPos);
    bool prograde = (a >= 0.0);
    float a_abs = abs(a);

    float ds = 0.05;            // step length
    float pathLength = 0.0;  // accumulate along the march
    float D_prev = 1.0;

    for(int i = 0; i < MAX_STEPS; i++) {
        float r = length(pos);
        if (r < SchwarzschildRadius) { insideHorizon = true; break; }

        // Kerr bending
        rd = kerrBending(pos, rd, -a);
        vec3 pos_prev = pos;
        pos += rd * ds;
        pathLength += ds;

        if(length(pos - camPos) > MAX_DISTANCE) break;

        // Mid-step radius for gravitational factor
        float r_mid = 0.5 * (length(pos_prev) + length(pos));
        float g_step = kerrGravFactor(r_mid, a_abs, r_cam);

        // Step-length normalized accumulation (log-space)
        log_g_path += log(max(g_step, 1e-6)) * (ds / pathLength);
        float g_path = exp(log_g_path);  // normalized accumulation

        // Disk factor / density
        float disk_factor = computeDiskFactor(pos, DiskRadius, SchwarzschildRadius, 2);
        if(disk_factor <= 0.001) continue;

        // --- Kerr orbital motion ---
        float Omega = kerrOmega(r, a_abs, prograde) * 1.0;
        float beta  = betaLNRF(r, a_abs, Omega) * 1.0;

        vec3 t_hat = normalize(azimuthalTangent(pos) + 1e-6) * sign(-a);
        vec3 v_hat = t_hat * (prograde ? +1.0 : -1.0);

        // Photon direction
        vec3 n_to_cam = normalize(-rd);
        float costh = clamp(dot(v_hat, n_to_cam), -0.999, 0.999);

        // Doppler factor
        float D = dopplerSR(beta, costh);

        // Total energy shift
        float g_total = g_path * D;

        // Beaming (Doppler dominates)
        float beaming = pow(D, beamingPower) * pow(max(g_path, 1e-6), 1.0);

        // Disk noise / rotation
        vec3 rotatedPos = pos;
        float angle = DiskSpeed * iTime / (length(pos.xz) * 0.5 +0.1);
        float cosA = cos(-angle);
        float sinA = sin(-angle);
        rotatedPos.xz = mat2(cosA, -sinA, sinA, cosA) * pos.xz;

        float density = (noise(rotatedPos * .35) * 0.5 + 0.5) * disk_factor;

        // Spectral shift
        vec3 emitColor = kelvinToRGB(1000.0 * g_total)* 1;

        // Accumulate emission
        vec3 dI = emitColor * density * alpha * beaming;
        col += dI;

        alpha *= (1.0 - density);
        if(alpha < 0.01) break;
    }



    vec3 panoCol = insideHorizon ? vec3(0.0) : samplePanorama(rd) * 0.01;
    vec4 c = vec4(mix(panoCol, col, 1 - alpha), 1.0);

    fragColor.xyz = ACESFilm(c.xyz);
}