/* Earth's Jet Streams - interactive WebGL2 viewer of NOAA GFS 250 mb winds.
   No dependencies. Particles are advected on the CPU (lon/lat, Euler, time-interpolated field),
   their last K positions live in a float texture ring buffer and are drawn as additive GL lines
   projected in the vertex shader (orthographic globe <-> equirectangular morph), then bloomed
   and tone-mapped to match the rendered video. */
(() => {
'use strict';
const DEG = Math.PI / 180, EARTH_R = 6.371e6;
const K = 10;                          // trail history slots per particle
const TEXW = 1024, MAXP = 131072, ROWS = MAXP / TEXW;
const NLON = 720, NLAT = 361, NG = NLON * NLAT;
const HOURS_PER_SEC = 168 / 60;        // 1x: the week in one minute
const SPEEDS = [0.25, 0.5, 1, 2, 4];
const STOPS = [0, 12, 24, 36, 48, 62, 76, 95];
const COLS = [[.10,.20,.62],[.13,.42,.92],[.08,.74,.92],[.28,.86,.46],[.86,.90,.26],[1,.62,.14],[1,.42,.18],[1,.94,.86]];
const LABELS = [['PACIFIC OCEAN',-150,8],['ATLANTIC OCEAN',-38,14],['INDIAN OCEAN',72,-12],['NORTH AMERICA',-104,40],
  ['SOUTH AMERICA',-62,-12],['AFRICA',18,4],['EUROPE',12,50],['ASIA',92,44],['AUSTRALIA',128,-25],['ANTARCTICA',40,-78]];
const MON = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
const COMPASS = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];

const qs = new URLSearchParams(location.search);
const $ = id => document.getElementById(id);
const NOSTYLE = { setProperty() {} };
const sty = id => { const e = $(id); return e ? e.style : NOSTYLE; };   // null-safe element.style
class DataError extends Error {}   // network / decode failures of the wind data only
const clamp = (x, a, b) => x < a ? a : x > b ? b : x;
function ramp(s) {
  if (s <= STOPS[0]) return COLS[0];
  for (let i = 1; i < STOPS.length; i++) if (s <= STOPS[i]) {
    const t = (s - STOPS[i-1]) / (STOPS[i] - STOPS[i-1]), a = COLS[i-1], b = COLS[i];
    return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t];
  }
  return COLS[COLS.length-1];
}
const css = c => `rgb(${c.map(v => Math.round(v*255)).join(',')})`;
function fatal(msg) { const d = document.createElement('div'); d.className = 'fatal'; d.innerHTML = msg; document.body.appendChild(d); }

// ------------------------------------------------------------------ GL setup
const canvas = $('gl');
const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, depth: false, stencil: false,
  premultipliedAlpha: false, powerPreference: 'high-performance' });
if (!gl) { fatal('This visualization needs <b>WebGL2</b>, which this browser does not provide.'); return; }
const hasCBF = !!gl.getExtension('EXT_color_buffer_float');
gl.getExtension('EXT_color_buffer_half_float');

const PROJ = `
uniform mat3 uRot; uniform vec2 uRes, uGC, uFC; uniform float uGR, uFK, uLon0, uFLat, uMorph;
vec3 proj(vec2 ll){
  float lo = radians(ll.x), la = radians(ll.y), cl = cos(la);
  vec3 v = uRot * vec3(cl*cos(lo), cl*sin(lo), sin(la));
  vec2 g = uGC + uGR * v.xy;
  float dl = mod(ll.x - uLon0 + 540.0, 360.0) - 180.0;
  vec2 f = uFC + uFK * radians(vec2(dl, ll.y - uFLat));
  float gf = v.z > 0.0 ? sqrt(v.z) : -1.0;
  return vec3(mix(g, f, uMorph), mix(gf, 1.0, uMorph));
}`;
const RAMP = `
vec3 ramp(float s){
  const float S[8] = float[8](0.,12.,24.,36.,48.,62.,76.,95.);
  const vec3 C[8] = vec3[8](vec3(.10,.20,.62),vec3(.13,.42,.92),vec3(.08,.74,.92),vec3(.28,.86,.46),vec3(.86,.90,.26),vec3(1.,.62,.14),vec3(1.,.42,.18),vec3(1.,.94,.86));
  vec3 c = C[0];
  for (int i = 1; i < 8; i++) c = mix(c, C[i], clamp((s - S[i-1]) / (S[i] - S[i-1]), 0., 1.));
  return c;
}`;
const VS_PART = `#version 300 es
precision highp float; precision highp int;
uniform highp sampler2D uHist; uniform int uHead; uniform float uGain, uMaxSeg;
${PROJ}${RAMP}
out vec3 vCol;
vec4 fetchP(int p, int slot){ return texelFetch(uHist, ivec2(p & 1023, slot * ${ROWS} + (p >> 10)), 0); }
void main(){
  int seg = gl_VertexID >> 1, e = gl_VertexID & 1;
  int p = seg / ${K-1}, j = seg - p * ${K-1};
  vec4 now = fetchP(p, uHead);
  float age = floor(now.w), lf = fract(now.w) * 2.0;
  vec4 a = fetchP(p, (uHead - j + ${K}) % ${K});
  vec4 b = fetchP(p, (uHead - j - 1 + ${2*K}) % ${K});
  vec3 pa = proj(a.xy), pb = proj(b.xy);
  bool ok = float(j + 1) <= age && pa.z > 0.0 && pb.z > 0.0 && distance(pa.xy, pb.xy) < uMaxSeg;
  vec3 me = e == 0 ? pa : pb;
  float spd = e == 0 ? a.z : b.z;
  float hw = pow(max(0.0, 1.0 - (float(j) + float(e)) / ${(K-1).toFixed(1)}), 1.2);
  float ml = abs(e == 0 ? a.y : b.y);
  float polar = 1.0 - uMorph * 0.85 * smoothstep(70.0, 89.0, ml);   // tame equirectangular stretching near the poles
  float w = me.z * lf * hw * uGain * polar;
  vCol = ramp(spd) * (0.30 + 0.70 * pow(clamp(spd / 60.0, 0.0, 1.0), 0.85)) * w;
  gl_Position = ok ? vec4(me.xy / uRes * 2.0 - 1.0, 0.0, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
}`;
const FS_PART = `#version 300 es
precision mediump float; in vec3 vCol; out vec4 o; void main(){ o = vec4(vCol, 1.0); }`;
const VS_COAST = `#version 300 es
precision highp float;
in vec2 aA, aB; uniform float uMaxSeg;
${PROJ}
out float vA;
void main(){
  vec3 pa = proj(aA), pb = proj(aB);
  bool ok = pa.z > 0.0 && pb.z > 0.0 && distance(pa.xy, pb.xy) < uMaxSeg;
  vA = 0.55 + 0.45 * clamp(pa.z, 0.0, 1.0);
  gl_Position = ok ? vec4(pa.xy / uRes * 2.0 - 1.0, 0.0, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
}`;
const FS_COAST = `#version 300 es
precision highp float; in float vA; uniform sampler2D uScene; uniform vec2 uRes; uniform float uAlpha; out vec4 o;
void main(){
  vec3 c = texture(uScene, gl_FragCoord.xy / uRes).rgb;
  float lum = max(c.r, max(c.g, c.b));
  o = vec4(0.40, 0.415, 0.45, vA * uAlpha * (1.0 - clamp(lum * 1.3, 0.0, 1.0)));
}`;
const VS_FULL = `#version 300 es
void main(){ vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;
const FS_DOWN = `#version 300 es
precision mediump float; uniform sampler2D uT; uniform vec2 uTexel; uniform vec2 uOut; out vec4 o;
void main(){ vec2 uv = gl_FragCoord.xy / uOut;
  o = 0.25 * (texture(uT, uv + uTexel * vec2(-1.,-1.)) + texture(uT, uv + uTexel * vec2(1.,-1.)) + texture(uT, uv + uTexel * vec2(-1.,1.)) + texture(uT, uv + uTexel * vec2(1.,1.))); }`;
const FS_BLUR = `#version 300 es
precision mediump float; uniform sampler2D uT; uniform vec2 uDir; uniform vec2 uOut; out vec4 o;
void main(){ vec2 uv = gl_FragCoord.xy / uOut;
  o = 0.2270270 * texture(uT, uv) + 0.3162162 * (texture(uT, uv + uDir * 1.3846154) + texture(uT, uv - uDir * 1.3846154))
    + 0.0702703 * (texture(uT, uv + uDir * 3.2307692) + texture(uT, uv - uDir * 3.2307692)); }`;
const FS_COMP = `#version 300 es
precision highp float;
uniform sampler2D uScene, uBloom; uniform vec2 uRes, uGC; uniform float uGR, uMorph, uDpr, uBloomK, uExp; uniform vec4 uRect;
out vec4 o;
void main(){
  vec2 fc = gl_FragCoord.xy, uv = fc / uRes;
  vec3 rgb = texture(uScene, uv).rgb + uBloomK * texture(uBloom, uv).rgb;
  float r = length(fc - uGC);
  float disk = clamp(uGR - r + 0.5, 0.0, 1.0);
  float d = (r - uGR) / uDpr;
  vec3 bgG = disk * vec3(.022, .025, .034) + exp(-d * d / 4.5) * 0.10 * vec3(.55, .7, 1.)
           + (d > 0.0 ? exp(-d / 16.0) * 0.035 * vec3(.45, .62, 1.) : vec3(0.0));
  vec2 q = min(fc - uRect.xy, uRect.zw - fc);
  float m = clamp(min(q.x, q.y) + 0.5, 0.0, 1.0);
  float sd = min(q.x, q.y) > 0.0 ? min(q.x, q.y) : length(max(-q, 0.0));   // distance to the map outline
  float edge = exp(-pow(sd / uDpr, 2.0) / 1.2);
  vec3 bgF = m * vec3(.022, .025, .034) + edge * 0.07 * vec3(.6, .7, .9);
  vec3 bg = mix(bgG, bgF, uMorph);
  float lum = max(rgb.r, max(rgb.g, rgb.b));
  vec3 c = bg * (1.0 - clamp(lum * 2.0, 0.0, 1.0)) + rgb;
  c = (1.0 - exp(-c * uExp)) / (1.0 - exp(-uExp));
  o = vec4(c, 1.0);
}`;

function program(vs, fs, attribs) {
  const mk = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) + '\n' + src); return s; };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  (attribs || []).forEach((a, i) => gl.bindAttribLocation(p, i, a));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const nm = gl.getActiveUniform(p, i).name.replace(/\[0\]$/, ''); u[nm] = gl.getUniformLocation(p, nm); }
  return { p, u };
}
let P_PART, P_COAST, P_DOWN, P_BLUR, P_COMP;
try {
  P_PART = program(VS_PART, FS_PART); P_COAST = program(VS_COAST, FS_COAST, ['aA', 'aB']);
  P_DOWN = program(VS_FULL, FS_DOWN); P_BLUR = program(VS_FULL, FS_BLUR); P_COMP = program(VS_FULL, FS_COMP);
} catch (e) { console.error(e); fatal('WebGL shader compilation failed on this device.'); return; }

const emptyVAO = gl.createVertexArray();
// history ring buffer texture
const histTex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, histTex);
gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, TEXW, ROWS * K);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
const staging = new Float32Array(MAXP * 4);

function makeTarget(w, h, hdr) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  let ok = false; const fb = gl.createFramebuffer();
  const tryFmt = (ifmt, type) => {
    gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, gl.RGBA, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    return gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  };
  if (hdr) ok = tryFmt(gl.RGBA16F, gl.HALF_FLOAT);
  if (!ok) { gl.bindTexture(gl.TEXTURE_2D, t); tryFmt(gl.RGBA8, gl.UNSIGNED_BYTE); }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { t, fb, w, h, hdr: ok };
}
let RT = null;

// ------------------------------------------------------------------ state
const view = { lon0: -40, lat0: 30, zoomG: 1, zoomF: 1, flatLat: 0, morph: 0, target: 0 };
if (qs.get('lon')) view.lon0 = +qs.get('lon');
if (qs.get('lat')) view.lat0 = clamp(+qs.get('lat'), -85, 85);
const geo = { W: 0, H: 0, dpr: 1, cx: 0, cy: 0, baseR: 1, fcx: 0, fcy: 0, baseK: 1, R: 1, Kf: 1 };
const time = { hours: 0, playing: qs.get('play') !== '0', speedIdx: 2, NT: 0, loaded: [], prefix: 0 };
let pendingT = qs.get('t') ? Math.max(0, +qs.get('t') || 0) : null;   // ?t= applied once that frame is loaded
const inter = { last: -1e9, dragging: false, vx: 0, vy: 0 };
let frames = [], manifest = null, times0 = 0;
const fu = new Float32Array(NG), fv = new Float32Array(NG);
let fieldTf = -1, fieldReady = false;
const COSINV = new Float32Array(NLAT);      // 1/cos(lat) per 0.5 deg row
for (let i = 0; i < NLAT; i++) COSINV[i] = 1 / Math.max(0.03, Math.cos((90 - i * 0.5) * DEG));

// particles
const plon = new Float32Array(MAXP), plat = new Float32Array(MAXP), pspd = new Float32Array(MAXP);
const page = new Int16Array(MAXP), plife = new Int16Array(MAXP);
let nP = 0, quality = qs.get('q') ? +qs.get('q') : 1, head = 0, frameNo = 0;
const adapt = qs.get('noadapt') === null;
const DENS = +(qs.get('dens') || 0.13);
const COARSE = matchMedia('(pointer:coarse)').matches;
const MAX_ACTIVE = +(qs.get('maxp') || (COARSE ? 36864 : 90112));

// ------------------------------------------------------------------ layout
const bottomEl = $('bottom');
function rotMatrix() {
  const L = view.lon0 * DEG, A = view.lat0 * DEG, cL = Math.cos(L), sL = Math.sin(L), cA = Math.cos(A), sA = Math.sin(A);
  // rows: screenX, screenY(up), depth
  return [-sL, cL, 0,  -sA * cL, -sA * sL, cA,  cA * cL, cA * sL, sA];
}
let ROT = rotMatrix();
function layout() {
  const W = innerWidth, H = innerHeight, dpr = Math.min(window.devicePixelRatio || 1, +(qs.get('dpr') || 2));
  const portrait = W < 760 && H > W * 1.15;
  document.body.classList.toggle('portrait', portrait);
  Object.assign(geo, { W, H, dpr });
  const bw = Math.round(W * dpr), bh = Math.round(H * dpr);
  if (canvas.width !== bw || canvas.height !== bh || !RT) {
    canvas.width = bw; canvas.height = bh;
    RT = { scene: makeTarget(bw, bh, hasCBF), a: makeTarget(Math.max(1, bw >> 2), Math.max(1, bh >> 2), hasCBF),
           b: makeTarget(Math.max(1, bw >> 2), Math.max(1, bh >> 2), hasCBF) };
  }
  if (portrait) {
    const top = bottomEl.getBoundingClientRect().top;
    const area = Math.max(160, top - 6);
    geo.baseR = Math.max(60, Math.min(W / 2 - 16, area / 2 - 14));
    geo.cx = W / 2; geo.cy = Math.max(geo.baseR + 16, area / 2 + 4);
    geo.baseK = Math.min(W * 1.6 / (2 * Math.PI), (area - 24) / Math.PI);   // wider than the screen; pan sideways
    geo.fcx = W / 2; geo.fcy = area / 2;
  } else {
    const short = H < 520;
    geo.baseR = Math.min(H * (short ? 0.40 : 0.41), W * 0.34);
    geo.cx = W / 2; geo.cy = short ? H * 0.44 : H * 0.455;
    const availH = short ? H - 90 : bottomEl.getBoundingClientRect().top - 6;
    geo.baseK = Math.min((W - 40) / (2 * Math.PI), (availH - 30) / Math.PI);
    geo.fcx = W / 2; geo.fcy = Math.max(availH / 2, geo.baseK * Math.PI / 2 + 16);
  }
}

// ------------------------------------------------------------------ projection (CSS px, y down)
function projectLL(lon, lat, out) {
  const m = view.morph, lo = lon * DEG, la = lat * DEG, cl = Math.cos(la);
  const x = cl * Math.cos(lo), y = cl * Math.sin(lo), z = Math.sin(la), R = ROT;
  const vx = R[0]*x + R[1]*y + R[2]*z, vy = R[3]*x + R[4]*y + R[5]*z, vz = R[6]*x + R[7]*y + R[8]*z;
  const gx = geo.cx + geo.R * vx, gy = geo.cy - geo.R * vy;
  const dl = ((lon - view.lon0 + 540) % 360 + 360) % 360 - 180;
  const fx = geo.fcx + geo.Kf * dl * DEG, fy = geo.fcy - geo.Kf * (lat - view.flatLat) * DEG;
  out[0] = gx + (fx - gx) * m; out[1] = gy + (fy - gy) * m;
  out[2] = (vz > 0 ? Math.sqrt(vz) : -1) * (1 - m) + m;
  return out;
}
function unproject(sx, sy) {
  if (view.morph < 0.5) {
    const x = (sx - geo.cx) / geo.R, y = (geo.cy - sy) / geo.R, r2 = x * x + y * y;
    if (r2 > 1) return null;
    const z = Math.sqrt(1 - r2), R = ROT;
    const px = R[0]*x + R[3]*y + R[6]*z, py = R[1]*x + R[4]*y + R[7]*z, pz = R[2]*x + R[5]*y + R[8]*z;
    return [Math.atan2(py, px) / DEG, Math.asin(clamp(pz, -1, 1)) / DEG];
  }
  const lat = view.flatLat + (geo.fcy - sy) / geo.Kf / DEG;
  if (lat > 90 || lat < -90) return null;
  let lon = view.lon0 + (sx - geo.fcx) / geo.Kf / DEG;
  if (Math.abs(lon - view.lon0) > 180) return null;
  lon = ((lon + 540) % 360) - 180;
  return [lon, lat];
}
function viewBox() {       // visible part of the globe / map in CSS px
  let x0, x1, y0, y1;
  if (view.morph < 0.5) { x0 = geo.cx - geo.R; x1 = geo.cx + geo.R; y0 = geo.cy - geo.R; y1 = geo.cy + geo.R; }
  else { const hw = geo.Kf * Math.PI, hh = geo.Kf * Math.PI / 2, yc = geo.fcy + geo.Kf * view.flatLat * DEG;
    x0 = geo.fcx - hw; x1 = geo.fcx + hw; y0 = yc - hh; y1 = yc + hh; }
  x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(geo.W, x1); y1 = Math.min(geo.H, y1);
  return [x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0)];
}

// ------------------------------------------------------------------ field
function sampleUV(lon, lat) {
  let fi = (90 - lat) * 2, fj = lon * 2; fj = ((fj % NLON) + NLON) % NLON;
  let i0 = fi | 0; if (i0 > NLAT - 2) i0 = NLAT - 2; if (i0 < 0) i0 = 0;
  const di = fi - i0, j0 = fj | 0, dj = fj - j0, j1 = j0 + 1 === NLON ? 0 : j0 + 1;
  const a = i0 * NLON, b = a + NLON;
  const w00 = (1 - di) * (1 - dj), w01 = (1 - di) * dj, w10 = di * (1 - dj), w11 = di * dj;
  return [fu[a+j0]*w00 + fu[a+j1]*w01 + fu[b+j0]*w10 + fu[b+j1]*w11, fv[a+j0]*w00 + fv[a+j1]*w01 + fv[b+j0]*w10 + fv[b+j1]*w11];
}
function updateField(tf) {
  if (Math.abs(tf - fieldTf) < 1e-5 && fieldReady) return;
  const a = Math.min(Math.floor(tf), time.NT - 2), f = tf - a;
  const A = frames[a], B = frames[a + 1]; if (!A || !B) return;
  const g = 1 - f;
  const su0 = A.su * g, su1 = B.su * f, cu = A.u0 * g + B.u0 * f;
  const sv0 = A.sv * g, sv1 = B.sv * f, cv = A.v0 * g + B.v0 * f;
  const qa = A.qu, qb = B.qu, ra = A.qv, rb = B.qv;
  for (let i = 0; i < NG; i++) { fu[i] = cu + qa[i] * su0 + qb[i] * su1; fv[i] = cv + ra[i] * sv0 + rb[i] * sv1; }
  fieldTf = tf; fieldReady = true;
}

// ------------------------------------------------------------------ particles
const tmp3 = [0, 0, 0];
function spawn(i) {
  const [bx, by, bw, bh] = viewBox();
  let ll = null;
  const flat = view.morph >= 0.5;
  for (let k = 0; k < 32 && !ll; k++) {
    ll = unproject(bx + Math.random() * bw, by + Math.random() * bh);
    // on the flat map keep particle density uniform per unit area of the sphere (not per pixel)
    if (ll && flat && Math.random() > Math.cos(ll[1] * DEG)) ll = null;
  }
  if (!ll) { const z = Math.random() * 2 - 1; ll = [Math.random() * 360 - 180, Math.asin(z) / DEG]; }
  plon[i] = ll[0]; plat[i] = ll[1]; page[i] = 0;
  plife[i] = 60 + (Math.random() * 150) | 0;
}
function targetCount() {
  const [, , bw, bh] = viewBox();
  const area = view.morph < 0.5 ? Math.min(bw * bh, Math.PI * geo.R * geo.R) : bw * bh;
  const z = view.morph < 0.5 ? view.zoomG : view.zoomF;
  const d = DENS * quality * (view.morph < 0.5 ? 1 : 0.5) / Math.sqrt(Math.max(1, z));
  return clamp(Math.round(area * d), 3000, MAX_ACTIVE) & ~1023 || 1024;
}
function setCount(n) {
  if (n > nP) for (let i = nP; i < n; i++) { spawn(i); page[i] = -(Math.random() * 40 | 0); }   // stagger births
  nP = n;
}
function isVisible(lon, lat) {
  projectLL(lon, lat, tmp3);
  return tmp3[2] > 0.05 && tmp3[0] > -20 && tmp3[0] < geo.W + 20 && tmp3[1] > -20 && tmp3[1] < geo.H + 20;
}
function stepParticles(dtAdv) {
  const n = nP, kLat = dtAdv / EARTH_R / DEG, check = frameNo % 3;
  head = (head + 1) % K;
  for (let i = 0; i < n; i++) {
    let lo = plon[i], la = plat[i], a = page[i] + 1;
    // bilinear sample (inlined)
    let fi = (90 - la) * 2, fj = lo * 2; if (fj < 0) fj += NLON;
    let i0 = fi | 0; if (i0 > NLAT - 2) i0 = NLAT - 2; else if (i0 < 0) i0 = 0;
    let j0 = fj | 0; const di = fi - i0, dj = fj - j0; if (j0 >= NLON) j0 -= NLON;
    const j1 = j0 + 1 === NLON ? 0 : j0 + 1, r0 = i0 * NLON, r1 = r0 + NLON;
    const w00 = (1 - di) * (1 - dj), w01 = (1 - di) * dj, w10 = di * (1 - dj), w11 = di * dj;
    const u = fu[r0+j0]*w00 + fu[r0+j1]*w01 + fu[r1+j0]*w10 + fu[r1+j1]*w11;
    const v = fv[r0+j0]*w00 + fv[r0+j1]*w01 + fv[r1+j0]*w10 + fv[r1+j1]*w11;
    const s = Math.sqrt(u * u + v * v);
    if (a > 0) {
      la += v * kLat; lo += u * kLat * COSINV[(fi + 0.5) | 0];
      if (lo >= 180) lo -= 360; else if (lo < -180) lo += 360;
    }
    let dead = a >= plife[i] || la > 88.5 || la < -88.5;
    if (!dead && i % 3 === check && !isVisible(lo, la)) dead = true;
    if (dead) { spawn(i); lo = plon[i]; la = plat[i]; a = 0; }
    plon[i] = lo; plat[i] = la; page[i] = a; pspd[i] = s;
    const ag = a < 0 ? 0 : a;
    const lf = Math.max(0, Math.min(1, ag / 14, (plife[i] - ag) / 14));
    const o = i * 4;
    staging[o] = lo; staging[o+1] = la; staging[o+2] = s; staging[o+3] = Math.min(ag, 1000) + lf * 0.499;
  }
  gl.bindTexture(gl.TEXTURE_2D, histTex);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, head * ROWS, TEXW, Math.ceil(n / TEXW), gl.RGBA, gl.FLOAT, staging, 0);
}

// ------------------------------------------------------------------ coastlines
let coastVAO = null, coastCount = 0;
function buildCoast(j) {
  const s = 1 / j.scale, verts = [];
  for (const L of j.lines) {
    let x = L[0], y = L[1], px = x * s, py = y * s;
    for (let k = 2; k < L.length; k += 2) {
      x += L[k]; y += L[k+1]; const nx = x * s, ny = y * s;
      verts.push(px, py, nx, ny, nx, ny, px, py); px = nx; py = ny;
    }
  }
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
  coastVAO = gl.createVertexArray(); gl.bindVertexArray(coastVAO);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
  gl.bindVertexArray(null);
  coastCount = verts.length / 4;
}

// ------------------------------------------------------------------ render
function setProj(pr) {
  const d = geo.dpr, u = pr.u;
  gl.uniformMatrix3fv(u.uRot, false, [ROT[0], ROT[3], ROT[6], ROT[1], ROT[4], ROT[7], ROT[2], ROT[5], ROT[8]]);
  gl.uniform2f(u.uRes, canvas.width, canvas.height);
  gl.uniform2f(u.uGC, geo.cx * d, (geo.H - geo.cy) * d); gl.uniform1f(u.uGR, geo.R * d);
  gl.uniform2f(u.uFC, geo.fcx * d, (geo.H - geo.fcy) * d); gl.uniform1f(u.uFK, geo.Kf * d);
  gl.uniform1f(u.uLon0, view.lon0); gl.uniform1f(u.uFLat, view.flatLat); gl.uniform1f(u.uMorph, view.morph);
}
function render() {
  const d = geo.dpr, bw = canvas.width, bh = canvas.height;
  // 1. particles -> HDR scene
  gl.bindFramebuffer(gl.FRAMEBUFFER, RT.scene.fb); gl.viewport(0, 0, bw, bh);
  gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
  if (nP && fieldReady) {
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(P_PART.p); setProj(P_PART);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, histTex); gl.uniform1i(P_PART.u.uHist, 0);
    gl.uniform1i(P_PART.u.uHead, head);
    const gain = +(qs.get('gain') || 0.8) * (d > 1.4 ? 1.25 : 1);
    gl.uniform1f(P_PART.u.uGain, (RT.scene.hdr ? gain : gain * 0.8) * (1 - 0.3 * view.morph));
    gl.uniform1f(P_PART.u.uMaxSeg, 60 * d);
    gl.bindVertexArray(emptyVAO);
    gl.drawArrays(gl.LINES, 0, nP * (K - 1) * 2);
    gl.disable(gl.BLEND);
  }
  // 2. bloom: down 4x + separable blur
  const A = RT.a, B = RT.b;
  gl.bindVertexArray(emptyVAO);
  gl.useProgram(P_DOWN.p); gl.bindFramebuffer(gl.FRAMEBUFFER, A.fb); gl.viewport(0, 0, A.w, A.h);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, RT.scene.t); gl.uniform1i(P_DOWN.u.uT, 0);
  gl.uniform2f(P_DOWN.u.uTexel, 1 / bw, 1 / bh); gl.uniform2f(P_DOWN.u.uOut, A.w, A.h);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.useProgram(P_BLUR.p); gl.uniform1i(P_BLUR.u.uT, 0); gl.uniform2f(P_BLUR.u.uOut, A.w, A.h);
  for (let it = 0; it < 2; it++) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, B.fb); gl.bindTexture(gl.TEXTURE_2D, A.t); gl.uniform2f(P_BLUR.u.uDir, (1.4 + it) / A.w, 0); gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, A.fb); gl.bindTexture(gl.TEXTURE_2D, B.t); gl.uniform2f(P_BLUR.u.uDir, 0, (1.4 + it) / A.h); gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  // 3. composite to screen
  gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, bw, bh);
  gl.useProgram(P_COMP.p);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, RT.scene.t); gl.uniform1i(P_COMP.u.uScene, 0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, A.t); gl.uniform1i(P_COMP.u.uBloom, 1);
  gl.uniform2f(P_COMP.u.uRes, bw, bh); gl.uniform2f(P_COMP.u.uGC, geo.cx * d, (geo.H - geo.cy) * d);
  gl.uniform1f(P_COMP.u.uGR, geo.R * d); gl.uniform1f(P_COMP.u.uMorph, view.morph); gl.uniform1f(P_COMP.u.uDpr, d);
  gl.uniform1f(P_COMP.u.uBloomK, 0.6); gl.uniform1f(P_COMP.u.uExp, 1.25);
  const hw = geo.Kf * Math.PI * d, fcy = (geo.H - geo.fcy) * d, K2 = geo.Kf * d * DEG;
  gl.uniform4f(P_COMP.u.uRect, geo.fcx * d - hw, fcy + (-90 - view.flatLat) * K2, geo.fcx * d + hw, fcy + (90 - view.flatLat) * K2);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  // 4. coastlines on top, dimmed under bright flow
  if (coastVAO) {
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(P_COAST.p); setProj(P_COAST);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, RT.scene.t); gl.uniform1i(P_COAST.u.uScene, 0);
    gl.uniform1f(P_COAST.u.uMaxSeg, 200 * d); gl.uniform1f(P_COAST.u.uAlpha, d > 1.4 ? 0.85 : 0.7);
    gl.bindVertexArray(coastVAO); gl.drawArrays(gl.LINES, 0, coastCount);
    gl.disable(gl.BLEND);
  }
  gl.bindVertexArray(null);
}

// ------------------------------------------------------------------ UI
const elTime = $('time'), elScrub = $('scrub'), elFill = $('fill'), elBuf = $('buf'), scrubBox = elScrub.parentElement;
const elPlay = $('play'), elSpeed = $('speed');
const fmtT = (ms, year) => { const t = new Date(ms); const p = n => String(n).padStart(2, '0');
  return `${p(t.getUTCDate())} ${MON[t.getUTCMonth()]}${year ? ' ' + t.getUTCFullYear() : ''} ${p(t.getUTCHours())}:${p(t.getUTCMinutes())}`; };
function maxHours() { return 6 * Math.max(0, time.prefix - 1); }
function totalHours() { return 6 * (time.NT - 1); }
function setPlaying(p) {
  time.playing = p; elPlay.setAttribute('aria-label', p ? 'Pause' : 'Play');
  sty('ic-pause').display = p ? '' : 'none'; sty('ic-play').display = p ? 'none' : '';
}
elPlay.onclick = () => { if (!time.playing && time.hours >= totalHours() - 0.01) time.hours = 0; setPlaying(!time.playing); };
elSpeed.onclick = () => { time.speedIdx = (time.speedIdx + 1) % SPEEDS.length; const s = SPEEDS[time.speedIdx];
  elSpeed.textContent = (s < 1 ? String(s).replace(/^0/, '') : s) + '×'; };
elScrub.addEventListener('input', () => { pendingT = null; time.hours = Math.min(+elScrub.value, maxHours()); scrubBox.classList.add('active'); refreshReadout(); });
elScrub.addEventListener('change', () => scrubBox.classList.remove('active'));
function setMode(flat) {
  view.target = flat ? 1 : 0; $('v-globe').classList.toggle('on', !flat); $('v-map').classList.toggle('on', flat);
  $('hint').textContent = (flat ? 'Drag to pan' : 'Drag to rotate') + (matchMedia('(pointer:coarse)').matches ? ' · Pinch to zoom · Tap for wind' : ' · Scroll to zoom · Click for wind');
}
$('v-globe').onclick = () => setMode(false); $('v-map').onclick = () => setMode(true);
{ // legend gradient
  const st = []; for (let v = 0; v <= 90; v += 3) st.push(`${css(ramp(v))} ${(v / 90 * 100).toFixed(1)}%`);
  sty('lg-bar').background = `linear-gradient(90deg,${st.join(',')})`;
}
function updateTimeUI() {
  const H = totalHours() || 168, p = time.hours / H;
  elTime.textContent = fmtT(times0 + time.hours * 3600e3) + ' UTC';
  elFill.style.width = (p * 100) + '%'; scrubBox.style.setProperty('--p', p);
  elBuf.style.width = (maxHours() / H * 100) + '%';
  if (document.activeElement !== elScrub) elScrub.value = time.hours;
}
addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' && e.key !== ' ') return;
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') pendingT = null;
  if (e.key === ' ') { e.preventDefault(); elPlay.click(); }
  else if (e.key === 'ArrowRight') time.hours = Math.min(maxHours(), time.hours + 1);
  else if (e.key === 'ArrowLeft') time.hours = Math.max(0, time.hours - 1);
  else if (e.key === 'g' || e.key === 'm') setMode(view.target < 0.5);
  else if (e.key === 'f') $('fps').hidden = !$('fps').hidden;
  else if (e.key === 'Escape') closeReadout();
});
if (qs.get('fps')) $('fps').hidden = false;
if (qs.get('view') === 'map') { setMode(true); view.morph = 1; } else setMode(false);

// labels
const labelEls = LABELS.map(([n]) => { const e = document.createElement('div'); e.className = 'lb'; e.textContent = n; $('labels').appendChild(e); return e; });
function updateLabels() {
  const small = geo.W < 500;
  LABELS.forEach(([, lo, la], i) => {
    projectLL(lo, la, tmp3); const e = labelEls[i];
    let a = tmp3[2] < 0.35 ? 0 : Math.min(1, (tmp3[2] - 0.35) / 0.25);
    const wLab = e._w || (e._w = e.offsetWidth) || 100;
    if (tmp3[0] < 4 || tmp3[0] + wLab > geo.W - 4 || tmp3[1] < 30 || tmp3[1] > geo.H) a = 0;
    if (small && geo.R < 170 && i >= 6 && view.morph < 0.5) a *= 0; // declutter tiny globes
    e.style.opacity = (a * 0.85).toFixed(3);
    if (a > 0) e.style.transform = `translate(${(tmp3[0] - 2).toFixed(1)}px,${(tmp3[1] - 7).toFixed(1)}px)`;
  });
}

// readout
const RO = { on: false, lon: 0, lat: 0 }, elRO = $('readout'), elMarker = $('marker');
function fmtLL(lat, lon) {
  const la = `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? 'N' : 'S'}`, lo = `${Math.abs(lon).toFixed(2)}° ${lon >= 0 ? 'E' : 'W'}`;
  return `${la}  ${lo}`;
}
function refreshReadout() {
  if (!RO.on || !fieldReady) return;
  updateField(time.hours / 6);
  const [u, v] = sampleUV(RO.lon, RO.lat), s = Math.hypot(u, v);
  const from = (Math.atan2(-u, -v) / DEG + 360) % 360;
  $('ro-loc').textContent = fmtLL(RO.lat, RO.lon);
  $('ro-ms').textContent = s.toFixed(1);
  sty('ro-sw').background = css(ramp(s)); sty('ro-sw').color = css(ramp(s));
  $('ro-alt').textContent = `${(s * 2.236936).toFixed(1)} mph · ${(s * 3.6).toFixed(1)} km/h`;
  $('ro-dir').textContent = s < 0.5 ? 'Calm' : `From ${COMPASS[Math.round(from / 22.5) % 16]} · ${Math.round(from)}°`;
  sty('ro-arrow').transform = `rotate(${(from + 180).toFixed(0)}deg)`;
  $('ro-time').textContent = fmtT(times0 + time.hours * 3600e3, true) + ' UTC · 250 mb';
}
function openReadout(lon, lat) { Object.assign(RO, { on: true, lon, lat }); elRO.hidden = false; elMarker.hidden = false; refreshReadout(); }
function closeReadout() { RO.on = false; elRO.hidden = true; elMarker.hidden = true; }
$('ro-close').onclick = closeReadout;
function updateMarker() {
  if (!RO.on) return;
  projectLL(RO.lon, RO.lat, tmp3);
  const vis = tmp3[2] > 0;
  elMarker.style.opacity = vis ? Math.min(1, tmp3[2] * 3) : 0;
  elMarker.style.transform = `translate(${tmp3[0].toFixed(1)}px,${tmp3[1].toFixed(1)}px)`;
}

// ------------------------------------------------------------------ interaction
const ptrs = new Map(); let pinch = null, downInfo = null;
function touched() { inter.last = performance.now(); $('hint')?.classList.add('gone'); }
function zoomBy(f, mx, my) {
  if (view.target < 0.5) view.zoomG = clamp(view.zoomG * f, 0.6, 12);
  else {
    const z0 = view.zoomF, z1 = clamp(z0 * f, 1, 16); if (z1 === z0) return;
    const K0 = geo.baseK * z0, K1 = geo.baseK * z1;
    view.lon0 += (mx - geo.fcx) / DEG * (1 / K0 - 1 / K1);
    view.flatLat += (geo.fcy - my) / DEG * (1 / K0 - 1 / K1);
    view.zoomF = z1;
  }
}
function dragBy(dx, dy) {
  if (view.target < 0.5) {
    const R = geo.baseR * view.zoomG;
    view.lon0 -= dx / R / DEG; view.lat0 = clamp(view.lat0 + dy / R / DEG, -85, 85);
  } else { const Kf = geo.baseK * view.zoomF; view.lon0 -= dx / Kf / DEG; view.flatLat += dy / Kf / DEG; }
}
canvas.addEventListener('pointerdown', e => {
  canvas.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY }); touched();
  inter.vx = inter.vy = 0;
  if (ptrs.size === 1) { downInfo = { x: e.clientX, y: e.clientY, t: performance.now(), moved: 0 }; inter.dragging = true; canvas.classList.add('drag'); }
  if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 }; downInfo = null; }
});
canvas.addEventListener('pointermove', e => {
  const p = ptrs.get(e.pointerId); if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y; p.x = e.clientX; p.y = e.clientY; touched();
  if (ptrs.size === 1) { dragBy(dx, dy); inter.vx = inter.vx * 0.5 + dx * 0.5; inter.vy = inter.vy * 0.5 + dy * 0.5; if (downInfo) downInfo.moved += Math.abs(dx) + Math.abs(dy); }
  else if (ptrs.size === 2 && pinch) {
    const [a, b] = [...ptrs.values()], d = Math.hypot(a.x - b.x, a.y - b.y), mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    zoomBy(d / pinch.d, mx, my); dragBy((mx - pinch.mx) * 0.9, (my - pinch.my) * 0.9); pinch = { d, mx, my };
  }
});
function endPtr(e) {
  if (!ptrs.has(e.pointerId)) return;
  ptrs.delete(e.pointerId);
  if (ptrs.size < 2) pinch = null;
  if (ptrs.size === 0) {
    inter.dragging = false; canvas.classList.remove('drag');
    if (downInfo && downInfo.moved < 8 && performance.now() - downInfo.t < 600 && e.type === 'pointerup') {
      inter.vx = inter.vy = 0;
      const ll = unproject(e.clientX, e.clientY); if (ll) openReadout(ll[0], ll[1]); else closeReadout();
    }
    downInfo = null;
  }
}
canvas.addEventListener('pointerup', endPtr); canvas.addEventListener('pointercancel', endPtr);
canvas.addEventListener('wheel', e => { e.preventDefault(); touched();
  const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY; zoomBy(Math.exp(-dy * 0.0015), e.clientX, e.clientY); }, { passive: false });
addEventListener('resize', () => { layout(); labelEls.forEach(e => e._w = 0); });
if (document.fonts) document.fonts.ready.then(() => labelEls.forEach(e => e._w = 0));
new ResizeObserver(() => layout()).observe(bottomEl);
// expose a tiny hook for automated testing
window.__jet = { openAt(sx, sy) { const ll = unproject(sx, sy); if (ll) openReadout(ll[0], ll[1]); return ll; }, stats: {}, view, time, setMode };

// ------------------------------------------------------------------ main loop
let lastT = performance.now(), acc = { n: 0, t: 0, cpu: 0 }, lowCount = 0, started = false;
function frame(now) {
  requestAnimationFrame(frame);
  const dtReal = Math.min(0.1, (now - lastT) / 1000); lastT = now;
  if (!started) return;
  const c0 = performance.now();
  // time
  if (time.playing && !inter.dragging) {
    time.hours += dtReal * HOURS_PER_SEC * SPEEDS[time.speedIdx];
    if (time.hours > maxHours()) { time.hours = time.prefix >= time.NT ? 0 : maxHours(); }
  }
  time.hours = clamp(time.hours, 0, maxHours());
  // view animation
  const wasFlat = view.morph >= 0.5;
  view.morph += (view.target - view.morph) * Math.min(1, dtReal * 5);
  if ((view.morph >= 0.5) !== wasFlat) for (let i = 0; i < nP; i++) plife[i] = Math.max(1, page[i]) + (Math.random() * 24 | 0);  // re-seed quickly into the new view
  if (Math.abs(view.target - view.morph) < 0.002) view.morph = view.target;
  const idle = now - inter.last;
  if (!inter.dragging) {
    if (Math.abs(inter.vx) + Math.abs(inter.vy) > 0.05) { dragBy(inter.vx, inter.vy); inter.vx *= 0.92; inter.vy *= 0.92; }
    if (view.target < 0.5 && qs.get('spin') !== '0') {
      const k = clamp((idle - 4000) / 2500, 0, 1);         // resume gently after 4 s idle
      view.lon0 -= 3.2 * dtReal * k * k * (3 - 2 * k);
    }
  }
  view.lon0 = ((view.lon0 + 540) % 360) - 180;
  geo.R = geo.baseR * view.zoomG; geo.Kf = geo.baseK * view.zoomF;
  const maxLat = Math.max(0, 90 - geo.fcy / geo.Kf / DEG);
  view.flatLat = clamp(view.flatLat, -maxLat, maxLat);
  ROT = rotMatrix();
  // particles
  updateField(time.hours / 6);
  const want = targetCount();
  if (Math.abs(want - nP) > nP * 0.08 || want < nP) setCount(want);
  // advection step per frame: constant on-screen speed at zoom 1, longer trails when zoomed in
  const zNow = view.zoomG + (view.zoomF - view.zoomG) * view.morph;
  const scale = (geo.baseR + (geo.baseK - geo.baseR) * view.morph) * Math.sqrt(zNow);
  const dtAdv = 230000 / scale * clamp(dtReal * 60, 0.5, 2);
  stepParticles(dtAdv);
  const c1 = performance.now();
  render();
  updateTimeUI(); updateLabels(); updateMarker();
  if (RO.on && time.playing) refreshReadout();
  frameNo++;
  // fps + adaptive quality
  acc.n++; acc.t += dtReal; acc.cpu += c1 - c0;
  if (acc.t >= 1) {
    const fps = acc.n / acc.t, cpu = acc.cpu / acc.n;
    Object.assign(window.__jet.stats, { fps: +fps.toFixed(1), cpuMs: +cpu.toFixed(2), particles: nP, quality: +quality.toFixed(2), hdr: RT.scene.hdr, dpr: geo.dpr });
    if (!$('fps').hidden) $('fps').textContent = `${fps.toFixed(0)} FPS · ${nP.toLocaleString()} PARTICLES\nCPU ${cpu.toFixed(1)} MS · Q ${quality.toFixed(2)}${RT.scene.hdr ? '' : ' · LDR'}`;
    if (adapt && frameNo > 90 && !document.hidden) {
      if (fps < 48) { if (++lowCount >= 2) { quality = Math.max(0.25, quality * 0.8); lowCount = 0; } }
      else { lowCount = 0; if (fps > 58 && quality < 1) quality = Math.min(1, quality * 1.05); }
    }
    acc = { n: 0, t: 0, cpu: 0 };
  }
}

// ------------------------------------------------------------------ loading
const dcv = document.createElement('canvas'); dcv.width = NLON; dcv.height = NLAT;
const dctx = dcv.getContext('2d', { willReadFrequently: true });
async function fetchOK(url) {
  let r;
  try { r = await fetch(url); } catch (e) { throw new DataError(url + ': ' + (e.message || e)); }
  if (!r.ok) throw new DataError(url + ' HTTP ' + r.status);
  return r;
}
async function fetchJSON(url) {
  const r = await fetchOK(url);
  try { return await r.json(); } catch (e) { throw new DataError(url + ': invalid JSON'); }
}
async function decodePNG(url) {
  const r = await fetchOK(url);
  let blob;
  try { blob = await r.blob(); } catch (e) { throw new DataError(url + ': ' + (e.message || e)); }
  let img;
  if (window.createImageBitmap) {
    try { img = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' }); }
    catch (e) { img = await createImageBitmap(blob); }
  } else {
    img = new Image(); img.src = URL.createObjectURL(blob); await img.decode();
  }
  dctx.clearRect(0, 0, NLON, NLAT); dctx.drawImage(img, 0, 0);
  return dctx.getImageData(0, 0, NLON, NLAT).data;
}
async function decodePNGRetry(url, tries = 3) {
  for (let k = 1; ; k++) {
    try { return await decodePNG(url); }
    catch (e) {
      if (k >= tries) throw e instanceof DataError ? e : new DataError(url + ': ' + (e.message || e));
      await new Promise(r => setTimeout(r, 1000 * k));
    }
  }
}
async function loadFrame(k) {
  const m = manifest.frames[k], d = await decodePNGRetry('data/' + m.file);
  const qu = new Uint16Array(NG), qv = new Uint16Array(NG);
  for (let i = 0, o = 0; i < NG; i++, o += 4) { const b = d[o+2]; qu[i] = (d[o] << 4) | (b >> 4); qv[i] = (d[o+1] << 4) | (b & 15); }
  frames[k] = { qu, qv, u0: m.u[0], su: (m.u[1] - m.u[0]) / 4095, v0: m.v[0], sv: (m.v[1] - m.v[0]) / 4095 };
  time.loaded[k] = true;
  while (time.loaded[time.prefix]) time.prefix++;
  sty('ld-fill').width = (time.prefix / time.NT * 100) + '%';   // #loading is removed once playback starts
  if (pendingT !== null && maxHours() >= Math.min(pendingT, totalHours())) { time.hours = Math.min(pendingT, totalHours()); pendingT = null; }
}
async function boot() {
  try {
    const [man, coast] = await Promise.all([fetchJSON('data/manifest.json'), fetchJSON('data/coast.json')]);
    manifest = man; time.NT = man.times.length; frames = new Array(time.NT);
    times0 = Date.parse(man.times[0].replace('Z', ':00Z'));
    const t1 = times0 + (time.NT - 1) * 6 * 3600e3;
    $('range').textContent = `${fmtT(times0)} – ${fmtT(t1, true)} UTC`;
    elScrub.max = totalHours();
    const ticks = $('ticks'); const days = totalHours() / 24;
    for (let k = 1; k < days; k++) { const i = document.createElement('i'); i.style.left = (k / days * 100) + '%'; ticks.appendChild(i); }
    buildCoast(coast);
    layout(); geo.R = geo.baseR * view.zoomG; geo.Kf = geo.baseK * view.zoomF;
    // load in time order, a few in parallel; start once the first two are in
    let next = 0;
    let bootErr = null;
    const worker = async () => {
      while (next < time.NT) {
        const k = next++;
        try { await loadFrame(k); }
        catch (e) {
          if (!started) { bootErr = bootErr || e; return; }
          console.warn('wind frame', k, 'failed to load; the timeline stops before it:', e);   // keep playing what we have
        }
      }
    };
    const all = Promise.all([worker(), worker(), worker(), worker()]);
    while (time.prefix < 2) {
      if (bootErr) throw bootErr;
      await new Promise(r => setTimeout(r, 30));
    }
    time.hours = clamp(time.hours, 0, maxHours());
    updateField(time.hours / 6);
    setCount(targetCount());
    for (let i = 0; i < 40; i++) { frameNo++; stepParticles(230000 / geo.R); }   // warm up trails
    started = true; setPlaying(time.playing);
    $('loading')?.classList.add('done'); setTimeout(() => $('loading')?.remove(), 1000);
    setTimeout(() => $('hint')?.classList.add('gone'), 9000);
    await all;
  } catch (e) {
    console.error(e);
    const esc = x => String(x).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
    if (started && !(e instanceof DataError)) return;   // already running: never blank the screen for a late UI error
    fatal(location.protocol === 'file:'
      ? 'Browsers block data loading from <b>file://</b> pages.<br>Serve this folder with any static server, e.g.<br><code>python3 -m http.server</code> and open <code>http://localhost:8000</code>'
      : e instanceof DataError
        ? 'Could not load the wind data (' + esc(e.message) + ').<br>Check your connection and reload the page.'
        : 'The visualization failed to start (' + esc(e.message || e) + ').');
  }
}
requestAnimationFrame(frame);
boot();
})();
