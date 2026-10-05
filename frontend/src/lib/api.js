// Thin client for the FastAPI backend.
const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");

export const apiUrl = (path) => `${BASE}${path}`;

export class ApiError extends Error {
  constructor(message, code = "ERROR", status = 0) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

async function handle(res) {
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) {
    const err = body?.error;
    if (err) throw new ApiError(err.message, err.code, res.status);
    if (res.status === 413) throw new ApiError("The image is too large for the server.", "TOO_LARGE", 413);
    throw new ApiError(`Server error (${res.status}). Please try again.`, "HTTP", res.status);
  }
  return body;
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(apiUrl(path), options);
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError("Can't reach the server. Check your connection or wait for it to wake up.", "NETWORK");
  }
  return handle(res);
}

function form(file, fields = {}) {
  const fd = new FormData();
  fd.append("file", file, file.name || "image.png");
  for (const [k, v] of Object.entries(fields)) {
    if (v === undefined || v === null) continue;
    fd.append(k, typeof v === "object" ? JSON.stringify(v) : String(v));
  }
  return fd;
}

export const api = {
  health: (signal) => request("/api/health", { signal }),
  modules: () => request("/api/modules"),
  samples: () => request("/api/samples"),
  diagnose: (file, signal) => request("/api/diagnose", { method: "POST", body: form(file), signal }),
  process: (file, module, params, engine, signal) =>
    request("/api/process", { method: "POST", body: form(file, { module, params, engine }), signal }),
  guided: (file, intent, params, engine, signal) =>
    request("/api/guided", { method: "POST", body: form(file, { intent, params, engine }), signal }),
  auto: (file, engine, signal) => request("/api/auto", { method: "POST", body: form(file, { engine }), signal }),
  inspect: (file, signal) => request("/api/inspect", { method: "POST", body: form(file), signal }),
  compress: (file, params, engine, signal) =>
    request("/api/compress", { method: "POST", body: form(file, { params, engine }), signal }),
  sweep: (file, intent, params, engine, signal) =>
    request("/api/sweep", { method: "POST", body: form(file, { intent, params, engine }), signal }),
  compareCompression: (file, keepPercent, jpegQuality, engine, signal) =>
    request("/api/compare-compression", {
      method: "POST", body: form(file, { keep_percent: keepPercent, jpeg_quality: jpegQuality ?? "auto", engine }), signal,
    }),
};

/** Ping /api/health until the server answers (Render's free tier sleeps and needs ~30-60 s to wake). */
export async function waitForServer(onAttempt, { tries = 40, delayMs = 3000 } = {}) {
  for (let i = 0; i < tries; i++) {
    onAttempt?.(i);
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 8000);
      const r = await api.health(ctrl.signal);
      clearTimeout(t);
      if (r?.status === "ok") return r;
    } catch {
      /* keep trying */
    }
    await new Promise((r) => setTimeout(r, delayMs));
  }
  throw new ApiError("The server did not respond.", "OFFLINE");
}
