// Client-side image helpers: load, shrink before upload, convert, download.
// Longest side sent to the server. 0 (the default) = upload the full resolution, for running locally.
// For a small hosted backend (Render free tier) set VITE_MAX_UPLOAD_SIDE=2048 at build time.
export const MAX_UPLOAD_SIDE = Number(import.meta.env.VITE_MAX_UPLOAD_SIDE || 0);
const UPLOAD_AS_IS = ["image/png", "image/jpeg", "image/webp", "image/bmp"];

async function decode(blob) {
  try {
    return await createImageBitmap(blob, { imageOrientation: "from-image" });
  } catch {
    // Fallback for browsers without createImageBitmap options
    const url = URL.createObjectURL(blob);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      return img;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

function canvasToBlob(canvas, type = "image/png", quality) {
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("encode failed"))), type, quality),
  );
}

/**
 * Prepare a user file for upload.
 *
 * By default the file is uploaded at full resolution. If MAX_UPLOAD_SIDE is set (e.g. 2048 for
 * Render's free tier, 512 MB RAM), larger photos are shrunk here first. The shrink uses NEAREST-NEIGHBOUR
 * sampling (imageSmoothingEnabled = false): every output pixel is a copy of one input pixel,
 * so salt-and-pepper specks stay specks and their density is preserved. (The default bilinear
 * smoothing would blend them into their neighbours and hide the defect.) A shrunk JPEG is sent as a
 * JPEG (quality 95), anything else as PNG: re-saving a photo as PNG would make the file several times
 * BIGGER, which defeats the purpose on a server with an upload limit.
 *
 * PNG/JPEG/WebP/BMP files within the limit are uploaded unchanged; other formats are re-encoded
 * to PNG at full size. The caller checks the result against the server's limits (limitProblem).
 * Returns { file, url, width, height, name, resized, originalSize } where width/height are the
 * dimensions of the UPLOADED image.
 */
export async function prepareImage(blob, name = "image.png") {
  if (!blob.type.startsWith("image/")) throw new Error("That file is not an image.");
  const bmp = await decode(blob);
  const w0 = bmp.width, h0 = bmp.height;
  if (!w0 || !h0) throw new Error("Could not read that image.");

  const scale = MAX_UPLOAD_SIDE > 0 ? Math.min(1, MAX_UPLOAD_SIDE / Math.max(w0, h0)) : 1;
  const width = Math.max(1, Math.round(w0 * scale));
  const height = Math.max(1, Math.round(h0 * scale));
  const resized = scale < 1;
  const reencode = resized || !UPLOAD_AS_IS.includes(blob.type);

  let out = blob;
  let ext = null;
  if (reencode) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;      // nearest-neighbour: keeps isolated specks intact
    ctx.drawImage(bmp, 0, 0, width, height);
    const asJpeg = resized && blob.type === "image/jpeg";
    out = await canvasToBlob(canvas, asJpeg ? "image/jpeg" : "image/png", asJpeg ? 0.95 : undefined);
    ext = asJpeg ? "jpg" : "png";
  }
  bmp.close?.();
  const base = name.replace(/\.[^.]+$/, "") || "image";
  const file = new File([out], ext ? `${base}.${ext}` : name, { type: out.type });
  return { file, url: URL.createObjectURL(file), width, height, name: base, resized, originalSize: [w0, h0] };
}

/** Save a URL or a Blob/File under `filename`. */
export function download(urlOrBlob, filename) {
  const isBlob = typeof urlOrBlob !== "string";
  const url = isBlob ? URL.createObjectURL(urlOrBlob) : urlOrBlob;
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  if (isBlob) setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** The server's limits as the dashboard shows them, e.g. "4.2 MB · 25 MP". */
export function limitsText(limits) {
  if (!limits) return "";
  return `${formatMB(limits.max_upload_bytes)} · ${limits.max_pixels / 1e6} MP`;
}

const formatMB = (n) => `${(n / 1024 / 1024).toFixed(1)} MB`;

/** Why a file cannot be uploaded (over the server's file-size or pixel limit), or null if it is fine.
 *  Checked in the browser so the user gets a clear message instead of a failed upload. */
export function limitProblem(limits, { bytes, width, height }) {
  if (!limits) return null;
  if (bytes > limits.max_upload_bytes) {
    return `This file is ${formatMB(bytes)}; the limit is ${formatMB(limits.max_upload_bytes)}. ` +
      "Save it as a JPEG or reduce its size, then try again.";
  }
  if (width * height > limits.max_pixels) {
    return `This image is ${width} × ${height} (${(width * height / 1e6).toFixed(1)} MP); the limit is ` +
      `${limits.max_pixels / 1e6} MP. Reduce its size, then try again.`;
  }
  return null;
}

/** 256-bin luminance histogram computed in the browser (used before the first server result). */
export async function lumaHistogram(url) {
  const img = new Image();
  img.src = url;
  await img.decode();
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const h = new Array(256).fill(0);
  for (let i = 0; i < data.length; i += 4) {
    h[Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2])]++;
  }
  return h;
}

