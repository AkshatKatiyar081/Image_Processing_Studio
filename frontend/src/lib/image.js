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
 * smoothing would blend them into their neighbours and hide the defect.) The shrunk image is
 * sent as PNG, which is lossless.
 *
 * PNG/JPEG/WebP/BMP files within the limit are uploaded unchanged; other formats are re-encoded
 * to PNG at full size.
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
  if (reencode) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;      // nearest-neighbour: keeps isolated specks intact
    ctx.drawImage(bmp, 0, 0, width, height);
    out = await canvasToBlob(canvas, "image/png");
  }
  bmp.close?.();
  const base = name.replace(/\.[^.]+$/, "") || "image";
  const file = new File([out], reencode ? `${base}.png` : name, { type: out.type });
  return { file, url: URL.createObjectURL(file), width, height, name: base, resized, originalSize: [w0, h0] };
}

export async function dataUrlToFile(dataUrl, name) {
  const blob = await (await fetch(dataUrl)).blob();
  return new File([blob], name, { type: blob.type });
}

export function download(url, filename) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
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

export const extFor =(dataUrl) => (dataUrl?.startsWith("data:image/jpeg") ? "jpg" : "png");
