// Client-side image helpers: load, shrink before upload, convert, download.
export const MAX_SIDE = 1024;
const LOSSLESS_KEEP = ["image/png", "image/jpeg", "image/webp", "image/bmp"];

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
 * Prepare a user file for processing.
 * Files are sent at their ORIGINAL resolution — the server's prepare() step
 * downscales with INTER_AREA and measures impulse_density before doing so,
 * so salt-and-pepper spikes survive to the detection stage.
 * Browser canvas drawImage uses bilinear smoothing that would average spikes
 * away if we resized here, which is why we no longer do that.
 * Unsupported formats are re-encoded as PNG (lossless) without resizing.
 * Returns { file, url, width, height, name, resized }.
 */
export async function prepareImage(blob, name = "image.png") {
  if (!blob.type.startsWith("image/")) throw new Error("That file is not an image.");
  const bmp = await decode(blob);
  const w0 = bmp.width, h0 = bmp.height;
  if (!w0 || !h0) throw new Error("Could not read that image.");

  // Only re-encode if the format isn't supported for direct upload — never resize.
  const needsReencode = !LOSSLESS_KEEP.includes(blob.type);
  let out = blob;
  if (needsReencode) {
    const canvas = document.createElement("canvas");
    canvas.width = w0;
    canvas.height = h0;
    canvas.getContext("2d").drawImage(bmp, 0, 0, w0, h0);
    out = await canvasToBlob(canvas, "image/png");
  }
  bmp.close?.();
  const base = name.replace(/\.[^.]+$/, "") || "image";
  const file = new File([out], needsReencode ? `${base}.png` : name, { type: out.type });
  // Compute display dimensions (thumbnail only — not what gets uploaded).
  const scale = Math.min(1, MAX_SIDE / Math.max(w0, h0));
  const dispW = Math.max(1, Math.round(w0 * scale));
  const dispH = Math.max(1, Math.round(h0 * scale));
  return { file, url: URL.createObjectURL(file), width: dispW, height: dispH, name: base, resized: scale < 1, originalSize: [w0, h0] };
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
