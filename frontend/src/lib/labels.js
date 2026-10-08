// Friendly names, presets and explanations for everything the pipeline exposes.
import { FileArchive, Scissors, Sparkles, Sun, ScanLine } from "lucide-react";

export const TOOLS = [
  {
    id: "contrast",
    name: "Light & contrast",
    icon: Sun,
    blurb: "Brightness, contrast, histogram equalization, uneven lighting.",
    presets: [
      { id: "auto_gamma", label: "Auto brightness", hint: "Gamma chosen so the average becomes mid-gray", params: { mode: "gamma", gamma: "auto" } },
      { id: "stretch", label: "Boost contrast", hint: "Stretch intensities to use the full 0–255 range", params: { mode: "stretch", clip_percentile: 1 } },
      { id: "hist_eq", label: "Equalize histogram", hint: "Spread intensities evenly over 0–255", params: { mode: "hist_eq" } },
    ],
  },
  {
    id: "denoise",
    name: "Remove noise",
    icon: Sparkles,
    blurb: "Median, box and Gaussian filters.",
    presets: [
      { id: "median3", label: "Salt & pepper specks", hint: "3×3 median filter", params: { filter_type: "median", kernel_size: 3 } },
      { id: "median5", label: "Heavy specks", hint: "5×5 median filter", params: { filter_type: "median", kernel_size: 5 } },
      { id: "median7", label: "Big specks (e.g. screenshots)", hint: "7×7 median: specks 2–3 px wide", params: { filter_type: "median", kernel_size: 7 } },
      { id: "gauss", label: "Grainy / sensor noise", hint: "Gaussian blur, σ = 1", params: { filter_type: "gaussian", sigma: 1.0 } },
      { id: "gauss_coarse", label: "Coarse grain (e.g. screenshots)", hint: "Gaussian blur, σ = 1.5: grain 2–3 px wide", params: { filter_type: "gaussian", sigma: 1.5 } },
      { id: "box3", label: "Simple blur", hint: "3×3 averaging filter", params: { filter_type: "box", kernel_size: 3 } },
    ],
  },
  {
    id: "sharpen_edges",
    name: "Sharpen & edges",
    icon: ScanLine,
    blurb: "Unsharp masking, Laplacian high-boost, Sobel / Prewitt / Roberts.",
    presets: [
      { id: "sharpen", label: "Sharpen", hint: "Unsharp masking, k = 1", params: { operation: "unsharp", k: 1.0, blur_sigma: 1.0 } },
      { id: "strong", label: "Strong sharpen", hint: "High-boost via unsharp masking, k = 2", params: { operation: "unsharp", k: 2.0, blur_sigma: 1.0 } },
      { id: "laplacian", label: "Laplacian sharpen", hint: "Laplacian high-boost, centre weight 9", params: { operation: "laplacian_boost", A: 1.0 } },
      { id: "sobel", label: "Edges — Sobel", hint: "Gradient magnitude", params: { operation: "gradient", operator: "sobel", edge_threshold: 0 } },
      { id: "sobel_bw", label: "Edges — black & white", hint: "Sobel, thresholded at 20 %", params: { operation: "gradient", operator: "sobel", edge_threshold: 0.2 } },
      { id: "prewitt", label: "Edges — Prewitt", hint: "Gradient magnitude", params: { operation: "gradient", operator: "prewitt", edge_threshold: 0 } },
      { id: "roberts", label: "Edges — Roberts", hint: "2×2 diagonal differences", params: { operation: "gradient", operator: "roberts", edge_threshold: 0 } },
      { id: "laplace_edges", label: "Edges — Laplacian", hint: "Laplacian high-pass, centre weight 8", params: { operation: "laplacian_boost", A: 0 } },
    ],
  },
  {
    id: "compress",
    name: "Compression",
    icon: FileArchive,
    blurb: "Make the file smaller: JPEG-style 8×8 block DCT with quantisation. Whole-image DCT and Walsh–Hadamard show the transforms on their own.",
    presets: [
      { id: "jpeg_high", label: "High quality", hint: "JPEG-style, quality 90: hard to tell apart", params: { method: "jpeg", quality: 90 } },
      { id: "jpeg_balanced", label: "Balanced", hint: "JPEG-style, quality 75: much smaller, little visible loss", params: { method: "jpeg", quality: 75 } },
      { id: "jpeg_small", label: "Small file", hint: "JPEG-style, quality 50", params: { method: "jpeg", quality: 50 } },
      { id: "jpeg_tiny", label: "Smallest file", hint: "JPEG-style, quality 20: visible 8×8 blocks", params: { method: "jpeg", quality: 20 } },
      { id: "dct_global", label: "Whole-image DCT, keep 10 %", hint: "Transform only: zeroes the smaller coefficients; saved in your file's format", params: { method: "dct", mode: "keep_percent", keep_percent: 10 } },
      { id: "wht_global", label: "Walsh–Hadamard, keep 10 %", hint: "Transform only, square-wave basis; saved in your file's format", params: { method: "hadamard", mode: "keep_percent", keep_percent: 10 } },
    ],
  },
];

/** Schema options that exist in the pipeline but are not offered in the Tools tab.
 *  contrast "flatten" stays in the pipeline because guided segmentation (Compare) uses it. */
export const HIDDEN_OPTIONS = { contrast: { mode: ["flatten"] } };

export const toolById = (id) => TOOLS.find((t) => t.id === id);

export const PARAM_LABELS = {
  mode: "Method", gamma: "Gamma (γ)", clip_percentile: "Clip each end (%)", se_radius: "Background disk radius (px)",
  object_polarity: "Objects are", filter_type: "Filter", kernel_size: "Window size", sigma: "Gaussian σ",
  operation: "Operation", A: "High-boost factor A", k: "Unsharp gain k", blur_sigma: "Blur σ", operator: "Operator",
  magnitude: "Magnitude", component: "Show", edge_threshold: "Edge threshold", method: "Method",
  foreground: "Object is", seed: "Seed point", delta_t: "Similarity tolerance ΔT", connectivity: "Connectivity",
  adaptive: "Adaptive (running mean)", morph_open: "Remove specks (opening)", morph_close: "Fill small holes (closing)",
  disk_radius: "Cleanup disk radius", fill_holes: "Fill enclosed holes", keep_largest: "Keep largest object only",
  keep_percent: "Keep % of coefficients", threshold: "Threshold |F|", zonal_fraction: "Zone size k (fraction of side)",
  zonal_shape: "Zone shape", quality: "Quality (1–100)", chroma_subsampling: "Colour subsampling",
};

/** Labels that depend on the tool (the same parameter name means different things in different modules). */
export const TOOL_PARAM_LABELS = {
  compress: { method: "Compression method", mode: "Coefficient selection" },
};

export const OPTION_LABELS = {
  hist_eq: "Equalize", gamma: "Gamma", stretch: "Stretch", flatten: "Flatten light",
  auto: "Auto", bright: "Bright", dark: "Dark", median: "Median", box: "Box", gaussian: "Gaussian",
  laplacian_boost: "Laplacian", unsharp: "Unsharp", gradient: "Gradient", sobel: "Sobel", prewitt: "Prewitt",
  roberts: "Roberts", l2: "√(Gx²+Gy²)", l1: "|Gx|+|Gy|", magnitude: "Magnitude", gx: "Gx", gy: "Gy",
  otsu: "Otsu", region_growing: "Region growing",
  dct: "Whole-image DCT", hadamard: "Walsh–Hadamard", keep_percent: "Keep largest %", threshold: "Threshold", zonal: "Zonal",
  square: "Square", l_shape: "L-shape", jpeg: "JPEG-style (8×8 blocks)",
};

export const FINDINGS = {
  IMPULSE_NOISE: { title: "Salt & pepper noise", fix: { tool: "denoise", preset: "median3", label: "Remove specks" },
    // specks smeared over 2–3 px by resizing (screenshots): found at half scale, need a bigger median
    enlarged: { title: "Salt & pepper specks (enlarged)", fix: { tool: "denoise", preset: "median5", label: "Remove specks" } } },
  GAUSSIAN_NOISE: { title: "Grainy noise", fix: { tool: "denoise", preset: "gauss", label: "Smooth grain" },
    enlarged: { title: "Coarse grain (enlarged noise)", fix: { tool: "denoise", preset: "gauss_coarse", label: "Smooth grain" } } },
  UNEVEN_ILLUMINATION: { title: "Uneven lighting (possibly)" },
  LOW_SEPARABILITY: { title: "Object/background not clearly separated" },
  UNDEREXPOSED: { title: "Too dark", fix: { tool: "contrast", preset: "auto_gamma", label: "Auto brightness" } },
  OVEREXPOSED: { title: "Too bright", fix: { tool: "contrast", preset: "auto_gamma", label: "Auto brightness" } },
  CLIPPING: { title: "Clipped shadows or highlights" },
  LOW_CONTRAST: { title: "Low contrast", fix: { tool: "contrast", preset: "stretch", label: "Boost contrast" } },
};

/** The median preset that matches a speck finding's suggested window size. */
export const MEDIAN_PRESET = { 3: "median3", 5: "median5", 7: "median7" };

/** Shortcuts offered when the Health card found no noise but the user still sees some. */
export const MANUAL_NOISE_FIXES = [
  { tool: "denoise", preset: "median5", label: "Remove specks" },
  { tool: "denoise", preset: "gauss", label: "Smooth grain" },
];

/** Findings that only matter for segmentation. The health card hides them; Compare → Segment shows them under "Why?". */
export const SEGMENTATION_ONLY_FINDINGS = new Set(["UNEVEN_ILLUMINATION", "LOW_SEPARABILITY"]);

export const METRIC_LABELS = {
  mean_before: "Mean before", mean_after: "Mean after", std_before: "Contrast (σ) before", std_after: "Contrast (σ) after",
  gamma_used: "Gamma used", polarity_used: "Objects detected as",
  noise_sigma_before: "Noise σ before", noise_sigma_after: "Noise σ after",
  impulse_density_before: "Specks before", impulse_density_after: "Specks after", kernel_size_used: "Kernel size",
  speck_density_coarse_before: "Enlarged specks before", speck_density_coarse_after: "Enlarged specks after",
  kernel_centre: "Mask centre weight", mean_magnitude: "Mean gradient", edge_density: "Edge pixels",
  threshold: "Otsu threshold", eta: "Separability η", foreground_fraction: "Object area",
  n_components_raw: "Pieces before cleanup", n_components: "Pieces after cleanup",
  psnr_db: "PSNR", ssim: "SSIM", kept_percent: "Coefficients kept (non-zero)", compression_ratio: "Coefficient ratio",
  quality: "JPEG quality", chroma_subsampling: "Colour subsampling", match_with_real_jpeg_db: "Our decoder vs libjpeg",
  method_psnr_db: "PSNR before saving", jpeg_file_bytes: "JPEG file",
  energy_kept_percent: "Energy kept", coefficients_kept: "Coefficients kept (count)", coefficients_total: "Coefficients (total)",
  error_amplification: "Error map gain", size_original_png_bytes: "Original as PNG",
  size_compressed_png_bytes: "Result as PNG", size_compressed_jpeg_q90_bytes: "Result as JPEG q90",
};

/** The compression metrics shown in "Result details", in this order (those present for the method).
 *  File sizes are shown in their own card, from the real downloadable file. */
export const COMPRESS_DETAIL_KEYS = [
  "psnr_db", "ssim", "quality", "chroma_subsampling", "kept_percent", "compression_ratio",
  "energy_kept_percent", "method_psnr_db", "match_with_real_jpeg_db",
];

export function formatBytes(n) {
  if (typeof n !== "number") return "–";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

const PERCENT_KEYS = new Set(["impulse_density_before", "impulse_density_after", "speck_density_coarse_before",
  "speck_density_coarse_after", "edge_density", "foreground_fraction"]);
const ALREADY_PERCENT = new Set(["kept_percent", "energy_kept_percent"]);

export function formatMetric(key, v) {
  if (key === "psnr_db" || key === "method_psnr_db") return v === null || v === undefined ? "∞ (lossless)" : `${v.toFixed(2)} dB`;
  if (key === "match_with_real_jpeg_db" && typeof v === "number") return v >= 99 ? "identical" : `${v.toFixed(1)} dB`;
  if (key.endsWith("_bytes")) return formatBytes(v);
  if (key === "ssim" && typeof v === "number") return v.toFixed(3);
  if (key === "compression_ratio" && typeof v === "number") return `${v.toFixed(1)} : 1`;
  if (key === "error_amplification" && typeof v === "number") return `×${v}`;
  if (ALREADY_PERCENT.has(key) && typeof v === "number") return `${v < 1 ? v.toFixed(2) : v.toFixed(1)} %`;
  if (typeof v !== "number") return String(v);
  if (PERCENT_KEYS.has(key)) return `${(v * 100).toFixed(v < 0.01 ? 2 : 1)} %`;
  if (Number.isInteger(v)) return v.toLocaleString();
  return v.toFixed(Math.abs(v) < 10 ? 2 : 1);
}

export const INTENTS = [
  { id: "segment", label: "Segment", desc: "Otsu thresholding", icon: Scissors },
  { id: "edges", label: "Find edges", desc: "Sobel / Prewitt / Roberts", icon: ScanLine },
  { id: "compress", label: "Compression", desc: "DCT · Hadamard · JPEG", icon: FileArchive },
];

export const RULE_LABELS = {
  IMPULSE_NOISE: "Median filter (salt & pepper)",
  GAUSSIAN_NOISE: "Gaussian smoothing (grain)",
  UNEVEN_ILLUMINATION: "Background flattening (lighting)",
};

export const STAGE_LABELS = {
  "denoise:median": "Median filter",
  "denoise:gaussian": "Gaussian blur",
  "contrast:flatten": "Flatten lighting",
};

export const VERDICTS = {
  recommended: { tone: "good", title: "Fixing first clearly helps", text: "The guided pipeline measurably beats running the operation directly." },
  mixed: { tone: "warn", title: "Partly better", text: "Some checks improved, others did not. Compare the two results yourself." },
  no_benefit: { tone: "neutral", title: "Fixing first did not help here", text: "The diagnosis flagged something, but the prerequisite didn't improve the result, so the direct route is fine. This often happens with natural photos whose sky/ground look like 'uneven lighting'." },
  not_needed: { tone: "good", title: "Your image is ready for this", text: "No defect that would break this operation was found, so it runs directly." },
};
