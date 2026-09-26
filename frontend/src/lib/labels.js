// Friendly names, presets and explanations for everything the pipeline exposes.
import { Scissors, Sparkles, Sun, ScanLine } from "lucide-react";

export const TOOLS = [
  {
    id: "contrast",
    name: "Light & contrast",
    icon: Sun,
    blurb: "Brightness, contrast, histogram equalization, uneven lighting.",
    presets: [
      { id: "auto_gamma", label: "Auto brightness", hint: "Gamma chosen so the average becomes mid-gray", params: { mode: "gamma", gamma: "auto" } },
      { id: "stretch", label: "Boost contrast", hint: "Stretch intensities to use the full 0–255 range", params: { mode: "stretch", clip_percentile: 1 } },
      { id: "hist_eq", label: "Equalize histogram", hint: "Spread intensities evenly (Exp 04)", params: { mode: "hist_eq" } },
      { id: "flatten", label: "Fix uneven lighting", hint: "Remove the background shading with morphology", params: { mode: "flatten" } },
      { id: "brighten", label: "Brighten shadows", hint: "Gamma 0.6", params: { mode: "gamma", gamma: 0.6 } },
      { id: "darken", label: "Tone down highlights", hint: "Gamma 1.6", params: { mode: "gamma", gamma: 1.6 } },
    ],
  },
  {
    id: "denoise",
    name: "Remove noise",
    icon: Sparkles,
    blurb: "Median, box and Gaussian filters.",
    presets: [
      { id: "median3", label: "Salt & pepper specks", hint: "3×3 median filter (Exp 03)", params: { filter_type: "median", kernel_size: 3 } },
      { id: "median5", label: "Heavy specks", hint: "5×5 median filter", params: { filter_type: "median", kernel_size: 5 } },
      { id: "gauss", label: "Grainy / sensor noise", hint: "Gaussian blur, σ = 1", params: { filter_type: "gaussian", sigma: 1.0 } },
      { id: "box3", label: "Simple blur", hint: "3×3 averaging filter (Exp 03)", params: { filter_type: "box", kernel_size: 3 } },
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
      { id: "laplacian", label: "Laplacian sharpen", hint: "Centre-9 mask (Exp 03 Task 4)", params: { operation: "laplacian_boost", A: 1.0 } },
      { id: "sobel", label: "Edges — Sobel", hint: "Gradient magnitude (Exp 08)", params: { operation: "gradient", operator: "sobel", edge_threshold: 0 } },
      { id: "sobel_bw", label: "Edges — black & white", hint: "Sobel, thresholded at 20 %", params: { operation: "gradient", operator: "sobel", edge_threshold: 0.2 } },
      { id: "prewitt", label: "Edges — Prewitt", hint: "Gradient magnitude", params: { operation: "gradient", operator: "prewitt", edge_threshold: 0 } },
      { id: "roberts", label: "Edges — Roberts", hint: "2×2 diagonal differences", params: { operation: "gradient", operator: "roberts", edge_threshold: 0 } },
      { id: "laplace_edges", label: "Edges — Laplacian", hint: "Centre-8 high-pass mask (Exp 03 Task 3)", params: { operation: "laplacian_boost", A: 0 } },
    ],
  },
  {
    id: "segment",
    name: "Cut out object",
    icon: Scissors,
    blurb: "Otsu thresholding, region growing, morphological cleanup.",
    presets: [
      { id: "otsu", label: "Automatic (Otsu)", hint: "Best global threshold", params: { method: "otsu" } },
      { id: "main", label: "Main object only", hint: "Otsu + fill holes + keep largest", params: { method: "otsu", fill_holes: true, keep_largest: true } },
      { id: "grow", label: "Click to select", hint: "Region growing from the point you click (Exp 09)", params: { method: "region_growing" }, needsSeed: true },
    ],
  },
];

export const toolById = (id) => TOOLS.find((t) => t.id === id);

export const PARAM_LABELS = {
  mode: "Method", gamma: "Gamma (γ)", clip_percentile: "Clip each end (%)", se_radius: "Background disk radius (px)",
  object_polarity: "Objects are", filter_type: "Filter", kernel_size: "Window size", sigma: "Gaussian σ",
  operation: "Operation", A: "High-boost factor A", k: "Unsharp gain k", blur_sigma: "Blur σ", operator: "Operator",
  magnitude: "Magnitude", component: "Show", edge_threshold: "Edge threshold", method: "Method",
  foreground: "Object is", seed: "Seed point", delta_t: "Similarity tolerance ΔT", connectivity: "Connectivity",
  adaptive: "Adaptive (running mean)", morph_open: "Remove specks (opening)", morph_close: "Fill small holes (closing)",
  disk_radius: "Cleanup disk radius", fill_holes: "Fill enclosed holes", keep_largest: "Keep largest object only",
};

export const OPTION_LABELS = {
  hist_eq: "Equalize", gamma: "Gamma", stretch: "Stretch", flatten: "Flatten light",
  auto: "Auto", bright: "Bright", dark: "Dark", median: "Median", box: "Box", gaussian: "Gaussian",
  laplacian_boost: "Laplacian", unsharp: "Unsharp", gradient: "Gradient", sobel: "Sobel", prewitt: "Prewitt",
  roberts: "Roberts", l2: "√(Gx²+Gy²)", l1: "|Gx|+|Gy|", magnitude: "Magnitude", gx: "Gx", gy: "Gy",
  otsu: "Otsu", region_growing: "Region growing",
};

export const FINDINGS = {
  IMPULSE_NOISE: { title: "Salt & pepper noise", fix: { tool: "denoise", preset: "median3", label: "Remove specks" } },
  GAUSSIAN_NOISE: { title: "Grainy noise", fix: { tool: "denoise", preset: "gauss", label: "Smooth grain" } },
  UNEVEN_ILLUMINATION: { title: "Uneven lighting (possibly)", fix: { tool: "contrast", preset: "flatten", label: "Flatten lighting" } },
  LOW_SEPARABILITY: { title: "Object/background not clearly separated" },
  UNDEREXPOSED: { title: "Too dark", fix: { tool: "contrast", preset: "auto_gamma", label: "Auto brightness" } },
  OVEREXPOSED: { title: "Too bright", fix: { tool: "contrast", preset: "auto_gamma", label: "Auto brightness" } },
  CLIPPING: { title: "Clipped shadows or highlights" },
  LOW_CONTRAST: { title: "Low contrast", fix: { tool: "contrast", preset: "stretch", label: "Boost contrast" } },
};

export const METRIC_LABELS = {
  mean_before: "Mean before", mean_after: "Mean after", std_before: "Contrast (σ) before", std_after: "Contrast (σ) after",
  gamma_used: "Gamma used", polarity_used: "Objects detected as",
  noise_sigma_before: "Noise σ before", noise_sigma_after: "Noise σ after",
  impulse_density_before: "Specks before", impulse_density_after: "Specks after", kernel_size_used: "Kernel size",
  kernel_centre: "Mask centre weight", mean_magnitude: "Mean gradient", edge_density: "Edge pixels",
  threshold: "Otsu threshold", eta: "Separability η", foreground_fraction: "Object area",
  n_components_raw: "Pieces before cleanup", n_components: "Pieces after cleanup",
};

const PERCENT_KEYS = new Set(["impulse_density_before", "impulse_density_after", "edge_density", "foreground_fraction"]);

export function formatMetric(key, v) {
  if (typeof v !== "number") return String(v);
  if (PERCENT_KEYS.has(key)) return `${(v * 100).toFixed(v < 0.01 ? 2 : 1)} %`;
  if (Number.isInteger(v)) return v.toLocaleString();
  return v.toFixed(Math.abs(v) < 10 ? 2 : 1);
}

export const INTENTS = [
  { id: "segment", label: "Segment objects", desc: "Otsu thresholding", icon: Scissors },
  { id: "edges", label: "Find edges", desc: "Sobel / Prewitt / Roberts", icon: ScanLine },
  { id: "sharpen", label: "Sharpen", desc: "Unsharp masking", icon: Sparkles },
  { id: "equalize", label: "Equalize", desc: "Histogram equalization", icon: Sun },
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
