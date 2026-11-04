import { useState } from "react";
import { Activity, ChevronDown, CircleCheck, Info, LoaderCircle, TriangleAlert } from "lucide-react";
import { FINDINGS, MANUAL_NOISE_FIXES, MEDIAN_PRESET, SEGMENTATION_ONLY_FINDINGS } from "../lib/labels.js";
import { Card, Stat, cx } from "./ui.jsx";

const NOISE_CODES = new Set(["IMPULSE_NOISE", "GAUSSIAN_NOISE"]);

/** Title and fix for a finding; enlarged noise (screenshots) has its own wording and a bigger filter. */
function findingMeta(f) {
  const base = FINDINGS[f.code] || { title: f.code };
  const meta = f.scale === "enlarged" && base.enlarged ? base.enlarged : base;
  if (f.code === "IMPULSE_NOISE" && meta.fix && MEDIAN_PRESET[f.median_ksize]) {
    return { ...meta, fix: { ...meta.fix, preset: MEDIAN_PRESET[f.median_ksize] } };
  }
  return meta;
}

function Finding({ f, onFix }) {
  const [open, setOpen] = useState(false);
  const meta = findingMeta(f);
  const warn = f.severity === "warning";
  const Icon = warn ? TriangleAlert : Info;
  return (
    <li className={cx("rounded-lg border px-3 py-2", warn ? "border-amber-200 bg-amber-50" : "border-slate-200 bg-slate-50")}>
      <div className="flex items-center gap-2">
        <Icon className={cx("h-4 w-4 shrink-0", warn ? "text-amber-600" : "text-slate-400")} />
        <button className="flex flex-1 items-center gap-1 text-left text-sm font-medium text-slate-700" onClick={() => setOpen(!open)} aria-expanded={open}>
          {meta.title}
          <ChevronDown className={cx("h-3.5 w-3.5 text-slate-400 transition", open && "rotate-180")} />
        </button>
        {meta.fix && onFix && (
          <button onClick={() => onFix(meta.fix.tool, meta.fix.preset)}
                  className="rounded-md bg-white px-2 py-0.5 text-xs font-medium text-brand-700 ring-1 ring-brand-100 hover:bg-brand-50">
            {meta.fix.label}
          </button>
        )}
      </div>
      {open && <p className="mt-1.5 pl-6 text-xs leading-relaxed text-slate-600">{f.message}</p>}
    </li>
  );
}

/** "Image health": the pipeline's diagnosis in plain words, with one-click fixes. */
export default function HealthCard({ health, loading, error, onFix }) {
  const m = health?.metrics;
  // Lighting/separability findings only matter for segmentation: they are shown in Compare → Segment instead.
  const findings = (health?.findings || []).filter((f) => !SEGMENTATION_ONLY_FINDINGS.has(f.code));
  return (
    <Card title="Image health" icon={Activity} tour="health"
          action={loading && <LoaderCircle className="h-4 w-4 animate-spin text-slate-400" />}>
      {error && <p className="text-sm text-red-700">{error}</p>}
      {!health && !error && <p className="text-sm text-slate-500">{loading ? "Checking the image…" : "Load an image to check it for noise, exposure and contrast problems."}</p>}
      {health && (
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-2">
            <Stat label="Brightness" value={`${Math.round(m.mean * 100)} %`} />
            <Stat label="Contrast σ" value={m.std.toFixed(0)} />
            <Stat label="Noise σ" value={m.noise_sigma.toFixed(1)} />
          </div>
          {findings.length === 0 ? (
            <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-800">
              <CircleCheck className="h-4 w-4" /> Looks healthy: no noise, exposure or contrast problems found.
            </div>
          ) : (
            <ul className="space-y-1.5">
              {[...findings]
                .sort((a, b) => (a.severity === "warning" ? -1 : 0) - (b.severity === "warning" ? -1 : 0))
                .map((f) => <Finding key={f.code} f={f} onFix={onFix} />)}
            </ul>
          )}
          {onFix && !findings.some((f) => NOISE_CODES.has(f.code)) && (
            // No detector is perfect: let the user act on what they see.
            <div className="flex flex-wrap items-center gap-1.5 text-xs text-slate-500">
              <span>Still looks noisy to you?</span>
              {MANUAL_NOISE_FIXES.map((x) => (
                <button key={x.preset} onClick={() => onFix(x.tool, x.preset)}
                        className="rounded-md bg-white px-2 py-0.5 font-medium text-brand-700 ring-1 ring-brand-100 hover:bg-brand-50">
                  {x.label}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
