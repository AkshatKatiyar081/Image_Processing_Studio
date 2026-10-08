import { useEffect, useRef, useState } from "react";
import { ArrowRight, CircleCheck, Download, WandSparkles } from "lucide-react";
import { EXTENSION, api, files } from "../lib/api.js";
import { download } from "../lib/image.js";
import { FINDINGS, toolById } from "../lib/labels.js";
import BeforeAfterView, { ViewToggle } from "./BeforeAfterView.jsx";
import Histogram from "./Histogram.jsx";
import { Button, Card, ErrorBanner, Spinner, cx } from "./ui.jsx";

const STEP_TITLE = {
  denoise: (p) => (p.filter_type === "median" ? `Median filter ${p.kernel_size}×${p.kernel_size}` : `Gaussian blur σ = ${p.sigma}`),
  contrast: (p) => (p.mode === "gamma" ? "Gamma correction" : p.mode === "stretch" ? "Contrast stretch" : p.mode),
  sharpen_edges: () => "Unsharp masking (mild)",
};

export default function AutoTab({ image, left, layout: Layout, beforeHist, onChain }) {
  const [result, setResult] = useState(null);
  const [view, setView] = useState("slider");   // "slider" | "side"
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(null);   // "download" | "chain" while the full-quality file is fetched
  const [notice, setNotice] = useState(null);
  const ctrl = useRef(null);

  useEffect(() => { setResult(null); setError(null); setNotice(null); ctrl.current?.abort(); }, [image?.id]);

  /** Download / Keep editing use the FULL-QUALITY result (the reply only has a preview). */
  const save = async (action) => {
    setSaving(action);
    setError(null);
    setNotice(null);
    try {
      const f = await files.auto(image.file, "opencv");
      if (f.note) setNotice(f.note);
      if (action === "chain") await onChain(f.file, "Auto Fix");
      else download(f.file, `${image.name}_autofix.${EXTENSION[f.format] || "png"}`);
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(null);
    }
  };

  const run = async () => {
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    setRunning(true);
    setError(null);
    try {
      const data = await api.auto(image.file, "opencv", c.signal);
      if (!c.signal.aborted) setResult(data);
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message);
    } finally {
      if (ctrl.current === c) setRunning(false);
    }
  };

  const sidebar = (
    <Card title="Auto Fix" icon={WandSparkles}>
      <p className="text-sm text-slate-600">One click: we measure the image, then apply only the fixes it needs, in the right order.</p>
      <ol className="mt-3 space-y-1 text-xs text-slate-500">
        <li>1. Remove salt & pepper specks (median; a bigger one for enlarged specks, e.g. screenshots)</li>
        <li>2. Smooth grain (Gaussian), also coarse grain from enlarged images</li>
        <li>3. Fix exposure (auto gamma)</li>
        <li>4. Stretch low contrast</li>
        <li>5. Mild sharpening, only if no noise was found at any scale</li>
      </ol>
      <Button className="mt-4 w-full" size="lg" onClick={run} loading={running} disabled={!image}>
        <WandSparkles className="h-4 w-4" />{image ? "Auto Fix my image" : "Load an image first"}
      </Button>
    </Card>
  );

  let main;
  if (!image) {
    main = <div className="flex min-h-[50vh] items-center justify-center rounded-xl border-2 border-dashed border-slate-200 bg-white p-8 text-center text-sm text-slate-500">Load an image, then press Auto Fix.</div>;
  } else {
    const before = result?.diagnosis_before, after = result?.diagnosis_after;
    main = (
      <div className="space-y-4">
        <Card>
          <div className="space-y-3">
            <BeforeAfterView view={view} before={image.url} after={result?.image} afterLabel="Auto fixed"
                             busy={running} overlay={<Spinner label="Analysing and fixing…" />} />
            <ErrorBanner error={error} onClose={() => setError(null)} />
            {notice && <p className="rounded-lg bg-sky-50 px-3 py-2 text-xs text-sky-800">{notice}</p>}
            {result && (
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <ViewToggle value={view} onChange={setView} />
                  <span className="text-xs text-slate-500">{view === "slider" ? "Drag the divider to compare · " : ""}{result.meta.total_ms} ms</span>
                </div>
                <div className="flex gap-2">
                  <Button size="sm" variant="secondary" onClick={() => save("chain")} loading={saving === "chain"} disabled={!!saving}>Keep editing this result</Button>
                  <Button size="sm" onClick={() => save("download")} loading={saving === "download"} disabled={!!saving}><Download className="h-4 w-4" />Download</Button>
                </div>
              </div>
            )}
            {!result && !running && <p className="text-center text-sm text-slate-500">Press <b>Auto Fix my image</b> to enhance it automatically.</p>}
          </div>
        </Card>

        {result && (
          <div className="grid gap-4 xl:grid-cols-2">
            <Card title="What was done">
              {result.steps.length === 0 ? (
                <p className="text-sm text-slate-500">Nothing needed fixing.</p>
              ) : (
                <ol className="space-y-2">
                  {result.steps.map((s, i) => (
                    <li key={i} className="flex gap-3">
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-brand-600 text-xs font-semibold text-white">{i + 1}</span>
                      <div className="text-sm">
                        <div className="font-medium text-slate-800">{STEP_TITLE[s.module]?.(s.params) ?? toolById(s.module)?.name}
                          {s.metrics?.gamma_used !== undefined && <span className="font-normal text-slate-500"> (γ = {s.metrics.gamma_used})</span>}
                        </div>
                        <div className="text-xs text-slate-500">because: {s.reason}</div>
                      </div>
                    </li>
                  ))}
                </ol>
              )}
            </Card>
            <Card title="Health before → after">
              <HealthDelta before={before} after={after} />
            </Card>
            <Card className="xl:col-span-2">
              <Histogram before={result.histograms.before || beforeHist} after={result.histograms.after} />
            </Card>
          </div>
        )}
      </div>
    );
  }

  return <Layout left={left} controls={sidebar} main={main} />;
}

function HealthDelta({ before, after }) {
  const rows = [
    ["Brightness", (d) => `${Math.round(d.metrics.mean * 100)} %`],
    ["Contrast σ", (d) => d.metrics.std.toFixed(0)],
    ["Noise σ", (d) => d.metrics.noise_sigma.toFixed(1)],
    ["Specks", (d) => `${(d.metrics.impulse_density * 100).toFixed(2)} %`],
    // specks enlarged by resizing (screenshots), measured at half scale
    ["Enlarged specks", (d) => `${((d.metrics.speck_density_coarse ?? 0) * 100).toFixed(2)} %`],
  ];
  const codesAfter = new Set(after.findings.map((f) => f.code));
  const fixed = before.findings.filter((f) => !codesAfter.has(f.code));
  return (
    <div className="space-y-3">
      <table className="w-full text-sm">
        <tbody>
          {rows.map(([label, get]) => (
            <tr key={label} className="border-b border-slate-100 last:border-0">
              <td className="py-1.5 text-slate-500">{label}</td>
              <td className="py-1.5 text-right tabular-nums">{get(before)}</td>
              <td className="px-2 text-slate-300"><ArrowRight className="h-3.5 w-3.5" /></td>
              <td className="py-1.5 font-semibold tabular-nums text-slate-800">{get(after)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {fixed.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {fixed.map((f) => (
            <span key={f.code} className={cx("inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700")}>
              <CircleCheck className="h-3 w-3" /> {FINDINGS[f.code]?.title || f.code} fixed
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
