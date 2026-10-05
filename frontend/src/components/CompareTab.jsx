import { useEffect, useRef, useState } from "react";
import { ArrowRight, CircleCheck, Download, GitCompare, Info, Maximize2, TriangleAlert, X } from "lucide-react";
import { api } from "../lib/api.js";
import { download, extFor } from "../lib/image.js";
import { INTENTS, OPTION_LABELS, STAGE_LABELS, VERDICTS, formatBytes } from "../lib/labels.js";
import CompareSlider from "./CompareSlider.jsx";
import LineChart from "./charts/LineChart.jsx";
import PairedBars from "./charts/PairedBars.jsx";
import ZoomLens from "./ZoomLens.jsx";
import { Button, Card, ErrorBanner, Segmented, Spinner, cx } from "./ui.jsx";

const TARGET_NAME = { segment: "Otsu", edges: "Edges" };
const KEEP_CHIPS = [2, 5, 10, 25];
const QUALITY_CHIPS = [25, 50, 75, 90];
const C = {
  direct: "var(--series-direct)", guided: "var(--series-guided)",
  dct: "var(--series-dct)", hadamard: "var(--series-hadamard)", jpeg: "var(--series-jpeg)",
};

function paramsFor(intent, operator) {
  if (intent === "segment") return { method: "otsu" };
  return { operator, edge_threshold: 0.2 };
}

const fmt = (v) => (Math.abs(v) < 1 && v !== 0 ? v.toFixed(3) : Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2));
const fmtPct = (v) => `${+v.toPrecision(2)}`;

export default function CompareTab({ image, left, layout: Layout }) {
  const [intent, setIntent] = useState("segment");
  const [operator, setOperator] = useState("sobel");
  const [keepPercent, setKeepPercent] = useState(10);
  const [jpegAuto, setJpegAuto] = useState(true);       // JPEG-style quality: match DCT's PSNR …
  const [jpegQuality, setJpegQuality] = useState(75);   // … or this quality
  const [result, setResult] = useState(null);     // guided result (segment / edges)
  const [cresult, setCresult] = useState(null);   // compression comparison
  const [sweep, setSweep] = useState({ data: null, loading: false, error: null });
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [display, setDisplay] = useState("slider");
  const [maskView, setMaskView] = useState(false);
  const ctrl = useRef(null);
  const sweepCtrl = useRef(null);

  const clear = () => {
    ctrl.current?.abort();
    sweepCtrl.current?.abort();
    setResult(null);
    setCresult(null);
    setSweep({ data: null, loading: false, error: null });
    setError(null);
  };

  useEffect(() => { clear(); setRunning(false); }, [image?.id]); // eslint-disable-line

  const runSweep = async (it, params) => {
    sweepCtrl.current?.abort();
    const c = new AbortController();
    sweepCtrl.current = c;
    setSweep({ data: null, loading: true, error: null });
    try {
      const data = await api.sweep(image.file, it, params, "opencv", c.signal);
      if (!c.signal.aborted) setSweep({ data, loading: false, error: null });
    } catch (e) {
      if (e.name !== "AbortError") setSweep({ data: null, loading: false, error: e.message });
    }
  };

  const run = async (kp = keepPercent, jq = jpegAuto ? null : jpegQuality) => {
    if (!image) return;
    clear();
    const c = new AbortController();
    ctrl.current = c;
    setRunning(true);
    try {
      if (intent === "compress") {
        const data = await api.compareCompression(image.file, kp, jq, "opencv", c.signal);
        if (!c.signal.aborted) setCresult(data);
      } else {
        const params = paramsFor(intent, operator);
        const data = await api.guided(image.file, intent, params, "opencv", c.signal);
        if (!c.signal.aborted) {
          setResult(data);
          runSweep(intent, params);           // in the background, with its own loading state
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message);
    } finally {
      if (ctrl.current === c) setRunning(false);
    }
  };

  const chooseIntent = (id) => { if (id !== intent) { setIntent(id); clear(); } };
  const chooseChip = (v) => { setKeepPercent(v); if (cresult) run(v); };
  const chooseQualityChip = (q) => { setJpegAuto(false); setJpegQuality(q); if (cresult) run(keepPercent, q); };
  const toggleJpegAuto = (on) => { setJpegAuto(on); if (cresult) run(keepPercent, on ? null : jpegQuality); };

  const sidebar = (
    <Card title="What do you want to do?" icon={GitCompare}>
      <p className="mb-3 text-xs leading-relaxed text-slate-500">
        {intent === "compress" ? (
          <>DCT and Hadamard keep the <b className="text-slate-700">same share of coefficients</b> (equal budget, different
            basis). <b className="text-slate-700">JPEG-style</b> then runs at the quality that matches DCT's result, so you
            can see how much smaller the real file is.</>
        ) : (
          <>We check your image for problems that break this operation, then run it twice:
            <b className="text-slate-700"> directly</b>, and <b className="text-slate-700">after fixing the problem first</b>, and measure which is better.</>
        )}
      </p>
      <div className="grid grid-cols-3 gap-2">
        {INTENTS.map((it) => (
          <button key={it.id} onClick={() => chooseIntent(it.id)} aria-pressed={intent === it.id}
                  className={cx("flex min-w-0 flex-col items-start gap-1 rounded-lg border px-2.5 py-2.5 text-left transition",
                    intent === it.id ? "border-brand-500 bg-brand-50 ring-1 ring-brand-500" : "border-slate-200 hover:bg-slate-50")}>
            <it.icon className={cx("h-4 w-4", intent === it.id ? "text-brand-600" : "text-slate-400")} />
            <span className="text-sm font-semibold text-slate-800">{it.label}</span>
            <span className="text-[11px] leading-tight text-slate-500">{it.desc}</span>
          </button>
        ))}
      </div>
      {intent === "edges" && (
        <div className="mt-3">
          <div className="mb-1 text-xs font-semibold text-slate-600">Edge operator</div>
          <Segmented size="sm" value={operator} onChange={(v) => { setOperator(v); clear(); }}
                     options={["sobel", "prewitt", "roberts"].map((o) => ({ value: o, label: OPTION_LABELS[o] }))} />
          <p className="mt-1 text-[11px] text-slate-400">Roberts has no built-in smoothing, so it needs a fix at lower noise levels than Sobel.</p>
        </div>
      )}
      {intent === "compress" && (
        <div className="mt-3">
          <div className="mb-1 flex items-baseline justify-between text-xs">
            <span className="font-semibold text-slate-600">Keep % of coefficients</span>
            <span className="font-semibold text-slate-700 tabular-nums">{keepPercent} %</span>
          </div>
          <input type="range" min="0.5" max="50" step="0.5" value={keepPercent} aria-label="Keep % of coefficients"
                 className="w-full" onChange={(e) => setKeepPercent(parseFloat(e.target.value))} />
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {KEEP_CHIPS.map((v) => (
              <button key={v} onClick={() => chooseChip(v)}
                      className={cx("rounded-full px-2.5 py-0.5 text-xs font-medium ring-1",
                        keepPercent === v ? "bg-brand-600 text-white ring-brand-600" : "bg-white text-slate-600 ring-slate-200 hover:bg-slate-50")}>
                {v} %
              </button>
            ))}
          </div>

          <div className="mt-4 mb-1 flex items-baseline justify-between text-xs">
            <span className="font-semibold text-slate-600">JPEG-style quality</span>
            <span className="font-semibold text-slate-700 tabular-nums">
              {jpegAuto ? (cresult?.summary?.jpeg?.matched_to_dct ? `auto (${cresult.summary.jpeg.quality})` : "auto") : jpegQuality}
            </span>
          </div>
          <input type="range" min="1" max="100" step="1" value={jpegAuto ? (cresult?.summary?.jpeg?.quality ?? jpegQuality) : jpegQuality}
                 aria-label="JPEG-style quality" disabled={jpegAuto} className="w-full disabled:opacity-50"
                 onChange={(e) => setJpegQuality(parseInt(e.target.value, 10))} />
          <label className="mt-1 flex items-center gap-1.5 text-xs text-slate-600">
            <input type="checkbox" checked={jpegAuto} onChange={(e) => toggleJpegAuto(e.target.checked)} />
            Auto: match DCT's image quality
          </label>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {QUALITY_CHIPS.map((q) => (
              <button key={q} onClick={() => chooseQualityChip(q)}
                      className={cx("rounded-full px-2.5 py-0.5 text-xs font-medium ring-1",
                        !jpegAuto && jpegQuality === q ? "bg-brand-600 text-white ring-brand-600" : "bg-white text-slate-600 ring-slate-200 hover:bg-slate-50")}>
                q{q}
              </button>
            ))}
          </div>
          <p className="mt-1 text-[11px] text-slate-400">Lower quality = coarser quantisation steps = smaller file.</p>
        </div>
      )}
      <Button className="mt-4 w-full" size="lg" onClick={() => run()} loading={running} disabled={!image}>
        {image ? "Run comparison" : "Load an image first"}
      </Button>
    </Card>
  );

  let main;
  if (!image) {
    main = <Placeholder text="Load an image, choose what you want to do, and press Run comparison." />;
  } else if (running && !result && !cresult) {
    main = <Card><div className="py-24"><Spinner label={intent === "compress" ? "Transforming with DCT and Hadamard…" : "Diagnosing and running both routes…"} /></div></Card>;
  } else if (intent === "compress" && cresult) {
    main = <CompressionResults r={cresult} image={image} error={error} onCloseError={() => setError(null)} />;
  } else if (intent !== "compress" && result) {
    main = (
      <GuidedResults result={result} sweep={sweep} image={image} display={display} setDisplay={setDisplay}
                     maskView={maskView} setMaskView={setMaskView} running={running}
                     error={error} onCloseError={() => setError(null)} />
    );
  } else {
    const label = INTENTS.find((i) => i.id === intent).label;
    main = (
      <div className="space-y-3">
        <ErrorBanner error={error} onClose={() => setError(null)} />
        <Placeholder img={image.url} text={intent === "compress"
          ? `Press "Run comparison" to compress the image with DCT and with Walsh–Hadamard at ${keepPercent} % of coefficients.`
          : `Press "Run comparison" to see the direct result next to the guided (fix-first) result for “${label}”.`} />
      </div>
    );
  }

  return <Layout left={left} controls={sidebar} main={main} />;
}

// =============================================================== Segment / Find edges
function GuidedResults({ result, sweep, image, display, setDisplay, maskView, setMaskView, running, error, onCloseError }) {
  const v = VERDICTS[result.verdict.result];
  const a = result.branch_a, b = result.branch_b;
  const pick = (branch) => (branch && result.intent === "segment" && maskView ? branch.extras.mask : branch?.image);
  const aUrl = pick(a), bUrl = pick(b);
  return (
    <div className="space-y-4">
      <ErrorBanner error={error} onClose={onCloseError} />
      <Verdict v={v} checks={result.verdict.checks} />
      <SweepCard sweep={sweep} />
      {result.difference_map && <DifferenceCard url={result.difference_map} legend={result.difference_legend} intent={result.intent} />}
      {result.comparison && (
        <Card title="Direct vs Guided, measured">
          <PairedBars series={[{ label: "A · Direct", color: C.direct }, { label: "B · Guided", color: C.guided }]}
                      rows={guidedRows(result.intent, result.comparison)} />
        </Card>
      )}

      <Card>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            {b && <Segmented size="sm" value={display} onChange={setDisplay}
                             options={[{ value: "slider", label: "Slider" }, { value: "side", label: "Side by side" }]} />}
            {result.intent === "segment" && (
              <Segmented size="sm" value={maskView ? "mask" : "overlay"} onChange={(x) => setMaskView(x === "mask")}
                         options={[{ value: "overlay", label: "Overlay" }, { value: "mask", label: "Mask" }]} />
            )}
          </div>
          <div className="flex gap-2">
            <Button size="sm" variant="secondary" onClick={() => download(aUrl, `${image.name}_direct.${extFor(aUrl)}`)}><Download className="h-3.5 w-3.5" />Direct</Button>
            {b && <Button size="sm" variant="secondary" onClick={() => download(bUrl, `${image.name}_guided.${extFor(bUrl)}`)}><Download className="h-3.5 w-3.5" />Guided</Button>}
          </div>
        </div>
        {!b ? (
          <CompareSlider before={result.input || image.url} after={aUrl} beforeLabel="Input" afterLabel="Result" busy={running} overlay={<Spinner />} />
        ) : display === "slider" ? (
          <>
            <CompareSlider before={aUrl} after={bUrl} beforeLabel="A · Direct" afterLabel="B · Guided" busy={running} overlay={<Spinner />} />
            <p className="mt-2 text-center text-xs text-slate-500">Left of the divider: direct result. Right: result after fixing first.</p>
          </>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2">
            <Figure url={aUrl} title="A · Direct" sub={`${TARGET_NAME[result.intent]} on the image as it is`} />
            <Figure url={bUrl} title="B · Guided" sub={b.stages.map((s) => STAGE_LABELS[s.name] || s.name).join(" → ") + ` → ${TARGET_NAME[result.intent]}`} highlight />
          </div>
        )}
      </Card>

      {b && <PipelineStrip input={result.input || image.url} stages={b.stages} target={TARGET_NAME[result.intent]} />}
      <WhyCard findings={result.diagnosis.findings} stages={b?.stages || []} />
    </div>
  );
}

function guidedRows(intent, c) {
  const n = (v) => v.toLocaleString();
  if (intent === "edges") {
    return [
      { label: "False edges in flat areas", a: c.false_edge_density_a * 100, b: c.false_edge_density_b * 100,
        format: (v) => `${v.toFixed(2)} %`, better: "lower" },
      { label: "Edge pixels found only by this route", a: c.only_a_pixels, b: c.only_b_pixels, format: n,
        hint: "red / green in the map" },
      { label: "Edge pixels in total", a: c.edge_pixels_a, b: c.edge_pixels_b, format: n },
    ];
  }
  return [
    { label: "Pieces before cleanup (islands + holes)", a: c.fragmentation_a, b: c.fragmentation_b, format: n, better: "lower" },
    { label: "Separability η", a: c.eta_a, b: c.eta_b, format: (v) => v.toFixed(3), better: "higher" },
    { label: "Objects after cleanup", a: c.n_components_a, b: c.n_components_b, format: n },
    { label: "Pixels marked as object only by this route", a: c.only_a_pixels, b: c.only_b_pixels, format: n,
      hint: "red / green in the map" },
  ];
}

const AXIS_NOUN = { illumination: "uneven lighting", impulse: "salt & pepper noise", gaussian: "noise" };

function SweepCard({ sweep }) {
  const d = sweep.data;
  const title = d
    ? `${d.metric === "IoU" ? "Segmentation" : "Edge"} accuracy vs ${AXIS_NOUN[d.axis.name] || d.axis.label}`
    : "Accuracy as the defect gets worse";
  return (
    <Card title={title}>
      {sweep.loading && <div className="py-10"><Spinner label="Adding noise at fixed levels and re-running both routes…" /></div>}
      {sweep.error && <p className="text-sm text-red-700">Could not compute the chart: {sweep.error}</p>}
      {d && (() => {
        const unit = d.axis.unit ? ` (${d.axis.unit})` : "";
        const fx = (v) => (d.axis.name === "illumination" ? v.toFixed(2) : v.toFixed(1));
        const series = [
          { id: "direct", label: "Direct", color: C.direct, points: d.points.map((p) => ({ x: p.x, y: p.direct })) },
          { id: "guided", label: "Guided", color: C.guided, points: d.points.map((p) => ({ x: p.x, y: p.guided })) },
        ];
        return (
          <>
            <LineChart series={series} xLabel={`${d.axis.label}${unit}`} yLabel={d.metric} yDomain={[0, 1]}
                       marker={{ x: d.points[d.user_point_index].x, label: "Your image" }}
                       formatX={fx} formatY={(v) => v.toFixed(2)} ariaLabel={`${d.metric} versus ${d.axis.label}, direct and guided`} />
            <p className="mt-2 text-[11px] leading-snug text-slate-500">
              {d.reference_note} The first point is your image; we then add more {AXIS_NOUN[d.axis.name] || "degradation"} at fixed
              levels ({d.axis.level_label}: {d.points.map((p) => p.level).join(", ")}) and run both routes again.
              {d.sweep_scale < 1 && " Computed on a ≤ 512 px copy for speed."}
            </p>
          </>
        );
      })()}
    </Card>
  );
}

function DifferenceCard({ url, legend, intent }) {
  const [open, setOpen] = useState(false);
  const [zoom, setZoom] = useState(1);
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);
  const chips = legend.filter((l) => l.color);
  const note = legend.find((l) => l.note)?.note;
  return (
    <Card title={intent === "edges" ? "Where the edge maps differ" : "Where the masks differ"}>
      <button onClick={() => { setZoom(1); setOpen(true); }} className="group relative block w-full overflow-hidden rounded-md border border-slate-200 bg-black" aria-label="Open the difference map larger">
        <img src={url} alt="Difference map between the direct and guided results" className="pixelated mx-auto max-h-[60vh] w-auto max-w-full" />
        <span className="absolute top-2 right-2 rounded bg-black/60 p-1 text-white opacity-80 group-hover:opacity-100"><Maximize2 className="h-4 w-4" /></span>
      </button>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {chips.map((l) => (
          <span key={l.label} className="inline-flex items-center gap-1.5 rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-700">
            <span className="h-2.5 w-2.5 rounded-sm ring-1 ring-black/20" style={{ background: l.color }} />{l.label}
          </span>
        ))}
      </div>
      {note && <p className="mt-1.5 text-[11px] text-slate-500">{note}</p>}
      {open && (
        <div className="fixed inset-0 z-50 flex flex-col bg-black/80 p-3" role="dialog" aria-modal="true" aria-label="Difference map" onClick={() => setOpen(false)}>
          <div className="mb-2 flex items-center justify-end gap-2" onClick={(e) => e.stopPropagation()}>
            <Segmented size="sm" value={zoom} onChange={setZoom} options={[1, 2, 4].map((z) => ({ value: z, label: `${z}×` }))} />
            <button onClick={() => setOpen(false)} className="rounded bg-white/90 p-1.5 text-slate-700" aria-label="Close"><X className="h-4 w-4" /></button>
          </div>
          <div className="min-h-0 flex-1 overflow-auto" onClick={(e) => e.stopPropagation()}>
            <img src={url} alt="Difference map, enlarged" className="pixelated mx-auto block max-w-none origin-top-left"
                 style={{ width: zoom === 1 ? "min(100%, 1024px)" : `${zoom * 100}%` }} />
          </div>
        </div>
      )}
    </Card>
  );
}

// =============================================================== Compression
function CompressionResults({ r, image, error, onCloseError }) {
  const [rdMetric, setRdMetric] = useState("psnr_db");
  const [view, setView] = useState("slider");
  const kp = r.keep_percent;
  const d = r.dct, h = r.hadamard, j = r.jpeg;
  const s = r.summary;
  const jq = j?.metrics.quality;

  const rdSeries = [
    { id: "dct", label: "DCT", color: C.dct, points: r.rd_curve.map((p) => ({ x: p.keep_percent, y: p.dct[rdMetric] })) },
    { id: "hadamard", label: "Hadamard", color: C.hadamard, points: r.rd_curve.map((p) => ({ x: p.keep_percent, y: p.hadamard[rdMetric] })) },
  ];
  const energySeries = [
    { id: "dct", label: "DCT", color: C.dct, points: r.energy_curve.map((p) => ({ x: p.percent, y: p.dct })) },
    { id: "hadamard", label: "Hadamard", color: C.hadamard, points: r.energy_curve.map((p) => ({ x: p.percent, y: p.hadamard })) },
  ];
  const eMin = Math.min(...r.energy_curve.flatMap((p) => [p.dct, p.hadamard]));
  const eLo = Math.max(0, Math.floor(eMin / 5) * 5);

  const pct = (v) => `${v.toFixed(2)} %`;
  const all = j ? [d, h, j] : [d, h];
  const vals = (key) => all.map((x) => x.metrics[key]);
  const bars = [
    { label: "File size (real file)", values: vals("file_bytes"), format: formatBytes, better: "lower",
      hint: "DCT / Hadamard saved as near-lossless JPEG" },
    { label: "PSNR", values: vals("psnr_db"), format: (v) => `${v.toFixed(2)} dB`, better: "higher" },
    { label: "SSIM", values: vals("ssim"), format: (v) => v.toFixed(3), better: "higher" },
    { label: "Coefficients kept (non-zero)", values: vals("kept_percent"), format: pct,
      hint: "JPEG-style: left after quantisation" },
    { label: "Energy kept", values: vals("energy_kept_percent"), format: pct },
  ];
  const barSeries = [{ label: "DCT", color: C.dct }, { label: "Walsh–Hadamard", color: C.hadamard }]
    .concat(j ? [{ label: `JPEG-style (q${jq})`, color: C.jpeg }] : []);

  return (
    <div className="space-y-4">
      <ErrorBanner error={error} onClose={onCloseError} />
      <SummaryBanner s={s} />
      {r.size_curve && <SizeCurveCard r={r} />}

      <Card title="Quality vs compression">
        <div className="mb-2 flex justify-end">
          <Segmented size="sm" value={rdMetric} onChange={setRdMetric}
                     options={[{ value: "psnr_db", label: "PSNR" }, { value: "ssim", label: "SSIM" }]} />
        </div>
        <LineChart series={rdSeries} xScale="log" xLabel="% of coefficients kept (log scale)"
                   yLabel={rdMetric === "psnr_db" ? "PSNR (dB)" : "SSIM"}
                   yDomain={rdMetric === "ssim" ? [0, 1] : undefined}
                   marker={{ x: kp, label: "Your setting" }} formatX={fmtPct}
                   formatY={(v) => (rdMetric === "ssim" ? v.toFixed(2) : v.toFixed(0))}
                   ariaLabel={`${rdMetric === "ssim" ? "SSIM" : "PSNR"} versus percent of coefficients kept, DCT and Hadamard`} />
        <p className="mt-2 text-[11px] text-slate-500">Higher is better. Each point keeps the same share of coefficients for both transforms.</p>
      </Card>

      <Card title="Energy compaction">
        <LineChart series={energySeries} xScale="log" xLabel="% of coefficients kept (largest first, log scale)"
                   yLabel="% of energy captured" yDomain={[eLo, 100]} marker={{ x: kp, label: "Your setting" }} showPoints={false}
                   formatX={fmtPct} formatY={(v) => v.toFixed(0)}
                   ariaLabel="Percent of energy captured versus percent of coefficients kept, DCT and Hadamard" />
        <p className="mt-2 text-[11px] leading-snug text-slate-500">
          The DC term (average brightness) alone holds most of the energy, so both curves start high. The gap shows
          how much more of the remaining detail each basis packs into its largest coefficients.
        </p>
      </Card>

      <Card>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <Segmented size="sm" value={view} onChange={setView}
                     options={[{ value: "slider", label: "DCT vs Hadamard" },
                       ...(j ? [{ value: "jpeg", label: "DCT vs JPEG-style" }] : []),
                       { value: "side", label: "Side by side" }]} />
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="secondary" onClick={() => download(d.image, `${image.name}_dct_${kp}pct.png`)}><Download className="h-3.5 w-3.5" />DCT</Button>
            <Button size="sm" variant="secondary" onClick={() => download(h.image, `${image.name}_hadamard_${kp}pct.png`)}><Download className="h-3.5 w-3.5" />Hadamard</Button>
            {j && <Button size="sm" variant="secondary" onClick={() => download(j.image, `${image.name}_jpeg_q${jq}.png`)}><Download className="h-3.5 w-3.5" />JPEG-style</Button>}
          </div>
        </div>
        {view === "slider" && (
          <>
            <CompareSlider before={h.image} after={d.image} beforeLabel="Hadamard" afterLabel="DCT" />
            <p className="mt-2 text-center text-xs text-slate-500">Left of the divider: Walsh–Hadamard. Right: DCT. Both keep {kp} % of coefficients.</p>
          </>
        )}
        {view === "jpeg" && j && (
          <>
            <CompareSlider before={d.image} after={j.image} beforeLabel={`DCT ${kp} % · ${formatBytes(d.metrics.file_bytes)}`}
                           afterLabel={`JPEG-style q${jq} · ${formatBytes(j.metrics.file_bytes)}`} />
            <p className="mt-2 text-center text-xs text-slate-500">
              Left: whole-image DCT keeping {kp} %. Right: JPEG-style at quality {jq}
              {s.jpeg?.matched_to_dct === false ? " (your choice)." : ", chosen to match its PSNR. Same quality, very different file size."}
            </p>
          </>
        )}
        {view === "side" && (
          <div className={cx("grid gap-3", j ? "sm:grid-cols-3" : "sm:grid-cols-2")}>
            <Figure url={d.image} title="DCT" sub={`PSNR ${d.metrics.psnr_db?.toFixed(2) ?? "∞"} dB · ${formatBytes(d.metrics.file_bytes)}`} highlight />
            <Figure url={h.image} title="Walsh–Hadamard" sub={`PSNR ${h.metrics.psnr_db?.toFixed(2) ?? "∞"} dB · ${formatBytes(h.metrics.file_bytes)}`} />
            {j && <Figure url={j.image} title={`JPEG-style (q${jq})`} sub={`PSNR ${j.metrics.psnr_db?.toFixed(2) ?? "∞"} dB · ${formatBytes(j.metrics.file_bytes)}`} />}
          </div>
        )}
        {[...(d.warnings || []), ...(h.warnings || [])].map((w) => (
          <p key={w} className="mt-2 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">{w}</p>
        ))}
      </Card>

      <Card title="Errors, magnified">
        <ZoomLens columns={j ? 3 : 2} items={[
          { url: d.extras.error_map, label: "DCT error", caption: "ringing near edges" },
          { url: h.extras.error_map, label: "Hadamard error", caption: "blocky steps" },
          ...(j ? [{ url: j.extras.error_map, label: `JPEG-style error (q${jq})`, caption: "spread thinly, inside 8×8 blocks" }] : []),
          { url: d.image, label: "DCT result" },
          { url: h.image, label: "Hadamard result" },
          ...(j ? [{ url: j.image, label: "JPEG-style result" }] : []),
        ]} />
        <p className="mt-1 text-[11px] text-slate-500">
          Error maps: |original − result|, brighter = bigger error, all ×{d.metrics.error_amplification} amplified (same scale).
        </p>
      </Card>

      <Card title="Coefficients (log magnitude, luminance)">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MapFigure url={d.extras.coef_before} label="DCT · all" />
          <MapFigure url={d.extras.coef_after} label="DCT · kept" />
          <MapFigure url={h.extras.coef_before} label="Hadamard · all" />
          <MapFigure url={h.extras.coef_after} label="Hadamard · kept" />
          {j && <MapFigure url={j.extras.coef_before} label="JPEG-style · 8×8 blocks" />}
          {j && <MapFigure url={j.extras.coef_after} label="JPEG-style · quantised" />}
        </div>
        <p className="mt-2 text-[11px] leading-snug text-slate-500">
          Low frequencies (DCT) and low sequencies (Hadamard) are at the top-left. The DCT's energy sits in a tighter
          corner, which is why fewer of its coefficients describe the image well.
          {j && " JPEG-style maps group each 8×8 block's frequencies: the top-left tile is the DC (a small copy of the image); after quantisation most finer tiles are 0."}
        </p>
      </Card>

      <Card title={j ? "DCT vs Hadamard vs JPEG-style, measured" : "DCT vs Hadamard, measured"}>
        <PairedBars series={barSeries} rows={bars} />
        <p className="mt-2 text-[11px] leading-snug text-slate-500">
          DCT and Hadamard keep the same share of coefficients ({kp} %); their results are saved as near-lossless JPEGs
          to get a real file size. JPEG-style runs at the quality (q{jq}) that matches the DCT result's PSNR, so its file
          size is the honest answer to "how small can this quality be stored?".
        </p>
      </Card>

      {r.curve_scale < 1 && (
        <p className="text-center text-[11px] text-slate-400">The curves are computed on a ≤ 512 px copy of the image for speed; the images and bars use the full working size.</p>
      )}
    </div>
  );
}

/** Real files: quality against bits per pixel for JPEG-style coding and the saved whole-image results. */
function SizeCurveCard({ r }) {
  const [metric, setMetric] = useState("psnr_db");
  const sc = r.size_curve;
  const jv = r.summary.jpeg_vs_dct;
  const pts = (list) => list.map((p) => ({ x: p.bpp, y: p[metric] }));
  const series = [
    { id: "jpeg", label: "JPEG-style", color: "var(--series-jpeg)", points: pts(sc.jpeg) },
    { id: "dct", label: "DCT", color: C.dct, points: pts(sc.dct) },
    { id: "hadamard", label: "Hadamard", color: C.hadamard, points: pts(sc.hadamard) },
  ];
  return (
    <Card title="Quality vs file size (real files)">
      <div className="mb-2 flex justify-end">
        <Segmented size="sm" value={metric} onChange={setMetric}
                   options={[{ value: "psnr_db", label: "PSNR" }, { value: "ssim", label: "SSIM" }]} />
      </div>
      <LineChart series={series} xScale="log" xLabel="File size (bits per pixel, log scale)"
                 yLabel={metric === "psnr_db" ? "PSNR (dB)" : "SSIM"} yDomain={metric === "ssim" ? [0, 1] : undefined}
                 marker={{ x: jv.dct_bpp, label: `DCT at ${r.keep_percent} %` }}
                 formatX={(v) => `${+v.toPrecision(2)}`} formatY={(v) => (metric === "ssim" ? v.toFixed(2) : v.toFixed(0))}
                 endLabels={false}
                 ariaLabel="Quality versus file size for JPEG-style coding, whole-image DCT and Hadamard" />
      {jv.times_smaller && (
        <p className="mt-2 rounded-lg bg-emerald-50 px-3 py-2 text-sm text-emerald-900">
          For the same quality as whole-image DCT at {r.keep_percent} % ({jv.dct_file_psnr_db.toFixed(1)} dB), JPEG-style
          coding needs <b>{jv.times_smaller.toFixed(1)}× fewer bytes</b> ({jv.jpeg_bpp_same_quality.toFixed(2)} instead
          of {jv.dct_bpp.toFixed(2)} bits per pixel){r.curve_scale < 1 ? " on this chart's smaller copy" : ""}.
        </p>
      )}
      <p className="mt-2 text-[11px] leading-snug text-slate-500">
        Up and to the left is better: more quality for fewer bits. Keeping fewer coefficients barely shrinks a file,
        because the result is still a full image that must be stored; quantisation + entropy coding (JPEG-style) is
        what makes it small. JPEG points: quality {sc.jpeg[0].quality}–{sc.jpeg[sc.jpeg.length - 1].quality}.{" "}
        {r.size_curve_note}{r.curve_scale < 1 ? " Computed on a ≤ 512 px copy." : ""}
      </p>
    </Card>
  );
}

function SummaryBanner({ s }) {
  let title, text;
  if (s.winner === "dct") {
    title = `At ${s.keep_percent} % of coefficients, DCT is ${s.psnr_margin_db.toFixed(1)} dB better than Hadamard`;
    text = "DCT packs more of the image's energy into fewer coefficients: its smooth cosines match natural images better than ±1 square waves.";
  } else if (s.winner === "hadamard") {
    title = `At ${s.keep_percent} % of coefficients, Hadamard is ${s.psnr_margin_db.toFixed(1)} dB better than DCT`;
    text = "Unusual for photos. Images made of flat blocks with sharp, axis-aligned steps suit Hadamard's square waves.";
  } else {
    title = `At ${s.keep_percent} % of coefficients, both transforms are equally good`;
    text = "The two bases capture this image about equally well at this budget.";
  }
  return (
    <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-emerald-900">
      <div className="flex items-start gap-3">
        <CircleCheck className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" />
        <div>
          <div className="font-semibold">{title}</div>
          <p className="mt-0.5 text-sm opacity-80">{text}</p>
          {s.jpeg && s.jpeg.matched_to_dct !== false && (
            <p className="mt-1.5 text-sm">
              <b>JPEG-style (quality {s.jpeg.quality})</b> reaches the same quality ({s.jpeg.psnr_db.toFixed(1)} dB) as a{" "}
              <b>{s.jpeg.times_smaller_than_dct.toFixed(1)}× smaller file</b>: {formatBytes(s.jpeg.file_bytes)} instead
              of {formatBytes(s.jpeg.dct_file_bytes)}. Quantisation + entropy coding is what actually shrinks the file.
            </p>
          )}
          {s.jpeg && s.jpeg.matched_to_dct === false && (
            <p className="mt-1.5 text-sm">
              <b>JPEG-style at quality {s.jpeg.quality}</b>: {s.jpeg.psnr_db.toFixed(1)} dB in {formatBytes(s.jpeg.file_bytes)},
              against DCT's {s.jpeg.dct_psnr_db.toFixed(1)} dB in {formatBytes(s.jpeg.dct_file_bytes)}
              {" "}({s.jpeg.times_smaller_than_dct >= 1
                ? `${s.jpeg.times_smaller_than_dct.toFixed(1)}× smaller file`
                : `${(1 / s.jpeg.times_smaller_than_dct).toFixed(1)}× larger file`}
              {s.jpeg.psnr_db >= s.jpeg.dct_psnr_db ? ", and better quality" : ", but lower quality"}).
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function MapFigure({ url, label }) {
  return (
    <figure className="min-w-0">
      <img src={url} alt={label} className="pixelated aspect-square w-full rounded-md border border-slate-200 bg-black object-contain" />
      <figcaption className="mt-1 truncate text-[11px] font-medium text-slate-600">{label}</figcaption>
    </figure>
  );
}

// =============================================================== shared pieces
function WhyCard({ findings, stages }) {
  const used = new Set(stages.map((s) => s.rule));
  const relevant = findings.filter((f) => used.has(f.code));
  const other = findings.filter((f) => !used.has(f.code));
  return (
    <Card title="Why?">
      <ul className="space-y-2 text-sm">
        {relevant.length === 0 && <li className="text-slate-600">Nothing in this image breaks this operation, so no fix was applied first.</li>}
        {relevant.map((f) => (
          <li key={f.code} className="flex gap-2">
            <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
            <span className="text-slate-700">{f.message}</span>
          </li>
        ))}
      </ul>
      {other.length > 0 && (
        <details className="mt-3 text-sm">
          <summary className="cursor-pointer text-xs font-medium text-slate-500">Other observations ({other.length})</summary>
          <ul className="mt-2 space-y-2">
            {other.map((f) => (
              <li key={f.code} className="flex gap-2">
                <Info className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
                <span className="text-slate-500">{f.message}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </Card>
  );
}

function Verdict({ v, checks }) {
  const tone = {
    good: "border-emerald-200 bg-emerald-50 text-emerald-900",
    warn: "border-amber-200 bg-amber-50 text-amber-900",
    neutral: "border-slate-200 bg-white text-slate-800",
  }[v.tone];
  return (
    <div className={cx("rounded-xl border p-4", tone)}>
      <div className="flex items-start gap-3">
        {v.tone === "good" ? <CircleCheck className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" /> : <Info className="mt-0.5 h-5 w-5 shrink-0 text-slate-400" />}
        <div className="min-w-0 flex-1">
          <div className="font-semibold">{v.title}</div>
          <p className="mt-0.5 text-sm opacity-80">{v.text}</p>
          {checks.length > 0 && (
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-wide opacity-60">
                  <tr><th className="py-1 pr-3 font-medium">Measured</th><th className="py-1 pr-3 font-medium">Direct</th><th className="py-1 pr-3 font-medium">Guided</th><th className="py-1 font-medium">Better is</th></tr>
                </thead>
                <tbody>
                  {checks.map((c) => (
                    <tr key={c.criterion} className="border-t border-black/5">
                      <td className="py-1.5 pr-3">{c.criterion}</td>
                      <td className="py-1.5 pr-3 tabular-nums">{fmt(c.a)}</td>
                      <td className={cx("py-1.5 pr-3 font-semibold tabular-nums", c.passed ? "text-emerald-700" : "")}>{fmt(c.b)} {c.passed && "✓"}</td>
                      <td className="py-1.5 text-xs opacity-70">{c.better}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PipelineStrip({ input, stages, target }) {
  const steps = [{ name: "Input", image: input }, ...stages.map((s) => ({ name: STAGE_LABELS[s.name] || s.name, image: s.image, reason: s.reason }))];
  return (
    <Card title="Guided route (B)">
      <div className="flex items-start gap-2 overflow-x-auto pb-1">
        {steps.map((s, i) => (
          <div key={i} className="flex shrink-0 items-start gap-2">
            <figure className="w-28">
              <img src={s.image} alt={s.name} className="aspect-square w-28 rounded-md border border-slate-200 object-cover" />
              <figcaption className="mt-1 text-xs font-medium text-slate-700">{s.name}</figcaption>
              {s.reason && <div className="text-[11px] leading-snug text-slate-500">{s.reason}</div>}
            </figure>
            <ArrowRight className="mt-12 h-4 w-4 shrink-0 text-slate-300" />
          </div>
        ))}
        <div className="flex h-28 w-28 shrink-0 items-center justify-center rounded-md border border-dashed border-brand-300 bg-brand-50 text-center text-sm font-semibold text-brand-700">
          {target}
        </div>
      </div>
    </Card>
  );
}

function Figure({ url, title, sub, highlight }) {
  return (
    <figure className={cx("overflow-hidden rounded-lg border", highlight ? "border-brand-300" : "border-slate-200")}>
      <img src={url} alt={title} className="w-full bg-slate-100" />
      <figcaption className="px-3 py-2">
        <div className={cx("text-sm font-semibold", highlight ? "text-brand-700" : "text-slate-700")}>{title}</div>
        {sub && <div className="text-xs text-slate-500">{sub}</div>}
      </figcaption>
    </figure>
  );
}

function Placeholder({ text, img }) {
  return (
    <div className="flex min-h-[50vh] flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed border-slate-200 bg-white p-8 text-center">
      {img ? <img src={img} alt="" className="max-h-[40vh] max-w-full rounded-lg opacity-90" /> : <GitCompare className="h-8 w-8 text-slate-300" />}
      <p className="max-w-md text-sm text-slate-500">{text}</p>
    </div>
  );
}
