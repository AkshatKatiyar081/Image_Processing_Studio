import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Cpu, Download, FileImage, SlidersHorizontal, WandSparkles } from "lucide-react";
import { EXTENSION, api, files } from "../lib/api.js";
import { download } from "../lib/image.js";
import { COMPRESS_DETAIL_KEYS, HIDDEN_OPTIONS, METRIC_LABELS, TOOL_PARAM_LABELS, TOOLS, formatBytes, formatMetric, toolById } from "../lib/labels.js";
import BeforeAfterView, { ViewToggle } from "./BeforeAfterView.jsx";
import Histogram from "./Histogram.jsx";
import ParamControl, { isVisible } from "./ParamControl.jsx";
import { Button, Card, ErrorBanner, Spinner, Stat, Toggle, cx } from "./ui.jsx";

const defaultsFor = (schema) => Object.fromEntries((schema?.params || []).map((p) => [p.name, p.default]));

/** Schema param as shown in Tools: hidden options removed (HIDDEN_OPTIONS), tool-specific label added. */
function visibleParam(toolId, p) {
  const hidden = HIDDEN_OPTIONS[toolId]?.[p.name];
  const out = hidden && p.options ? { ...p, options: p.options.filter((o) => !hidden.includes(o)) } : { ...p };
  const label = TOOL_PARAM_LABELS[toolId]?.[p.name];
  if (label) out.label = label;
  return out;
}

/** JPEG-style compression keeps the uploaded JPEG's colour subsampling, so sizes compare fairly. */
function withFileDefaults(tid, p, fileInfo) {
  const sub = fileInfo?.chroma_subsampling;
  if (tid === "compress" && p.method === "jpeg" && ["4:2:0", "4:2:2", "4:4:4"].includes(sub)) {
    return { ...p, chroma_subsampling: sub };
  }
  return p;
}

export default function ToolsTab({ image, schemas, left, request, beforeHist, onChain, layout: Layout }) {
  const [toolId, setToolId] = useState("contrast");
  const [presetId, setPresetId] = useState(null);
  const [params, setParams] = useState({});
  const [engine, setEngine] = useState("opencv");
  const [advanced, setAdvanced] = useState(false);
  const [result, setResult] = useState(null);   // { data, tool, send, engine }
  const [saving, setSaving] = useState(null);   // "download" | "chain" while the full-quality file is fetched
  const [notice, setNotice] = useState(null);   // e.g. "Saved as JPEG (quality 95) because …"
  const [view, setView] = useState("slider");   // "slider" | "side"
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [fileInfo, setFileInfo] = useState({ id: null, data: null, error: null });   // /api/inspect of the upload
  const ctrl = useRef(null);

  const tool = toolById(toolId);
  const schema = schemas?.[toolId];

  // Image details (format, file size, JPEG quality …) for the Compression tool, fetched once per image.
  useEffect(() => {
    if (toolId !== "compress" || !image || fileInfo.id === image.id) return undefined;
    const c = new AbortController();
    setFileInfo({ id: image.id, data: null, error: null });
    api.inspect(image.file, c.signal)
      .then((d) => setFileInfo({ id: image.id, data: d, error: null }))
      .catch((e) => e.name !== "AbortError" && setFileInfo({ id: image.id, data: null, error: e.message }));
    return () => c.abort();
  }, [toolId, image?.id]); // eslint-disable-line
  const info = fileInfo.id === image?.id ? fileInfo.data : null;

  const run = useCallback(async (tid, p, eng = engine) => {
    if (!image) return;
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    setRunning(true);
    setError(null);
    const send = Object.fromEntries(Object.entries(p).filter(([, v]) => v !== null && v !== undefined));
    try {
      // Compression returns a real file (same format as the upload); the other tools return pixels.
      const data = tid === "compress"
        ? await api.compress(image.file, send, eng, c.signal)
        : await api.process(image.file, tid, send, eng, c.signal);
      if (!c.signal.aborted) setResult({ data, tool: tid, send, engine: eng });
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message);
    } finally {
      if (ctrl.current === c) setRunning(false);
    }
  }, [image, engine]);

  const choosePreset = useCallback((tid, pid, { autoRun = true } = {}) => {
    const t = toolById(tid);
    if (!t) return;
    const preset = t.presets.find((x) => x.id === pid) || t.presets[0];
    const p = withFileDefaults(tid, { ...defaultsFor(schemas?.[tid]), ...preset.params }, info);
    setToolId(tid);
    setPresetId(preset.id);
    setParams(p);
    if (autoRun) run(tid, p);
  }, [schemas, run, info]);

  // new image -> clear result
  useEffect(() => { setResult(null); setError(null); setNotice(null); ctrl.current?.abort(); }, [image?.id]);

  // "Fix it" buttons in the health card
  useEffect(() => { if (request && schemas) choosePreset(request.tool, request.preset); }, [request]); // eslint-disable-line

  // initialise params once the schemas arrive
  useEffect(() => { if (schemas && !presetId) { setParams({ ...defaultsFor(schemas[toolId]), ...tool.presets[0].params }); } }, [schemas]); // eslint-disable-line

  const selectTool = (tid) => {
    if (tid === toolId) return;
    const t = toolById(tid);
    setToolId(tid);
    setPresetId(null);
    setParams(withFileDefaults(tid, { ...defaultsFor(schemas?.[tid]), ...t.presets[0].params }, info));
  };

  const setParam = (name, v) => {
    setParams({ ...params, [name]: v });
    setPresetId("custom");
  };

  const afterUrl = result?.data.image || null;      // a preview (≤ 1280 px)
  const beforeUrl = image?.url;

  /** Download / Keep editing use the FULL-QUALITY result, fetched only when clicked. */
  const save = async (action) => {
    setSaving(action);
    setError(null);
    setNotice(null);
    try {
      const f = result.tool === "compress"
        ? await files.compress(image.file, result.send, result.engine)
        : await files.process(image.file, result.tool, result.send, result.engine);
      if (f.note && result.tool !== "compress") setNotice(f.note);   // compression already lists it in its warnings
      const name = toolById(result.tool)?.name || result.tool;
      if (action === "chain") {
        await onChain(f.file, name);
      } else {
        const suffix = result.tool === "compress" ? "compressed" : result.tool;
        download(f.file, `${image.name}_${suffix}.${EXTENSION[f.format] || "png"}`);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(null);
    }
  };
  const blocked = isCompressNotDownloadable(result);
  const metrics = result?.data.metrics || {};
  const isCompress = result?.tool === "compress";

  const detailKeys = useMemo(() => (isCompress
    ? COMPRESS_DETAIL_KEYS.filter((k) => k in metrics)
    : Object.keys(metrics)), [isCompress, metrics]);

  const sidebar = (
    <>
      <Card title="1 · Choose a tool" icon={WandSparkles} tour="tools">
        <div className="grid grid-cols-2 gap-2">
          {TOOLS.map((t) => (
            <button key={t.id} onClick={() => selectTool(t.id)} aria-pressed={toolId === t.id}
                    className={cx("flex flex-col items-start gap-1 rounded-lg border px-3 py-2.5 text-left transition",
                      toolId === t.id ? "border-brand-500 bg-brand-50 ring-1 ring-brand-500" : "border-slate-200 hover:border-slate-300 hover:bg-slate-50")}>
              <t.icon className={cx("h-4 w-4", toolId === t.id ? "text-brand-600" : "text-slate-400")} />
              <span className="text-sm font-semibold text-slate-800">{t.name}</span>
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs text-slate-500">{tool.blurb}</p>
      </Card>

      <Card title="2 · Pick an effect" icon={SlidersHorizontal} tour="effects">
        {!image && <p className="mb-2 text-xs text-amber-700">Load an image first.</p>}
        <div className="space-y-1.5">
          {tool.presets.map((p) => (
            <button key={p.id} disabled={!image || !schemas} onClick={() => choosePreset(toolId, p.id)}
                    className={cx("w-full rounded-lg border px-3 py-2 text-left transition disabled:opacity-50",
                      presetId === p.id ? "border-brand-500 bg-brand-50" : "border-slate-200 hover:border-slate-300 hover:bg-slate-50")}>
              <div className="text-sm font-medium text-slate-800">{p.label}</div>
              <div className="text-xs text-slate-500">{p.hint}</div>
            </button>
          ))}
        </div>

        <button onClick={() => setAdvanced(!advanced)} aria-expanded={advanced}
                className="mt-3 flex w-full items-center justify-between rounded-lg px-1 py-1 text-sm font-medium text-slate-600 hover:text-slate-900">
          Advanced settings
          <ChevronDown className={cx("h-4 w-4 transition", advanced && "rotate-180")} />
        </button>
        {advanced && schema && (
          <div className="mt-2 space-y-4 border-t border-slate-100 pt-3">
            {schema.params.filter((p) => isVisible(p, params)).map((p) => (
              <ParamControl key={p.name} p={visibleParam(toolId, p)} value={params[p.name]} onChange={(v) => setParam(p.name, v)} />
            ))}
            <div className="border-t border-slate-100 pt-3">
              <Toggle checked={engine === "numpy"} onChange={(on) => setEngine(on ? "numpy" : "opencv")}
                      label={<span className="flex items-center gap-1.5"><Cpu className="h-3.5 w-3.5" />Use our from-scratch NumPy code</span>}
                      hint={toolId === "compress"
                        ? "JPEG-style: the preview comes from our own block-DCT decoder instead of libjpeg (the downloaded file is always a standard JPEG). Whole-image DCT: our matrix code instead of cv2.dct. Walsh–Hadamard is always our own code."
                        : "Same result as OpenCV (validated by the test suite), usually a bit slower."} />
            </div>
            <Button className="w-full" onClick={() => run(toolId, params)} loading={running} disabled={!image}>Apply</Button>
          </div>
        )}
      </Card>
    </>
  );

  const main = !image ? (
    <EmptyState />
  ) : (
    <div className="space-y-4">
      {toolId === "compress" && (
        <FileCard info={info} error={fileInfo.id === image.id ? fileInfo.error : null}
                  file={isCompress ? result.data.file : null} />
      )}
      <Card tour="result">
        <div className="space-y-3">
          <BeforeAfterView view={view} before={beforeUrl} after={afterUrl}
                           afterLabel={result ? toolById(result.tool)?.name || "After" : "After"}
                           busy={running} overlay={<Spinner label="Processing…" />} />
          {result && (
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <ViewToggle value={view} onChange={setView} />
                {view === "slider" && <span className="text-xs text-slate-500">Drag the divider to compare.</span>}
              </div>
              <div data-tour="actions" className="flex flex-wrap gap-2">
                <Button size="sm" variant="secondary" onClick={() => save("chain")} loading={saving === "chain"}
                        disabled={!!saving || blocked}>Keep editing this result</Button>
                <Button size="sm" onClick={() => save("download")} loading={saving === "download"} disabled={!!saving || blocked}
                        title={blocked ? "This file is over the download limit. Choose a lower quality." : undefined}>
                  <Download className="h-4 w-4" />
                  {isCompress ? `Download ${result.data.file.format} (${formatBytes(result.data.file.bytes)})` : "Download"}
                </Button>
              </div>
            </div>
          )}
          <ErrorBanner error={error} onClose={() => setError(null)} />
          {notice && <p className="rounded-lg bg-sky-50 px-3 py-2 text-xs text-sky-800">{notice}</p>}
          {result?.data.warnings?.map((w) => (
            <p key={w} className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">{w}</p>
          ))}
        </div>
      </Card>

      {isCompress && <CompressionCards data={result.data} />}

      <div data-tour="charts" className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <Card>
          <Histogram before={result?.data.histograms.before || beforeHist} after={result?.data.histograms.after} />
        </Card>
        <Card title="Result details">
          {result ? (
            <div className="space-y-2">
              <div className="grid grid-cols-2 gap-2">
                {detailKeys.map((k) => (
                  <Stat key={k} label={METRIC_LABELS[k] || k} value={formatMetric(k, metrics[k])} />
                ))}
                <Stat label="Processing time" value={`${result.data.meta.time_ms} ms`} sub={result.data.meta.engine === "numpy" ? "NumPy engine" : "OpenCV engine"} />
              </div>
              {isCompress && (
                <p className="text-[11px] leading-snug text-slate-500">
                  PSNR, SSIM and the error map describe the file you download. "Coefficients kept" counts the
                  coefficients that are not zero; the real saving is the file size shown above.
                </p>
              )}
            </div>
          ) : (
            <p className="text-sm text-slate-500">Pick an effect to see before/after numbers here.</p>
          )}
        </Card>
      </div>
    </div>
  );

  return <Layout left={left} controls={sidebar} main={main} />;
}

const isCompressNotDownloadable = (result) => result?.tool === "compress" && result.data.file?.downloadable === false;

/** Image details before compressing; your file vs the compressed file afterwards. */
function FileCard({ info, error, file }) {
  if (error) return <Card title="Image details" icon={FileImage}><p className="text-sm text-red-700">{error}</p></Card>;
  if (!info) return <Card title="Image details" icon={FileImage}><Spinner label="Reading the file…" /></Card>;
  const rows = [
    ["Format", info.format + (info.lossless_format ? " (lossless)" : "")],
    ["File size", formatBytes(info.file_bytes)],
    ["Pixels", `${info.width} × ${info.height} (${info.megapixels} MP)`],
    ["Bits per pixel", info.bits_per_pixel.toFixed(2)],
  ];
  if (info.format === "JPEG") {
    rows.push(["JPEG quality", info.jpeg_quality_estimate ? `≈ ${info.jpeg_quality_estimate}` : "unknown"]);
    rows.push(["Colour subsampling", info.chroma_subsampling || "unknown"]);
  }
  rows.push(["Metadata", [info.has_exif && "EXIF", info.has_icc_profile && "colour profile"].filter(Boolean).join(", ") || "none"]);

  const change = file ? file.size_percent - 100 : null;
  return (
    <Card title={file ? "Your file → compressed file" : "Image details"} icon={FileImage}>
      {!file ? (
        <>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm sm:grid-cols-3">
            {rows.map(([k, v]) => (
              <div key={k} className="min-w-0">
                <dt className="text-[11px] font-medium uppercase tracking-wide text-slate-500">{k}</dt>
                <dd className="truncate font-semibold text-slate-800 tabular-nums">{v}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-xs text-slate-500">
            Pick a compression on the left. The result is saved as a real file
            {info.format === "JPEG" ? " (a JPEG for your JPEG)" : ""} that you can download.
          </p>
        </>
      ) : (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <SizeBox label="Your file" format={info.format} bytes={file.original_bytes}
                     sub={info.jpeg_quality_estimate ? `quality ≈ ${info.jpeg_quality_estimate}, ${info.chroma_subsampling}` : null} />
            <SizeBox label="Compressed file" format={file.format} bytes={file.bytes}
                     sub={file.quality ? `quality ${file.quality}${file.chroma_subsampling ? `, ${file.chroma_subsampling}` : ""}` : "lossless"} />
            <div className={cx("col-span-2 flex flex-col justify-center rounded-lg px-3 py-2 sm:col-span-1",
              change < 0 ? "bg-emerald-50 text-emerald-800" : "bg-amber-50 text-amber-800")}>
              <div className="text-2xl font-bold tabular-nums">{change < 0 ? "−" : "+"}{Math.abs(change).toFixed(0)} %</div>
              <div className="text-xs">{change < 0 ? `${file.size_percent} % of your file` : "not smaller than your file"}</div>
            </div>
          </div>
          <div className="space-y-1" aria-hidden="true">
            <div className="h-2.5 rounded bg-slate-300" style={{ width: `${Math.min(100, 100 * 100 / Math.max(100, file.size_percent))}%` }} />
            <div className="h-2.5 rounded" style={{ width: `${Math.min(100, file.size_percent * 100 / Math.max(100, file.size_percent))}%`, background: change < 0 ? "#10b981" : "#f59e0b" }} />
          </div>
          {file.source === "same_format_as_upload" && (
            <p className="text-[11px] leading-snug text-slate-500">
              Saved in your file's format{file.quality ? ` at its own quality (${file.quality})` : ""}, so the size change comes from
              the transform step only. Use a JPEG-style preset to make the file clearly smaller.
            </p>
          )}
        </div>
      )}
    </Card>
  );
}

function SizeBox({ label, format, bytes, sub }) {
  return (
    <div className="min-w-0 rounded-lg bg-slate-50 px-3 py-2">
      <div className="text-[11px] font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className="text-lg font-semibold text-slate-800 tabular-nums">{formatBytes(bytes)}</div>
      <div className="truncate text-xs text-slate-500">{format}{sub ? ` · ${sub}` : ""}</div>
    </div>
  );
}

function CompressionCards({ data }) {
  const ex = data.extras || {};
  const m = data.metrics || {};
  const jpeg = m.method === "jpeg";
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Card title={jpeg ? "Block DCT coefficients (8×8)" : "Coefficients"}>
        <div className="grid grid-cols-3 gap-2">
          <MapFigure url={ex.coef_before} label={jpeg ? "Before quantisation" : "All coefficients"} />
          <MapFigure url={ex.coef_after} label={jpeg ? "After quantisation" : "Kept"} />
          <MapFigure url={ex.kept_mask} label={jpeg ? "Non-zero after quantisation" : "Kept positions"} />
        </div>
        <p className="mt-2 text-[11px] leading-snug text-slate-500">
          {jpeg
            ? <>log(1 + |F|) of the luminance. Coefficients are grouped by frequency: tile (u, v) holds frequency (u, v)
                of every 8×8 block, so the top-left (DC) tile is a small copy of the image and the bottom-right tiles are
                the finest detail. Quantisation turns {formatMetric("kept_percent", 100 - m.kept_percent)} of all
                coefficients into 0; those cost almost nothing in the file.</>
            : <>log(1 + |F|) of the luminance coefficients. Low frequencies are at the top-left; white in the mask marks
                the {formatMetric("kept_percent", m.kept_percent)} that were kept.</>}
        </p>
      </Card>
      <Card title="Error map">
        <img src={ex.error_map} alt="Error map" className="pixelated w-full rounded-md border border-slate-200" />
        <p className="mt-2 text-[11px] leading-snug text-slate-500">
          |original − result|, brighter = bigger error, ×{m.error_amplification} amplified.
        </p>
      </Card>
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

function EmptyState() {
  return (
    <div className="flex h-full min-h-[50vh] flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-200 bg-white p-8 text-center">
      <div className="mb-3 rounded-full bg-brand-50 p-3"><WandSparkles className="h-6 w-6 text-brand-600" /></div>
      <h2 className="text-lg font-semibold text-slate-800">Start by loading an image</h2>
      <p className="mt-1 max-w-sm text-sm text-slate-500">Upload your own, paste one, or click a sample on the left. Then pick a tool and an effect: results appear here with a before/after slider.</p>
    </div>
  );
}
