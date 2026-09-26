import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Cpu, Download, SlidersHorizontal, WandSparkles } from "lucide-react";
import { api } from "../lib/api.js";
import { download, extFor } from "../lib/image.js";
import { METRIC_LABELS, TOOLS, formatMetric, toolById } from "../lib/labels.js";
import CompareSlider from "./CompareSlider.jsx";
import Histogram from "./Histogram.jsx";
import ParamControl, { isVisible } from "./ParamControl.jsx";
import { Button, Card, ErrorBanner, Segmented, Spinner, Stat, Toggle, cx } from "./ui.jsx";

const defaultsFor = (schema) => Object.fromEntries((schema?.params || []).map((p) => [p.name, p.default]));

export default function ToolsTab({ image, schemas, left, request, beforeHist, onChain, layout: Layout }) {
  const [toolId, setToolId] = useState("contrast");
  const [presetId, setPresetId] = useState(null);
  const [params, setParams] = useState({});
  const [engine, setEngine] = useState("opencv");
  const [advanced, setAdvanced] = useState(false);
  const [pickMode, setPickMode] = useState(false);
  const [result, setResult] = useState(null);   // { data, tool, preset, view }
  const [view, setView] = useState("overlay");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const ctrl = useRef(null);

  const tool = toolById(toolId);
  const schema = schemas?.[toolId];

  const run = useCallback(async (tid, p, eng = engine) => {
    if (!image) return;
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    setRunning(true);
    setError(null);
    const send = Object.fromEntries(Object.entries(p).filter(([, v]) => v !== null && v !== undefined));
    try {
      const data = await api.process(image.file, tid, send, eng, c.signal);
      if (!c.signal.aborted) {
        setResult({ data, tool: tid });
        setView("overlay");
      }
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message);
    } finally {
      if (ctrl.current === c) setRunning(false);
    }
  }, [image, engine]);

  const choosePreset = useCallback((tid, pid, { autoRun = true } = {}) => {
    const t = toolById(tid);
    const preset = t.presets.find((x) => x.id === pid) || t.presets[0];
    const p = { ...defaultsFor(schemas?.[tid]), ...preset.params };
    setToolId(tid);
    setPresetId(preset.id);
    setParams(p);
    if (preset.needsSeed) {
      setPickMode(true);
      return;
    }
    setPickMode(false);
    if (autoRun) run(tid, p);
  }, [schemas, run]);

  // new image -> clear result
  useEffect(() => { setResult(null); setError(null); setPickMode(false); ctrl.current?.abort(); }, [image?.id]);

  // "Fix it" buttons in the health card
  useEffect(() => { if (request && schemas) choosePreset(request.tool, request.preset); }, [request]); // eslint-disable-line

  // initialise params once the schemas arrive
  useEffect(() => { if (schemas && !presetId) { setParams({ ...defaultsFor(schemas[toolId]), ...tool.presets[0].params }); } }, [schemas]); // eslint-disable-line

  const selectTool = (tid) => {
    if (tid === toolId) return;
    const t = toolById(tid);
    setToolId(tid);
    setPresetId(null);
    setPickMode(false);
    setParams({ ...defaultsFor(schemas?.[tid]), ...t.presets[0].params });
  };

  const setParam = (name, v) => {
    const next = { ...params, [name]: v };
    setParams(next);
    setPresetId("custom");
    if (name === "method") setPickMode(v === "region_growing");
  };

  const onPick = (xy) => {
    const next = { ...params, seed: xy };
    setParams(next);
    run(toolId, next);
  };

  const isSegment = result?.tool === "segment";
  const afterUrl = useMemo(() => {
    if (!result) return null;
    const d = result.data;
    if (isSegment && view === "mask") return d.extras.mask;
    if (isSegment && view === "cutout") return d.extras.cutout;
    return d.image;
  }, [result, view, isSegment]);

  const beforeUrl = result?.data.input || image?.url;
  const metrics = result?.data.metrics || {};
  const regionGrow = toolId === "segment" && params.method === "region_growing";

  const sidebar = (
    <>
      <Card title="1 · Choose a tool" icon={WandSparkles}>
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

      <Card title="2 · Pick an effect" icon={SlidersHorizontal}>
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
              <ParamControl key={p.name} p={p} value={params[p.name]} onChange={(v) => setParam(p.name, v)}
                            pickMode={pickMode} onTogglePick={() => setPickMode(!pickMode)} />
            ))}
            <div className="border-t border-slate-100 pt-3">
              <Toggle checked={engine === "numpy"} onChange={(on) => setEngine(on ? "numpy" : "opencv")}
                      label={<span className="flex items-center gap-1.5"><Cpu className="h-3.5 w-3.5" />Use our from-scratch NumPy code</span>}
                      hint="Same result as OpenCV (validated by the test suite), usually a bit slower." />
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
      <Card>
        <div className="space-y-3">
          {(regionGrow && pickMode) && !running && !result && (
            <p className="text-center text-sm text-brand-700">Click on the object you want to select. The region grows from that point.</p>
          )}
          <CompareSlider before={beforeUrl} after={afterUrl} afterLabel={result ? toolById(result.tool).name : "After"}
                         pickMode={regionGrow && pickMode} onPick={onPick} marker={regionGrow ? params.seed : null}
                         checker={isSegment && view === "cutout"} busy={running} overlay={<Spinner label="Processing…" />} />
          {result && (
            <div className="flex flex-wrap items-center justify-between gap-2">
              {isSegment ? (
                <Segmented size="sm" value={view} onChange={setView}
                           options={[{ value: "overlay", label: "Overlay" }, { value: "mask", label: "Mask" }, { value: "cutout", label: "Cut-out" }]} />
              ) : <span className="text-xs text-slate-500">Drag the divider to compare.</span>}
              <div className="flex flex-wrap gap-2">
                {regionGrow && !pickMode && <Button size="sm" variant="secondary" onClick={() => setPickMode(true)}>Pick another point</Button>}
                <Button size="sm" variant="secondary" onClick={() => onChain(afterUrl, toolById(result.tool).name)}>Keep editing this result</Button>
                <Button size="sm" onClick={() => download(afterUrl, `${image.name}_${result.tool}${isSegment ? `_${view}` : ""}.${extFor(afterUrl)}`)}>
                  <Download className="h-4 w-4" />Download
                </Button>
              </div>
            </div>
          )}
          <ErrorBanner error={error} onClose={() => setError(null)} />
          {result?.data.warnings?.map((w) => (
            <p key={w} className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">{w}</p>
          ))}
        </div>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <Card>
          <Histogram before={result?.data.histograms.before || beforeHist} after={result?.data.histograms.after} />
        </Card>
        <Card title="Result details">
          {result ? (
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(metrics).map(([k, v]) => (
                <Stat key={k} label={METRIC_LABELS[k] || k} value={formatMetric(k, v)} />
              ))}
              <Stat label="Processing time" value={`${result.data.meta.time_ms} ms`} sub={result.data.meta.engine === "numpy" ? "NumPy engine" : "OpenCV engine"} />
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

function EmptyState() {
  return (
    <div className="flex h-full min-h-[50vh] flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-200 bg-white p-8 text-center">
      <div className="mb-3 rounded-full bg-brand-50 p-3"><WandSparkles className="h-6 w-6 text-brand-600" /></div>
      <h2 className="text-lg font-semibold text-slate-800">Start by loading an image</h2>
      <p className="mt-1 max-w-sm text-sm text-slate-500">Upload your own, paste one, or click a sample on the left. Then pick a tool and an effect: results appear here with a before/after slider.</p>
    </div>
  );
}
