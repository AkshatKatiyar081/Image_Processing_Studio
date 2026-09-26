import { useEffect, useRef, useState } from "react";
import { ArrowRight, CircleCheck, Download, GitCompare, Info, TriangleAlert } from "lucide-react";
import { api } from "../lib/api.js";
import { download, extFor } from "../lib/image.js";
import { INTENTS, OPTION_LABELS, STAGE_LABELS, VERDICTS } from "../lib/labels.js";
import CompareSlider from "./CompareSlider.jsx";
import { Button, Card, ErrorBanner, Segmented, Spinner, cx } from "./ui.jsx";

const TARGET_NAME = { segment: "Otsu", edges: "Edges", sharpen: "Sharpen", equalize: "Equalize" };

function paramsFor(intent, operator) {
  if (intent === "segment") return { method: "otsu" };
  if (intent === "edges") return { operator, edge_threshold: 0.2 };
  if (intent === "sharpen") return { operation: "unsharp", k: 1.0 };
  return {};
}

const fmt = (v) => (Math.abs(v) < 1 && v !== 0 ? v.toFixed(3) : Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2));

export default function CompareTab({ image, left, layout: Layout }) {
  const [intent, setIntent] = useState("segment");
  const [operator, setOperator] = useState("sobel");
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [display, setDisplay] = useState("slider");
  const [maskView, setMaskView] = useState(false);
  const ctrl = useRef(null);

  useEffect(() => { setResult(null); setError(null); ctrl.current?.abort(); }, [image?.id]);

  const run = async () => {
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    setRunning(true);
    setError(null);
    try {
      const data = await api.guided(image.file, intent, paramsFor(intent, operator), "opencv", c.signal);
      if (!c.signal.aborted) setResult(data);
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message);
    } finally {
      if (ctrl.current === c) setRunning(false);
    }
  };

  const pickImage = (branch) => {
    if (!branch) return null;
    if (result.intent === "segment" && maskView) return branch.extras.mask;
    return branch.image;
  };

  const sidebar = (
    <Card title="What do you want to do?" icon={GitCompare}>
      <p className="mb-3 text-xs leading-relaxed text-slate-500">
        We check your image for problems that break this operation, then run it twice:
        <b className="text-slate-700"> directly</b>, and <b className="text-slate-700">after fixing the problem first</b>, and measure which is better.
      </p>
      <div className="grid grid-cols-2 gap-2">
        {INTENTS.map((it) => (
          <button key={it.id} onClick={() => { setIntent(it.id); setResult(null); }} aria-pressed={intent === it.id}
                  className={cx("flex flex-col items-start gap-1 rounded-lg border px-3 py-2.5 text-left transition",
                    intent === it.id ? "border-brand-500 bg-brand-50 ring-1 ring-brand-500" : "border-slate-200 hover:bg-slate-50")}>
            <it.icon className={cx("h-4 w-4", intent === it.id ? "text-brand-600" : "text-slate-400")} />
            <span className="text-sm font-semibold text-slate-800">{it.label}</span>
            <span className="text-[11px] text-slate-500">{it.desc}</span>
          </button>
        ))}
      </div>
      {intent === "edges" && (
        <div className="mt-3">
          <div className="mb-1 text-xs font-semibold text-slate-600">Edge operator</div>
          <Segmented size="sm" value={operator} onChange={(v) => { setOperator(v); setResult(null); }}
                     options={["sobel", "prewitt", "roberts"].map((o) => ({ value: o, label: OPTION_LABELS[o] }))} />
          <p className="mt-1 text-[11px] text-slate-400">Roberts has no built-in smoothing, so it needs a fix at lower noise levels than Sobel.</p>
        </div>
      )}
      <Button className="mt-4 w-full" size="lg" onClick={run} loading={running} disabled={!image}>
        {image ? "Run comparison" : "Load an image first"}
      </Button>
    </Card>
  );

  let main;
  if (!image) {
    main = <Placeholder text="Load an image, choose what you want to do, and press Run comparison." />;
  } else if (!result) {
    main = running ? <Card><div className="py-24"><Spinner label="Diagnosing and running both routes…" /></div></Card>
      : (
        <div className="space-y-3">
          <ErrorBanner error={error} onClose={() => setError(null)} />
          <Placeholder text={`Press "Run comparison" to see the direct result next to the guided (fix-first) result for “${INTENTS.find((i) => i.id === intent).label}”.`} img={image.url} />
        </div>
      );
  } else {
    const v = VERDICTS[result.verdict.result];
    const a = result.branch_a, b = result.branch_b;
    const aUrl = pickImage(a), bUrl = pickImage(b);
    main = (
      <div className="space-y-4">
        <ErrorBanner error={error} onClose={() => setError(null)} />
        <Verdict v={v} checks={result.verdict.checks} />

        {b && <PipelineStrip input={image.url} stages={b.stages} target={TARGET_NAME[result.intent]} />}

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
            <CompareSlider before={image.url} after={aUrl} beforeLabel="Input" afterLabel="Result" busy={running} overlay={<Spinner />} />
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

        <WhyCard findings={result.diagnosis.findings} stages={b?.stages || []} />
      </div>
    );
  }

  return <Layout left={left} controls={sidebar} main={main} />;
}

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
      {img ? <img src={img} alt="" className="max-h-[40vh] rounded-lg opacity-90" /> : <GitCompare className="h-8 w-8 text-slate-300" />}
      <p className="max-w-md text-sm text-slate-500">{text}</p>
    </div>
  );
}
