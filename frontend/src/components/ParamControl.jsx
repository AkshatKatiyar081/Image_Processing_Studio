import { OPTION_LABELS, PARAM_LABELS } from "../lib/labels.js";
import { Segmented, Toggle, cx } from "./ui.jsx";

export const isVisible = (p, values) =>
  !p.when || Object.entries(p.when).every(([k, allowed]) => allowed.includes(values[k]));

const optLabel = (v) => OPTION_LABELS[v] ?? String(v);

/** One control generated from a pipeline parameter schema entry. */
export default function ParamControl({ p, value, onChange }) {
  const label = p.label || PARAM_LABELS[p.name] || p.name;

  if (p.type === "bool") return <Toggle checked={value} onChange={onChange} label={label} hint={p.description} />;

  if (p.type === "enum") {
    const opts = p.options.map((o) => ({ value: o, label: p.name === "kernel_size" ? `${o}×${o}` : optLabel(o) }));
    return (
      <Field label={label} hint={p.description}>
        {opts.length <= 4 ? (
          <Segmented size="sm" value={value} onChange={onChange} options={opts} />
        ) : (
          <select className="w-full rounded-md border border-slate-200 bg-white px-2 py-1.5 text-sm" value={value}
                  onChange={(e) => onChange(p.options.find((o) => String(o) === e.target.value))}>
            {opts.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        )}
      </Field>
    );
  }

  // int / float slider (gamma also accepts "auto")
  const isAuto = value === "auto";
  const allowsAuto = p.also_allowed?.includes("auto");
  const step = p.step ?? (p.type === "int" ? 1 : 0.1);
  const num = isAuto ? p.default : value;
  return (
    <Field label={label} hint={p.description}
           right={<span className="text-xs font-semibold text-slate-700 tabular-nums">{isAuto ? "auto" : Number(num).toFixed(p.type === "int" ? 0 : step < 0.1 ? 2 : 1)}</span>}>
      <div className="flex items-center gap-2">
        <input type="range" className="w-full" min={p.min} max={p.max} step={step} value={num} disabled={isAuto}
               aria-label={label}
               onChange={(e) => onChange(p.type === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value))} />
        {allowsAuto && (
          <label className="flex items-center gap-1 text-xs text-slate-600">
            <input type="checkbox" checked={isAuto} onChange={(e) => onChange(e.target.checked ? "auto" : p.default)} /> auto
          </label>
        )}
      </div>
    </Field>
  );
}

function Field({ label, hint, right, children }) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-xs font-semibold text-slate-600">{label}</span>
        {right}
      </div>
      {children}
      {hint && <p className="mt-1 text-[11px] leading-snug text-slate-400">{hint}</p>}
    </div>
  );
}
