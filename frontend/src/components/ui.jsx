import { LoaderCircle, TriangleAlert, X } from "lucide-react";

export const cx = (...c) => c.filter(Boolean).join(" ");

export function Card({ title, icon: Icon, action, className, tour, children }) {
  return (
    <section data-tour={tour} className={cx("rounded-xl border border-slate-200 bg-white shadow-sm", className)}>
      {title && (
        <header className="flex items-center justify-between gap-2 border-b border-slate-100 px-4 py-2.5">
          <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-700">
            {Icon && <Icon className="h-4 w-4 text-slate-400" />}
            {title}
          </h2>
          {action}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Button({ variant = "primary", size = "md", className, loading, children, ...props }) {
  const base = "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 disabled:cursor-not-allowed disabled:opacity-50";
  const sizes = { sm: "px-2.5 py-1.5 text-xs", md: "px-3.5 py-2 text-sm", lg: "px-5 py-3 text-base" };
  const variants = {
    primary: "bg-brand-600 text-white shadow-sm hover:bg-brand-700",
    secondary: "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50",
    ghost: "text-slate-600 hover:bg-slate-100",
  };
  return (
    <button className={cx(base, sizes[size], variants[variant], className)} disabled={loading || props.disabled} {...props}>
      {loading && <LoaderCircle className="h-4 w-4 animate-spin" />}
      {children}
    </button>
  );
}

export function Spinner({ label }) {
  return (
    <div className="flex flex-col items-center gap-2 text-sm text-slate-600">
      <LoaderCircle className="h-7 w-7 animate-spin text-brand-600" />
      {label}
    </div>
  );
}

export function ErrorBanner({ error, onClose }) {
  if (!error) return null;
  return (
    <div role="alert" className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
      <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
      <span className="flex-1">{error}</span>
      {onClose && (
        <button onClick={onClose} aria-label="Dismiss" className="rounded p-0.5 hover:bg-red-100">
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}

/** Pill-style single choice. options: [{value, label}] */
export function Segmented({ value, onChange, options, size = "md", className }) {
  return (
    <div className={cx("inline-flex flex-wrap rounded-lg bg-slate-100 p-0.5", className)} role="radiogroup">
      {options.map((o) => (
        <button
          key={String(o.value)}
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            "rounded-md font-medium transition",
            size === "sm" ? "px-2 py-1 text-xs" : "px-3 py-1.5 text-sm",
            value === o.value ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({ checked, onChange, label, hint }) {
  return (
    <label className="flex cursor-pointer items-start gap-2.5 text-sm">
      <input type="checkbox" className="mt-0.5 h-4 w-4 rounded" checked={!!checked} onChange={(e) => onChange(e.target.checked)} />
      <span>
        <span className="text-slate-700">{label}</span>
        {hint && <span className="block text-xs text-slate-500">{hint}</span>}
      </span>
    </label>
  );
}

export function Stat({ label, value, sub }) {
  return (
    <div className="min-w-0 rounded-lg bg-slate-50 px-3 py-2">
      <div className="truncate text-[11px] font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className="truncate text-sm font-semibold text-slate-800 tabular-nums">{value}</div>
      {sub && <div className="truncate text-xs text-slate-500">{sub}</div>}
    </div>
  );
}
