/**
 * Grouped horizontal bars: one row per measure, one bar per series on a shared scale per row.
 * series: [{ label, color }, ...]                     (2 or more)
 * rows:   [{ label, values: [v0, v1, ...] | a, b, format?: (v) => string, better?: "lower" | "higher", hint? }]
 * The best value of a row gets a ✓ when `better` is set.
 */
export default function PairedBars({ series, rows }) {
  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600" aria-hidden="true">
        {series.map((s) => (
          <span key={s.label} className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: s.color }} />{s.label}
          </span>
        ))}
      </div>
      <dl className="space-y-3">
        {rows.map((r) => {
          const values = r.values || [r.a, r.b];
          const fmt = r.format || ((v) => String(v));
          const known = values.filter((v) => v != null);
          const max = Math.max(...known.map(Math.abs), 0) || 1;
          let best = -1;
          if (r.better && known.length > 1 && new Set(known).size > 1) {
            const target = r.better === "lower" ? Math.min(...known) : Math.max(...known);
            best = values.indexOf(target);
          }
          return (
            <div key={r.label}>
              <dt className="mb-1 flex flex-wrap items-baseline justify-between gap-x-2 text-xs">
                <span className="font-semibold text-slate-700">{r.label}</span>
                {r.better && <span className="text-slate-400">{r.better} is better{r.hint ? ` · ${r.hint}` : ""}</span>}
                {!r.better && r.hint && <span className="text-slate-400">{r.hint}</span>}
              </dt>
              <dd className="space-y-0.5">
                {values.map((v, i) => (
                  <div key={series[i]?.label || i} className="flex items-center gap-2">
                    <div className="h-3 min-w-0 flex-1">
                      <div className="h-3 rounded-[4px]"
                           style={{ width: `${v == null ? 0 : Math.max(0.5, (Math.abs(v) / max) * 100)}%`, background: series[i]?.color }}
                           title={`${series[i]?.label}: ${v == null ? "–" : fmt(v)}`} />
                    </div>
                    <span className="w-24 shrink-0 text-right text-xs text-slate-700 tabular-nums">
                      {v == null ? "–" : fmt(v)}{best === i && <span className="ml-1 text-emerald-600" aria-label="better">✓</span>}
                    </span>
                  </div>
                ))}
              </dd>
            </div>
          );
        })}
      </dl>
    </div>
  );
}
