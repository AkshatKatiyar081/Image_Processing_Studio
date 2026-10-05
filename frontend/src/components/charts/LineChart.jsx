import { useEffect, useMemo, useRef, useState } from "react";

const M = { top: 14, right: 96, bottom: 42, left: 48 };
const fmtDefault = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

/** Width of a container in CSS pixels (so the SVG is drawn 1:1 and text stays readable on phones). */
function useWidth(initial = 600) {
  const ref = useRef(null);
  const [width, setWidth] = useState(initial);
  useEffect(() => {
    if (!ref.current) return undefined;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(240, Math.round(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const r = raw / mag;
  return (r >= 5 ? 10 : r >= 2 ? 5 : r >= 1 ? 2 : 1) * mag;
}

function linearTicks(lo, hi, count = 5) {
  if (hi <= lo) return [lo];
  const step = niceStep(hi - lo, count);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) out.push(+v.toPrecision(12));
    }
  }
  return out;
}

/** Push end labels apart vertically so they never overlap (min gap in px). */
function spreadLabels(labels, gap, top, bottom) {
  const s = [...labels].sort((a, b) => a.y - b.y);
  for (let i = 1; i < s.length; i++) if (s[i].y - s[i - 1].y < gap) s[i].y = s[i - 1].y + gap;
  const over = s.length ? s[s.length - 1].y - bottom : 0;
  if (over > 0) s.forEach((l) => { l.y -= over; });
  for (let i = s.length - 2; i >= 0; i--) if (s[i + 1].y - s[i].y < gap) s[i].y = s[i + 1].y - gap;
  s.forEach((l) => { l.y = Math.max(top, l.y); });
  return s;
}

/**
 * Small responsive line chart (hand-written SVG, no library).
 * series: [{ id, label, color, points: [{ x, y }] }]   color may be a CSS variable, e.g. "var(--series-dct)".
 * marker: { x, label } draws a dashed guide line and rings the points at that x.
 * showPoints: draw a dot at every data point (turn off for dense curves).
 * endLabels: write each series' name at its last point (turn off when curves end inside others).
 */
export default function LineChart({
  series, xLabel, yLabel, xScale = "linear", yDomain, marker, formatX = fmtDefault, formatY = fmtDefault,
  height = 240, ariaLabel, showPoints = true, endLabels = true,
}) {
  const [boxRef, width] = useWidth();
  const [hover, setHover] = useState(null);
  const svgRef = useRef(null);
  const isLog = xScale === "log";

  const geo = useMemo(() => {
    const all = series.flatMap((s) => s.points);
    const xs = all.map((p) => p.x).filter((x) => (isLog ? x > 0 : Number.isFinite(x)));
    const ys = all.map((p) => p.y).filter(Number.isFinite);
    let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    if (marker && Number.isFinite(marker.x) && (!isLog || marker.x > 0)) { x0 = Math.min(x0, marker.x); x1 = Math.max(x1, marker.x); }
    if (x0 === x1) { x0 = isLog ? x0 / 2 : x0 - 1; x1 = isLog ? x1 * 2 : x1 + 1; }
    let [y0, y1] = yDomain || [Math.min(...ys), Math.max(...ys)];
    if (!yDomain) { const pad = (y1 - y0) * 0.08 || 1; y0 -= pad; y1 += pad; }
    const t = (x) => (isLog ? Math.log10(x) : x);
    const iw = Math.max(40, width - M.left - M.right);
    const ih = height - M.top - M.bottom;
    const sx = (x) => M.left + ((t(x) - t(x0)) / (t(x1) - t(x0))) * iw;
    const sy = (y) => M.top + (1 - (y - y0) / (y1 - y0)) * ih;
    const xTicks = isLog ? logTicks(x0, x1) : linearTicks(x0, x1, Math.max(3, Math.floor(iw / 80)));
    const yTicks = linearTicks(y0, y1, 4);
    const uniqueX = [...new Set(xs)].sort((a, b) => a - b);
    return { x0, x1, y0, y1, sx, sy, iw, ih, xTicks, yTicks, uniqueX };
  }, [series, width, height, yDomain, isLog, marker]);

  const { sx, sy, iw, ih, xTicks, yTicks, uniqueX } = geo;
  const near = (a, b) => Math.abs(a - b) <= Math.abs(b) * 1e-6 + 1e-9;

  const valueAt = (s, x) => {
    let best = null;
    for (const p of s.points) if (!best || Math.abs(sx(p.x) - sx(x)) < Math.abs(sx(best.x) - sx(x))) best = p;
    return best;
  };

  const onMove = (e) => {
    const r = svgRef.current.getBoundingClientRect();
    const px = e.clientX - r.left;
    let best = null;
    for (const x of uniqueX) if (best === null || Math.abs(sx(x) - px) < Math.abs(sx(best) - px)) best = x;
    setHover(best);
  };

  const placedLabels = !endLabels ? [] : spreadLabels(
    series.filter((s) => s.points.length).map((s) => {
      const last = s.points[s.points.length - 1];
      return { id: s.id, text: s.label, color: s.color, x: sx(last.x) + 8, y: sy(last.y) };
    }),
    13, M.top + 4, M.top + ih,
  );

  const markerX = marker && Number.isFinite(marker.x) ? sx(marker.x) : null;
  const hoverLeft = hover !== null ? Math.min(Math.max(sx(hover), 70), width - 70) : 0;

  return (
    <div ref={boxRef} className="relative w-full">
      <div className="mb-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-600" aria-hidden="true">
        {series.map((s) => (
          <span key={s.id} className="flex items-center gap-1.5">
            <span className="h-0.5 w-4 rounded" style={{ background: s.color }} />{s.label}
          </span>
        ))}
        {marker?.label && (
          <span className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-full border-2 border-slate-500 bg-white" />{marker.label}
          </span>
        )}
      </div>
      <svg ref={svgRef} width={width} height={height} className="block max-w-full touch-pan-y select-none"
           role="img" aria-label={ariaLabel || `${yLabel} versus ${xLabel}`}
           onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        {/* grid + y axis */}
        {yTicks.map((v) => (
          <g key={`y${v}`}>
            <line x1={M.left} x2={M.left + iw} y1={sy(v)} y2={sy(v)} style={{ stroke: "var(--chart-grid)" }} />
            <text x={M.left - 6} y={sy(v)} dy="0.32em" textAnchor="end" fontSize="11" style={{ fill: "var(--chart-ink)" }}>{formatY(v)}</text>
          </g>
        ))}
        {xTicks.map((v) => (
          <g key={`x${v}`}>
            <line x1={sx(v)} x2={sx(v)} y1={M.top} y2={M.top + ih} style={{ stroke: "var(--chart-grid)" }} strokeDasharray="2 3" />
            <text x={sx(v)} y={M.top + ih + 16} textAnchor="middle" fontSize="11" style={{ fill: "var(--chart-ink)" }}>{formatX(v)}</text>
          </g>
        ))}
        <line x1={M.left} x2={M.left + iw} y1={M.top + ih} y2={M.top + ih} stroke="#94a3b8" />
        <text x={M.left + iw / 2} y={height - 6} textAnchor="middle" fontSize="11" fontWeight="600" style={{ fill: "var(--chart-ink)" }}>{xLabel}</text>
        <text transform={`translate(12 ${M.top + ih / 2}) rotate(-90)`} textAnchor="middle" fontSize="11" fontWeight="600" style={{ fill: "var(--chart-ink)" }}>{yLabel}</text>

        {/* marker (user's image / setting) */}
        {markerX !== null && markerX >= M.left - 1 && markerX <= M.left + iw + 1 && (
          <g>
            <line x1={markerX} x2={markerX} y1={M.top} y2={M.top + ih} stroke="#64748b" strokeDasharray="4 3" />
            {series.map((s) => s.points.filter((p) => near(p.x, marker.x)).map((p) => (
              <circle key={`${s.id}m`} cx={sx(p.x)} cy={sy(p.y)} r="8" fill="none" stroke="#475569" strokeWidth="2" />
            )))}
          </g>
        )}

        {/* lines + points */}
        {series.map((s) => (
          <g key={s.id}>
            <path d={s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(" ")}
                  fill="none" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" style={{ stroke: s.color }} />
            {showPoints && s.points.map((p, i) => (
              <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r="4" stroke="#fff" strokeWidth="2" style={{ fill: s.color }} />
            ))}
          </g>
        ))}

        {/* direct labels at line ends */}
        {placedLabels.map((l) => (
          <text key={l.id} x={l.x} y={l.y} dy="0.32em" fontSize="11" fontWeight="600" style={{ fill: l.color }}>{l.text}</text>
        ))}

        {/* hover crosshair */}
        {hover !== null && (
          <line x1={sx(hover)} x2={sx(hover)} y1={M.top} y2={M.top + ih} stroke="#0f172a" strokeOpacity="0.35" />
        )}
        <rect x={M.left} y={M.top} width={iw} height={ih} fill="transparent" />
      </svg>

      {hover !== null && (
        <div className="pointer-events-none absolute top-6 z-10 -translate-x-1/2 rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs whitespace-nowrap text-slate-700 shadow"
             style={{ left: hoverLeft }}>
          <div className="mb-0.5 font-semibold">{xLabel}: {formatX(hover)}</div>
          {series.map((s) => {
            const p = valueAt(s, hover);
            return p && (
              <div key={s.id} className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
                {s.label}: <b className="tabular-nums">{formatY(p.y)}</b>
              </div>
            );
          })}
        </div>
      )}

      {/* A table ignores width: 1px, so the visually-hidden wrapper is a div (keeps phones from scrolling sideways). */}
      <div className="sr-only">
      <table>
        <caption>{ariaLabel || `${yLabel} versus ${xLabel}`}</caption>
        <thead><tr><th>{xLabel}</th>{series.map((s) => <th key={s.id}>{s.label}</th>)}</tr></thead>
        <tbody>
          {uniqueX.map((x) => (
            <tr key={x}><td>{formatX(x)}</td>{series.map((s) => {
              const p = s.points.find((q) => near(q.x, x));
              return <td key={s.id}>{p ? formatY(p.y) : "–"}</td>;
            })}</tr>
          ))}
        </tbody>
      </table>
      </div>
    </div>
  );
}
