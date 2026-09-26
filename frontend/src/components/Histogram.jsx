import { useMemo, useRef, useState } from "react";

const W = 512, H = 120;
const BEFORE = "#8a8983";   // neutral: the reference series
const AFTER = "#2a78d6";    // brand blue: the result

function path(values, cap, close) {
  const pts = values.map((v, i) => `${(i / 255) * W},${H - Math.min(1, v / cap) * (H - 4)}`);
  return close ? `M0,${H} L${pts.join(" L")} L${W},${H} Z` : `M${pts.join(" L")}`;
}

/** 256-bin intensity histogram(s) of the luminance channel, before vs after. */
export default function Histogram({ before, after, title = "Brightness histogram" }) {
  const [hover, setHover] = useState(null);
  const ref = useRef(null);

  const cap = useMemo(() => {
    // Ignore the two tallest spikes (e.g. pure black/white) so the shape stays readable.
    const all = [...(before || []), ...(after || [])].sort((a, b) => b - a);
    return Math.max(1, all[Math.min(2, all.length - 1)] || 1) * 1.05;
  }, [before, after]);

  if (!before) return null;

  const onMove = (e) => {
    const r = ref.current.getBoundingClientRect();
    const bin = Math.round(((e.clientX - r.left) / r.width) * 255);
    setHover(Math.min(255, Math.max(0, bin)));
  };

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="text-xs font-semibold text-slate-600">{title}</span>
        <div className="flex items-center gap-3 text-xs text-slate-500">
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: BEFORE, opacity: 0.55 }} />Before</span>
          {after && <span className="flex items-center gap-1.5"><span className="h-0.5 w-3 rounded" style={{ background: AFTER }} />After</span>}
        </div>
      </div>
      <div className="relative">
        <svg ref={ref} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-24 w-full cursor-crosshair rounded-md bg-slate-50"
             onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img"
             aria-label={`${title}: intensity distribution before${after ? " and after" : ""} processing`}>
          <path d={path(before, cap, true)} fill={BEFORE} fillOpacity="0.35" stroke={BEFORE} strokeOpacity="0.6" strokeWidth="1" vectorEffect="non-scaling-stroke" />
          {after && <path d={path(after, cap, false)} fill="none" stroke={AFTER} strokeWidth="2" vectorEffect="non-scaling-stroke" strokeLinejoin="round" />}
          {hover !== null && <line x1={(hover / 255) * W} x2={(hover / 255) * W} y1="0" y2={H} stroke="#0b0b0b" strokeOpacity="0.35" strokeWidth="1" vectorEffect="non-scaling-stroke" />}
        </svg>
        {hover !== null && (
          <div className="pointer-events-none absolute -top-2 z-10 -translate-y-full rounded-md border border-slate-200 bg-white px-2 py-1 text-xs whitespace-nowrap text-slate-700 shadow"
               style={{ left: `${Math.min(80, Math.max(0, (hover / 255) * 100 - 10))}%` }}>
            Level <b>{hover}</b> · before {before[hover].toLocaleString()}
            {after && <> · after <span className="font-semibold">{after[hover].toLocaleString()}</span></>} px
          </div>
        )}
        <div className="mt-0.5 flex justify-between text-[10px] text-slate-400"><span>0 (black)</span><span>255 (white)</span></div>
      </div>
    </div>
  );
}
