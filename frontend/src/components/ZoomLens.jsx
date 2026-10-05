import { useRef, useState } from "react";

const LENS = 120;   // lens size in px
const ZOOM = 3;

/**
 * A grid of same-sized images with a linked magnifier: pointing at one image shows the SAME region,
 * magnified 3x, on every image, so small artefacts (ringing, blockiness) can be compared directly.
 * items: [{ url, label, caption? }]
 */
export default function ZoomLens({ items, columns = 2 }) {
  const refs = useRef([]);
  const [pos, setPos] = useState(null);   // { fx, fy } as fractions of the image box

  const move = (e, i) => {
    const r = refs.current[i]?.getBoundingClientRect();
    if (!r) return;
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    setPos(fx < 0 || fy < 0 || fx > 1 || fy > 1 ? null : { fx, fy });
  };

  return (
    <div>
      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, ${columns > 2 ? 160 : 220}px), 1fr))` }}>
        {items.map((it, i) => {
          const img = refs.current[i];
          const w = img?.clientWidth || 0, h = img?.clientHeight || 0;
          return (
            <figure key={it.label} className="min-w-0">
              <div className="relative cursor-crosshair overflow-hidden rounded-md border border-slate-200 bg-slate-100"
                   onPointerMove={(e) => move(e, i)} onPointerDown={(e) => move(e, i)} onPointerLeave={() => setPos(null)}>
                <img ref={(el) => { refs.current[i] = el; }} src={it.url} alt={it.label} draggable={false} className="block h-auto w-full" />
                {pos && w > 0 && (
                  <div className="pointer-events-none absolute rounded-md border-2 border-white shadow-[0_0_0_1px_rgba(0,0,0,0.35)]"
                       style={{
                         width: LENS, height: LENS,
                         left: Math.min(Math.max(pos.fx * w - LENS / 2, 0), Math.max(0, w - LENS)),
                         top: Math.min(Math.max(pos.fy * h - LENS / 2, 0), Math.max(0, h - LENS)),
                         backgroundImage: `url(${it.url})`, backgroundRepeat: "no-repeat",
                         backgroundSize: `${w * ZOOM}px ${h * ZOOM}px`,
                         backgroundPosition: `${-(pos.fx * w * ZOOM - LENS / 2)}px ${-(pos.fy * h * ZOOM - LENS / 2)}px`,
                         imageRendering: "pixelated",
                       }} />
                )}
              </div>
              <figcaption className="mt-1 text-xs font-medium text-slate-700">{it.label}
                {it.caption && <span className="block font-normal text-slate-500">{it.caption}</span>}
              </figcaption>
            </figure>
          );
        })}
      </div>
      <p className="mt-2 text-[11px] text-slate-400">Point at any image: the same spot is magnified {ZOOM}× on all of them.</p>
    </div>
  );
}
