import { useCallback, useEffect, useRef, useState } from "react";
import { MousePointerClick } from "lucide-react";
import { cx } from "./ui.jsx";

/**
 * Before/after viewer. Drag anywhere (or use arrow keys) to move the divider.
 * In pick mode, clicks report the clicked pixel in natural image coordinates instead.
 */
export default function CompareSlider({
  before, after, beforeLabel = "Before", afterLabel = "After",
  pickMode = false, onPick, marker, checker = false, busy = false, overlay = null,
}) {
  const [pos, setPos] = useState(50);
  const [dragging, setDragging] = useState(false);
  const [natural, setNatural] = useState(null);
  const boxRef = useRef(null);
  const imgRef = useRef(null);
  const single = !after;

  useEffect(() => { if (after) setPos(50); }, [after]);

  const posFromEvent = useCallback((e) => {
    const r = boxRef.current.getBoundingClientRect();
    return Math.min(100, Math.max(0, ((e.clientX - r.left) / r.width) * 100));
  }, []);

  const pick = (e) => {
    const img = imgRef.current;
    if (!img || !onPick) return;
    const r = img.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * img.naturalWidth;
    const y = ((e.clientY - r.top) / r.height) * img.naturalHeight;
    if (x >= 0 && y >= 0 && x < img.naturalWidth && y < img.naturalHeight) onPick([Math.floor(x), Math.floor(y)]);
  };

  const onPointerDown = (e) => {
    if (pickMode) return pick(e);
    if (single) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
    setPos(posFromEvent(e));
  };
  const onPointerMove = (e) => dragging && setPos(posFromEvent(e));
  const stop = () => setDragging(false);
  const onKey = (e) => {
    if (single) return;
    if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 5));
    if (e.key === "ArrowRight") setPos((p) => Math.min(100, p + 5));
  };

  const markerStyle = marker && natural
    ? { left: `${((marker[0] + 0.5) / natural[0]) * 100}%`, top: `${((marker[1] + 0.5) / natural[1]) * 100}%` }
    : null;

  return (
    <div className="flex w-full justify-center">
      <div
        ref={boxRef}
        className={cx(
          "relative block max-w-full select-none overflow-hidden rounded-lg bg-slate-100 touch-none",
          pickMode ? "cursor-crosshair" : single ? "" : "cursor-ew-resize",
          checker && "checkerboard",
        )}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={stop}
        onPointerCancel={stop}
        onKeyDown={onKey}
        style={natural ? { width: `min(100%, calc(62vh * ${natural[0] / natural[1]}))` } : undefined}
        tabIndex={single ? -1 : 0}
        role={single ? "img" : "slider"}
        aria-label={single ? beforeLabel : "Before and after comparison"}
        aria-valuenow={single ? undefined : Math.round(pos)}
      >
        {/* The "before" image defines the box size; "after" is layered on top and clipped. */}
        <img ref={imgRef} src={before} alt={beforeLabel} draggable={false}
             className={cx("block", natural ? "h-auto w-full" : "max-h-[62vh] max-w-full")} onLoad={(e) => setNatural([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])} />
        {after && (
          <img src={after} alt={afterLabel} draggable={false}
               className="pointer-events-none absolute inset-0 h-full w-full object-contain"
               style={{ clipPath: `inset(0 0 0 ${pos}%)` }} />
        )}
        {after && (
          <>
            <div className="pointer-events-none absolute inset-y-0 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.25)]" style={{ left: `${pos}%` }}>
              {!pickMode && (
                <div className="absolute top-1/2 left-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-slate-300 bg-white text-xs text-slate-500 shadow">
                  ◀▶
                </div>
              )}
            </div>
            <span className="pointer-events-none absolute top-2 left-2 rounded bg-black/55 px-2 py-0.5 text-xs font-medium text-white">{beforeLabel}</span>
            <span className="pointer-events-none absolute top-2 right-2 rounded bg-brand-600/90 px-2 py-0.5 text-xs font-medium text-white">{afterLabel}</span>
          </>
        )}
        {markerStyle && (
          <span className="pointer-events-none absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-brand-600 shadow ring-2 ring-brand-600/40" style={markerStyle} />
        )}
        {pickMode && (
          <span className="pointer-events-none absolute bottom-2 left-1/2 flex -translate-x-1/2 items-center gap-1.5 rounded-full bg-black/65 px-3 py-1 text-xs font-medium text-white">
            <MousePointerClick className="h-3.5 w-3.5" /> Click on the object you want to select
          </span>
        )}
        {busy && <div className="absolute inset-0 flex items-center justify-center bg-white/60 backdrop-blur-[1px]">{overlay}</div>}
      </div>
    </div>
  );
}
