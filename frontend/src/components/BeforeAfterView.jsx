import CompareSlider from "./CompareSlider.jsx";
import { Segmented, cx } from "./ui.jsx";

/** "Slider | Side by side" switch for a before/after result. */
export function ViewToggle({ value, onChange }) {
  return (
    <Segmented size="sm" value={value} onChange={onChange}
               options={[{ value: "slider", label: "Slider" }, { value: "side", label: "Side by side" }]} />
  );
}

/**
 * Before/after viewer: the draggable slider, or both images next to each other.
 * Without an "after" image (nothing run yet) it always shows the single "before" image.
 */
export default function BeforeAfterView({ view, before, after, beforeLabel = "Before", afterLabel = "After", busy, overlay }) {
  if (view !== "side" || !after) {
    return <CompareSlider before={before} after={after} beforeLabel={beforeLabel} afterLabel={afterLabel} busy={busy} overlay={overlay} />;
  }
  return (
    <div className="relative">
      <div className="grid gap-3 sm:grid-cols-2">
        {[[before, beforeLabel, false], [after, afterLabel, true]].map(([url, label, highlight]) => (
          <figure key={label} className={cx("min-w-0 overflow-hidden rounded-lg border", highlight ? "border-brand-300" : "border-slate-200")}>
            <div className="flex items-center justify-center bg-slate-100">
              <img src={url} alt={label} className="block max-h-[62vh] w-full object-contain" />
            </div>
            <figcaption className={cx("px-3 py-1.5 text-sm font-semibold", highlight ? "text-brand-700" : "text-slate-700")}>{label}</figcaption>
          </figure>
        ))}
      </div>
      {busy && <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-white/60 backdrop-blur-[1px]">{overlay}</div>}
    </div>
  );
}
