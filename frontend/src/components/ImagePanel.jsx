import { useEffect, useRef, useState } from "react";
import { Image as ImageIcon, Upload, Undo2 } from "lucide-react";
import { api, apiUrl } from "../lib/api.js";
import { limitsText } from "../lib/image.js";
import { Button, Card, ErrorBanner, cx } from "./ui.jsx";

/** Upload area + sample gallery. When an image is loaded it collapses to a compact summary. */
export default function ImagePanel({ image, onFile, onReset, canUndo, onUndo, loading, limits, error, onCloseError }) {
  const [samples, setSamples] = useState([]);
  const [drag, setDrag] = useState(false);
  const [open, setOpen] = useState(!image);
  const input = useRef(null);

  useEffect(() => { api.samples().then(setSamples).catch(() => {}); }, []);
  useEffect(() => { setOpen(!image); }, [image?.id]);
  useEffect(() => { if (error) setOpen(true); }, [error]);   // e.g. a pasted file was refused: show why

  // Paste an image from the clipboard (Ctrl/Cmd+V) anywhere on the page
  useEffect(() => {
    const onPaste = (e) => {
      const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
      if (f) onFile(f, f.name || "pasted.png");
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [onFile]);

  const pickSample = async (s) => {
    const blob = await (await fetch(apiUrl(s.url))).blob();
    onFile(new File([blob], `${s.id}.png`, { type: "image/png" }), `${s.id}.png`, s);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f, f.name);
  };

  return (
    <Card title="Image" icon={ImageIcon} tour="image"
          action={image && !open && <button className="text-xs font-medium text-brand-600 hover:underline" onClick={() => setOpen(true)}>Change</button>}>
      {image && !open ? (
        <div className="flex items-center gap-3">
          <img src={image.url} alt="" className="h-14 w-14 shrink-0 rounded-md border border-slate-200 object-cover" />
          <div className="min-w-0 flex-1 text-sm">
            <div className="truncate font-medium text-slate-800" title={image.name}>{image.name}</div>
            <div className="text-xs text-slate-500">
              {image.width} × {image.height} px{image.resized ? ` (shrunk for upload from ${image.originalSize[0]} × ${image.originalSize[1]})` : ""}
            </div>
            {image.generation > 0 && <div className="text-xs text-brand-700">Edited {image.generation}× (working on a result)</div>}
          </div>
          <div className="flex flex-col gap-1">
            {canUndo && <Button size="sm" variant="secondary" onClick={onUndo} title="Go back one edit"><Undo2 className="h-3.5 w-3.5" />Undo</Button>}
            {image.generation > 0 && <Button size="sm" variant="ghost" onClick={onReset}>Original</Button>}
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
            onClick={() => input.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
            role="button"
            tabIndex={0}
            className={cx(
              "flex cursor-pointer flex-col items-center gap-1.5 rounded-lg border-2 border-dashed px-4 py-6 text-center transition",
              drag ? "border-brand-500 bg-brand-50" : "border-slate-300 hover:border-brand-500 hover:bg-slate-50",
            )}
          >
            <Upload className="h-6 w-6 text-slate-400" />
            <div className="text-sm font-medium text-slate-700">{loading ? "Loading…" : "Drop an image or click to browse"}</div>
            <div className="text-xs text-slate-500">PNG, JPEG, WebP, BMP · you can also paste</div>
            {limits && <div className="text-xs font-medium text-slate-600">Max {limitsText(limits)}</div>}
            <input ref={input} type="file" accept="image/*" className="hidden"
                   onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f, f.name); e.target.value = ""; }} />
          </div>
          {/* right under the drop zone, so a refused file (e.g. over the size limit) is explained where the user looks */}
          <ErrorBanner error={error} onClose={onCloseError} />
          {samples.length > 0 && (
            <div>
              <div className="mb-1.5 text-xs font-semibold text-slate-500">Or try a sample</div>
              <div className="grid grid-cols-3 gap-2">
                {samples.map((s) => (
                  <button key={s.id} onClick={() => pickSample(s)} title={s.hint}
                          className="group overflow-hidden rounded-lg border border-slate-200 bg-white text-left transition hover:border-brand-500 hover:shadow-sm">
                    <img src={apiUrl(s.url)} alt={s.title} loading="lazy" className="aspect-square w-full object-cover" />
                    <div className="truncate px-1.5 py-1 text-[11px] font-medium text-slate-600 group-hover:text-brand-700">{s.title}</div>
                  </button>
                ))}
              </div>
            </div>
          )}
          {image && <Button variant="ghost" size="sm" className="w-full" onClick={() => setOpen(false)}>Keep current image</Button>}
        </div>
      )}
    </Card>
  );
}
