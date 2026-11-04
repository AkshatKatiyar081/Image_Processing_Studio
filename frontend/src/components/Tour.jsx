import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { Button, cx } from "./ui.jsx";

const SEEN_KEY = "ips-tour-seen";
const PAD = 8;              // space between the highlighted element and the spotlight edge
const CARD_W = 340;
const GAP = 12;             // space between the spotlight and the card

/** Was the tour already shown in this browser? (Storage can be blocked: then it is simply shown again.) */
export function tourSeen() {
  try { return window.localStorage.getItem(SEEN_KEY) === "1"; } catch { return false; }
}

function markSeen() {
  try { window.localStorage.setItem(SEEN_KEY, "1"); } catch { /* private mode etc. */ }
}

const mb = (bytes) => `${(bytes / 1024 / 1024).toFixed(1)} MB`;

function Limits({ limits }) {
  return (
    <ul className="mt-2 space-y-0.5 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
      <li><b>Formats:</b> PNG, JPEG, WebP, BMP</li>
      <li><b>Max file size:</b> {limits ? mb(limits.max_upload_bytes) : "shown once the server is online"}</li>
      <li><b>Max resolution:</b> {limits ? `${limits.max_pixels / 1e6} megapixels` : "shown once the server is online"}</li>
      <li>Bigger photo? Save it as a JPEG or resize it first.</li>
    </ul>
  );
}

/** The tour, in order. `target` is a data-tour value; `missing` is shown when that element is not on screen. */
const STEPS = [
  {
    title: "Welcome to Image Processing Studio",
    body: () => "This app checks your photo, fixes only what is wrong, and shows you the difference. This quick tour takes about a minute.",
  },
  {
    target: "image",
    tab: "tools",
    title: "Start with a photo",
    needsImage: true,
    body: ({ image, limits }) => (
      <>
        {image
          ? "You're all set: your photo is loaded. You can change it any time with “Change”."
          : "Drop a photo here, click to browse, paste one (Ctrl + V), or pick one of the samples below."}
        <Limits limits={limits} />
      </>
    ),
  },
  {
    target: "health",
    tab: "tools",
    title: "Image health",
    body: () => "We measure brightness, contrast and noise. If something is wrong, it appears here with a one-click fix button.",
  },
  {
    target: "tabs",
    title: "Three ways to work",
    body: () => (
      <ul className="space-y-1">
        <li><b>Tools:</b> choose an effect yourself.</li>
        <li><b>Compare:</b> see whether fixing the photo first gives a better result.</li>
        <li><b>Auto Fix:</b> one click does it for you.</li>
      </ul>
    ),
  },
  {
    target: "tools",
    tab: "tools",
    title: "Choose a tool",
    body: () => "Pick what you want to change: light & contrast, noise, sharpness & edges, or file size (compression).",
  },
  {
    target: "effects",
    tab: "tools",
    title: "Pick an effect",
    body: () => "Click a ready-made effect to apply it. Open “Advanced settings” to fine-tune it, then press Apply.",
  },
  {
    target: "result",
    tab: "tools",
    title: "See the difference",
    body: () => "Drag the divider to compare before and after, or switch to “Side by side”.",
  },
  {
    target: "actions",
    tab: "tools",
    title: "Save or keep going",
    body: () => "“Download” saves the full-quality result. “Keep editing this result” lets you add another effect on top.",
    missing: "After you pick an effect, “Download” and “Keep editing this result” appear under the picture. Download saves the full-quality result; Keep editing adds another effect on top.",
  },
  {
    target: "charts",
    tab: "tools",
    title: "Before and after numbers",
    body: () => "The histogram shows how the brightness changed, and “Result details” lists the numbers before and after.",
  },
  {
    target: "tab-auto",
    title: "Not sure what to do?",
    body: () => "Auto Fix measures your photo and applies the right fixes in the right order, explaining each step.",
  },
  {
    target: "tab-compare",
    title: "Compare",
    body: () => "Compare runs an operation twice, directly and after fixing your photo, and tells you which is better.",
  },
  {
    target: "help",
    title: "That's it!",
    body: () => "You can replay this tour any time with this button.",
  },
];

const find = (target) => {
  const el = target && document.querySelector(`[data-tour="${target}"]`);
  return el && el.getClientRects().length ? el : null;   // hidden elements (other tabs) have no boxes
};

/** Where the card goes: below the spotlight, else above, else beside it; centred when there is no target.
 *  On narrow screens it is docked to the bottom of the screen instead. */
function cardStyle(rect, cardH, narrow) {
  if (narrow) return { left: 12, right: 12, bottom: 12 };
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  if (!rect) return { left: (vw - CARD_W) / 2, top: Math.max(12, (vh - cardH) / 2) };
  const left = Math.min(Math.max(12, rect.left + rect.width / 2 - CARD_W / 2), vw - CARD_W - 12);
  if (rect.bottom + GAP + cardH < vh - 12) return { left, top: rect.bottom + GAP };
  if (rect.top - GAP - cardH > 12) return { left, top: rect.top - GAP - cardH };
  const side = rect.right + GAP + CARD_W < vw - 12 ? rect.right + GAP : Math.max(12, rect.left - GAP - CARD_W);
  return { left: side, top: Math.min(Math.max(12, rect.top), vh - cardH - 12) };
}

/** Guided tour: dims the page, highlights one part at a time and explains it. */
export default function Tour({ open, onClose, image, limits, onTab }) {
  const [i, setI] = useState(0);
  const [rect, setRect] = useState(null);
  const [cardH, setCardH] = useState(200);
  const [narrow, setNarrow] = useState(() => window.innerWidth < 640);
  const card = useRef(null);
  const nextBtn = useRef(null);
  const step = STEPS[i];
  const blocked = step.needsImage && !image;   // step 2 waits until a photo is loaded
  const last = i === STEPS.length - 1;

  const close = useCallback(() => { markSeen(); onClose(); }, [onClose]);
  const next = useCallback(() => { if (!blocked) setI((n) => (n < STEPS.length - 1 ? n + 1 : n)); }, [blocked]);
  const back = useCallback(() => setI((n) => Math.max(0, n - 1)), []);

  useEffect(() => { if (open) setI(0); }, [open]);

  // the photo arrived while waiting on step 2: move on by itself
  const hadImage = useRef(!!image);
  useEffect(() => {
    if (open && step.needsImage && image && !hadImage.current) setI((n) => n + 1);
    hadImage.current = !!image;
  }, [image, open, step.needsImage]);

  // bring the target into view, then follow it while the page scrolls or resizes
  useLayoutEffect(() => {
    if (!open) return undefined;
    if (step.tab) onTab?.(step.tab);           // e.g. the user clicked another tab during the tour
    let frame = 0;
    let scrolled = false;
    const update = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const t = find(step.target);
        if (t && !scrolled) {
          scrolled = true;
          // phones: the card is docked at the bottom, so put the target's top just below the sticky header
          const small = window.innerWidth < 640;
          if (small) t.style.scrollMarginTop = `${(document.querySelector("header")?.offsetHeight || 0) + 12}px`;
          t.scrollIntoView({ block: small ? "start" : "center", inline: "nearest", behavior: "smooth" });
        }
        setRect(t ? t.getBoundingClientRect() : null);
        setNarrow(window.innerWidth < 640);
        if (card.current) setCardH(card.current.offsetHeight);
      });
    };
    update();
    const timer = setInterval(update, 250);          // layout changes (images loading, panels opening)
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);  // capture: also the two scrolling columns
    return () => {
      cancelAnimationFrame(frame);
      clearInterval(timer);
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [open, i, step.target, step.tab, onTab]);

  useEffect(() => { if (open) nextBtn.current?.focus({ preventScroll: true }); }, [open, i]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape") close();
      else if (e.key === "ArrowRight") next();
      else if (e.key === "ArrowLeft") back();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, close, next, back]);

  if (!open) return null;

  const hole = rect && {
    left: rect.left - PAD, top: rect.top - PAD, width: rect.width + 2 * PAD, height: rect.height + 2 * PAD,
  };
  const text = step.target && !rect && step.missing ? step.missing : step.body({ image, limits });
  const shade = "pointer-events-auto fixed bg-slate-900/55";

  return (
    <div className="pointer-events-none fixed inset-0 z-50" aria-live="polite">
      {/* four blockers around the spotlight: the page outside is dimmed and not clickable, the highlighted part is */}
      {hole ? (
        <>
          <div className={shade} style={{ left: 0, top: 0, right: 0, height: Math.max(0, hole.top) }} />
          <div className={shade} style={{ left: 0, top: hole.top + hole.height, right: 0, bottom: 0 }} />
          <div className={shade} style={{ left: 0, top: hole.top, width: Math.max(0, hole.left), height: hole.height }} />
          <div className={shade} style={{ left: hole.left + hole.width, top: hole.top, right: 0, height: hole.height }} />
          <div className="pointer-events-none fixed rounded-xl ring-2 ring-brand-500 transition-all duration-200"
               style={hole} />
        </>
      ) : (
        <div className={cx(shade, "inset-0")} />
      )}

      <div ref={card} role="dialog" aria-modal="true" aria-labelledby="tour-title"
           className={cx("pointer-events-auto fixed rounded-xl border border-slate-200 bg-white p-4 shadow-xl", !narrow && "w-[340px]")}
           style={cardStyle(rect, cardH, narrow)}>
        <div className="mb-1 flex items-start justify-between gap-3">
          <h2 id="tour-title" className="text-sm font-bold text-slate-900">{step.title}</h2>
          <button onClick={close} aria-label="Close the tour" className="rounded p-0.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="text-sm leading-relaxed text-slate-600">{text}</div>
        {blocked && (
          <p className="mt-2 text-xs font-medium text-amber-700">Choose a sample or upload a photo to continue.</p>
        )}
        <div className="mt-3 flex items-center justify-between gap-2">
          <span className="text-xs text-slate-400">{i + 1} / {STEPS.length}</span>
          <div className="flex gap-2">
            {!last && <Button size="sm" variant="ghost" onClick={close}>Skip</Button>}
            {i > 0 && <Button size="sm" variant="secondary" onClick={back}>Back</Button>}
            <Button size="sm" ref={nextBtn} onClick={last ? close : next} disabled={blocked}>
              {last ? "Finish" : "Next"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
