import { useCallback, useEffect, useRef, useState } from "react";
import { GitCompare, SlidersHorizontal, WandSparkles } from "lucide-react";
import { api, waitForServer } from "./lib/api.js";
import { limitProblem, lumaHistogram, prepareImage } from "./lib/image.js";
import AutoTab from "./components/AutoTab.jsx";
import CompareTab from "./components/CompareTab.jsx";
import HealthCard from "./components/HealthCard.jsx";
import ImagePanel from "./components/ImagePanel.jsx";
import ToolsTab from "./components/ToolsTab.jsx";
import { Button, cx } from "./components/ui.jsx";

const TABS = [
  { id: "tools", label: "Tools", icon: SlidersHorizontal },
  { id: "compare", label: "Compare", icon: GitCompare },
  { id: "auto", label: "Auto Fix", icon: WandSparkles },
];

const DESKTOP_QUERY = "(min-width: 1024px)";   // Tailwind's lg breakpoint

/** true on wide screens, where the page is split into two independently scrolling columns. */
function useIsDesktop() {
  const [desktop, setDesktop] = useState(() => window.matchMedia(DESKTOP_QUERY).matches);
  useEffect(() => {
    const mq = window.matchMedia(DESKTOP_QUERY);
    const onChange = (e) => setDesktop(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return desktop;
}

function Footer() {
  return (
    <footer className="px-4 pt-2 pb-6 text-center text-xs text-slate-400">
      Image &amp; Video Processing mini project · NMIMS MPSTME Shirpur · every algorithm also implemented from scratch in NumPy
    </footer>
  );
}

/** Desktop: two columns that scroll on their own (scrollbars hidden): image + health + tab controls on the
 *  left, results on the right; the mouse wheel moves the column under the cursor.
 *  Mobile: one normal page: image + health, then the result viewer, then the controls, so results are
 *  visible without scrolling past every control. */
function Layout({ left, controls, main }) {
  const desktop = useIsDesktop();
  if (desktop) {
    return (
      <div className="grid h-full min-h-0 w-full grid-cols-[360px_minmax(0,1fr)] gap-4">
        <aside data-scroll-column className="scroll-hidden min-h-0 space-y-4 overflow-y-auto overscroll-contain pb-6">
          {left}
          {controls}
        </aside>
        <main data-scroll-column className="scroll-hidden min-h-0 min-w-0 overflow-y-auto overscroll-contain">
          {main}
          <Footer />
        </main>
      </div>
    );
  }
  return (
    <div className="grid w-full gap-4">
      <aside className="space-y-4">{left}</aside>
      <main className="min-w-0">{main}</main>
      <aside className="space-y-4">{controls}</aside>
    </div>
  );
}

function ServerStatus({ status, onRetry }) {
  const map = {
    connecting: ["bg-amber-400 animate-pulse", "Waking server…"],
    online: ["bg-emerald-500", "Online"],
    offline: ["bg-red-500", "Offline"],
  };
  const [dot, text] = map[status];
  return (
    <button onClick={status === "offline" ? onRetry : undefined} title={status === "offline" ? "Retry" : "Server status"}
            className="flex items-center gap-1.5 rounded-full px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100">
      <span className={cx("h-2 w-2 rounded-full", dot)} />
      {text}
    </button>
  );
}

let nextId = 1;

export default function App() {
  const desktop = useIsDesktop();
  const [tab, setTab] = useState("tools");
  const [server, setServer] = useState("connecting");
  const [schemas, setSchemas] = useState(null);
  const [limits, setLimits] = useState(null);      // the server's upload / pixel limits (from /api/health)
  const [image, setImage] = useState(null);
  const [original, setOriginal] = useState(null);
  const [history, setHistory] = useState([]);
  const [health, setHealth] = useState({ data: null, loading: false, error: null });
  const [beforeHist, setBeforeHist] = useState(null);
  const [fixRequest, setFixRequest] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [loading, setLoading] = useState(false);
  const healthCtrl = useRef(null);

  const connect = useCallback(async () => {
    setServer("connecting");
    try {
      const h = await waitForServer();
      setLimits(h.limits || null);
      setServer("online");
      setSchemas(await api.modules());
    } catch {
      setServer("offline");
    }
  }, []);

  useEffect(() => { connect(); }, [connect]);

  // Diagnose + client-side histogram whenever the working image changes
  useEffect(() => {
    if (!image) return;
    lumaHistogram(image.url).then(setBeforeHist).catch(() => setBeforeHist(null));
    if (server !== "online") return;
    healthCtrl.current?.abort();
    const c = new AbortController();
    healthCtrl.current = c;
    setHealth({ data: null, loading: true, error: null });
    api.diagnose(image.file, c.signal)
      .then((d) => !c.signal.aborted && setHealth({ data: d, loading: false, error: null }))
      .catch((e) => e.name !== "AbortError" && setHealth({ data: null, loading: false, error: e.message }));
  }, [image?.id, server]); // eslint-disable-line

  const loadFile = useCallback(async (file, name) => {
    setLoadError(null);
    setLoading(true);
    try {
      const prepared = await prepareImage(file, name);
      // refuse files over the server's limits here, with a clear message, instead of failing on upload
      const problem = limitProblem(limits, { bytes: prepared.file.size, width: prepared.width, height: prepared.height });
      if (problem) {
        URL.revokeObjectURL(prepared.url);
        setLoadError(problem);
        return;
      }
      const img = { ...prepared, id: nextId++, generation: 0 };
      setImage(img);
      setOriginal(img);
      setHistory([]);
    } catch (e) {
      setLoadError(e.message || "Could not load that image.");
    } finally {
      setLoading(false);
    }
  }, [limits]);

  /** "Keep editing this result": `blob` is the FULL-QUALITY result file fetched from the server
   *  (never a preview), so chained edits do not lose quality. */
  const chain = useCallback(async (blob, label) => {
    const base = original?.name || "image";
    const ext = { "image/jpeg": "jpg", "image/webp": "webp" }[blob.type] || "png";
    const file = new File([blob], `${base}_edited.${ext}`, { type: blob.type });
    const prepared = await prepareImage(file, file.name);
    setHistory((h) => [...h, image]);
    setImage({ ...prepared, name: `${base} + ${label}`, id: nextId++, generation: (image?.generation || 0) + 1 });
    window.scrollTo({ top: 0, behavior: "smooth" });
    document.querySelectorAll("[data-scroll-column]").forEach((el) => el.scrollTo({ top: 0, behavior: "smooth" }));
  }, [image, original]);

  const undo = () => {
    if (!history.length) return;
    setImage(history[history.length - 1]);
    setHistory(history.slice(0, -1));
  };

  const reset = () => { setImage(original); setHistory([]); };

  const onFix = (tool, preset) => { setTab("tools"); setFixRequest({ tool, preset, n: Date.now() }); };

  const left = (
    <>
      <ImagePanel image={image} onFile={loadFile} onReset={reset} canUndo={history.length > 0} onUndo={undo} loading={loading}
                  limits={limits} error={loadError} onCloseError={() => setLoadError(null)} />
      <HealthCard health={health.data} loading={health.loading} error={health.error} onFix={onFix} />
    </>
  );

  // Only the visible tab gets the shared left column (avoids duplicate paste listeners etc.).
  const shared = (id) => ({ image, left: tab === id ? left : null, layout: Layout, beforeHist, onChain: chain });
  const tabBox = desktop ? "min-h-0 flex-1" : undefined;   // desktop: the tab fills the space below the header

  return (
    // Desktop: exactly one screen high (header + two scrolling columns). Mobile: a normal scrolling page.
    <div className={desktop ? "flex h-screen flex-col overflow-hidden" : "min-h-full"}>
      <header className="sticky top-0 z-20 shrink-0 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center justify-between gap-3 px-4 py-2.5">
          <div className="flex items-center gap-2.5">
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            <div>
              <div className="text-sm font-bold leading-tight text-slate-900">Image Processing Studio</div>
              <div className="text-[11px] leading-tight text-slate-500">Image enhancement &amp; diagnostics</div>
            </div>
          </div>
          <nav className="order-3 flex w-full gap-1 rounded-xl bg-slate-100 p-1 sm:order-none sm:w-auto" role="tablist">
            {TABS.map((t) => (
              <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}
                      className={cx("flex flex-1 items-center justify-center gap-1.5 rounded-lg px-4 py-1.5 text-sm font-medium transition sm:flex-none",
                        tab === t.id ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800")}>
                <t.icon className="h-4 w-4" />{t.label}
              </button>
            ))}
          </nav>
          <ServerStatus status={server} onRetry={connect} />
        </div>
      </header>

      <div className={cx("mx-auto w-full max-w-[1500px] px-4",
        desktop ? "flex min-h-0 flex-1 flex-col gap-4 pt-4" : "space-y-4 py-4")}>
        {server === "connecting" && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
            Connecting to the processing server… On the free hosting tier the first visit can take up to a minute while it wakes up.
          </div>
        )}
        {server === "offline" && (
          <div className="flex items-center justify-between gap-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
            The processing server is not responding.
            <Button size="sm" variant="secondary" onClick={connect}>Try again</Button>
          </div>
        )}

        {/* Tabs stay mounted so each keeps its result when you switch back and forth. */}
        <div hidden={tab !== "tools"} className={tabBox}><ToolsTab {...shared("tools")} schemas={schemas} request={fixRequest} /></div>
        <div hidden={tab !== "compare"} className={tabBox}><CompareTab {...shared("compare")} /></div>
        <div hidden={tab !== "auto"} className={tabBox}><AutoTab {...shared("auto")} /></div>
      </div>

      {!desktop && <Footer />}
    </div>
  );
}
