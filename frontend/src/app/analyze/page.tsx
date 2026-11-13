// src/app/analyze/page.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import SonarMark from "@/components/SonarMark";
import { AnalysisResult, analyzeImage } from "@/lib/api";

const MAX_FILES = 10;
const STORAGE_KEY = "seafloor-upload-flags";
// Score cutoffs on the 0-1 PatchCore scale (provisional, tuned from test-set percentiles).
const PC_STRONG = 0.6;
const PC_MODERATE = 0.4;

const TILE_SIZES = [
  { value: 640, label: "640 px: same scale as the training data" },
  { value: 320, label: "320 px: zoomed in" },
  { value: 160, label: "160 px: zoomed in further" },
];

type FlagStatus = "confirmed_debris" | "false_positive" | "unsure";

type SavedFlag = {
  status: FlagStatus;
  image: string;
  row: number;
  col: number;
  tileSize: number;
  score: number;
};

const FLAG_OPTIONS: {
  value: FlagStatus;
  label: string;
  icon: string;
  active: string;
}[] = [
  { value: "confirmed_debris", label: "Debris", icon: "✓", active: "bg-red-400 text-black" },
  { value: "false_positive", label: "False positive", icon: "✕", active: "bg-ocean-600 text-white" },
  { value: "unsure", label: "Unsure", icon: "?", active: "bg-amber-400 text-black" },
];

type Item = {
  id: number;
  name: string;
  size: number;
  status: "waiting" | "running" | "done" | "error";
  result?: AnalysisResult;
  error?: string;
};

// A flag belongs to one tile of one uploaded file at one tile size.
function flagKey(name: string, size: number, tileSize: number, row: number, col: number): string {
  return `${name}|${size}|${tileSize}|${row}|${col}`;
}

function scoreColor(score: number): string {
  if (score > PC_STRONG) return "bg-red-400";
  if (score > PC_MODERATE) return "bg-amber-400";
  return "bg-teal-400";
}

function verdict(top: number): string {
  if (top > PC_STRONG) return "Strong anomaly candidate: worth a close look.";
  if (top > PC_MODERATE) return "Moderate anomaly score: worth a quick check.";
  return "Looks like ordinary seabed to the model.";
}

function ResultCard({
  item,
  flags,
  onFlag,
}: {
  item: Item;
  flags: Record<string, SavedFlag>;
  onFlag: (key: string, flag: SavedFlag | null) => void;
}) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const r = item.result;
  const statusLabel =
    item.status === "waiting"
      ? "Waiting"
      : item.status === "running"
      ? "Analyzing..."
      : item.status === "error"
      ? "Failed"
      : "Done";

  return (
    <section className="reveal rounded-xl border border-ocean-700 bg-ocean-800 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="truncate font-mono text-sm text-teal-300">{item.name}</h3>
        <span className="shrink-0 text-xs text-ocean-300">{statusLabel}</span>
      </div>

      {item.status === "error" && <p className="text-sm text-red-400">{item.error}</p>}

      {r && (
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <div className="relative overflow-hidden rounded-lg border border-ocean-700 bg-black">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={r.image} alt={`Uploaded sonar image ${item.name}`} className="block w-full" />
              {showHeatmap && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={r.heatmap}
                  alt="Reconstruction error heatmap"
                  className="fade-in pointer-events-none absolute inset-0 h-full w-full"
                />
              )}
              {/* Boxes around the three highest-scoring tiles (skipped when the image is one tile) */}
              {r.tiles.length > 1 &&
                r.tiles.slice(0, 3).map((t) => (
                  <div
                    key={t.rank}
                    className={`pointer-events-none absolute border-2 border-dashed ${
                      t.pc > PC_STRONG ? "border-red-400" : "border-amber-400"
                    }`}
                    style={{
                      left: `${(t.x / r.width) * 100}%`,
                      top: `${(t.y / r.height) * 100}%`,
                      width: `${(Math.min(r.tile_size, r.width - t.x) / r.width) * 100}%`,
                      height: `${(Math.min(r.tile_size, r.height - t.y) / r.height) * 100}%`,
                    }}
                  >
                    <span className="absolute left-0 top-0 bg-black/70 px-1 font-mono text-[10px] text-white">
                      #{t.rank}
                    </span>
                  </div>
                ))}
            </div>
            <button
              onClick={() => setShowHeatmap((v) => !v)}
              className="mt-2 rounded-full border border-ocean-600 px-3 py-1 text-xs text-ocean-300 hover:border-teal-400"
            >
              {showHeatmap ? "Hide heatmap" : "Show heatmap"}
            </button>
          </div>

          <div>
            <p className="mb-1 text-sm font-medium text-white">{verdict(r.tiles[0].pc)}</p>
            <p className="mb-4 text-xs text-ocean-300">
              {r.width} x {r.height} px, split into {r.tiles.length} tile{r.tiles.length === 1 ? "" : "s"}
              {r.was_resized ? " (shrunk to fit the size limit)" : ""}.
              {r.tiles.length === 1 && " Choose a smaller tile size to split this image into several tiles."}
            </p>
            <div className="flex flex-col gap-4">
              {r.tiles.slice(0, 5).map((t) => {
                const key = flagKey(item.name, item.size, r.tile_size, t.row, t.col);
                const current = flags[key]?.status;
                return (
                  <div key={t.rank}>
                    <div className="mb-1 flex justify-between text-xs">
                      <span className="text-ocean-300">
                        #{t.rank} · row {t.row + 1}, col {t.col + 1}
                      </span>
                      <span className="font-mono text-teal-300">{t.pc.toFixed(2)}</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-ocean-700">
                      <div
                        className={`bar-grow h-full rounded-full ${scoreColor(t.pc)}`}
                        style={{ width: `${Math.round(t.pc * 100)}%` }}
                      />
                    </div>
                    <p className="mt-0.5 font-mono text-[10px] text-ocean-300">
                      ranked by pretrained detector · autoencoder {t.ae.toFixed(2)} · brightness {t.baseline.toFixed(2)}
                    </p>
                    {/* Review buttons: click again to clear a choice */}
                    <div className="mt-1.5 flex gap-1.5">
                      {FLAG_OPTIONS.map((opt) => (
                        <button
                          key={opt.value}
                          onClick={() =>
                            onFlag(
                              key,
                              current === opt.value
                                ? null
                                : {
                                    status: opt.value,
                                    image: item.name,
                                    row: t.row,
                                    col: t.col,
                                    tileSize: r.tile_size,
                                    score: t.pc,
                                  }
                            )
                          }
                          className={`flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs transition ${
                            current === opt.value
                              ? `${opt.active} border-transparent font-medium`
                              : "border-ocean-600 text-ocean-300 hover:border-teal-400"
                          }`}
                        >
                          <span>{opt.icon}</span>
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

export default function AnalyzePage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [tileSize, setTileSize] = useState(640);
  const [items, setItems] = useState<Item[]>([]);
  const [running, setRunning] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [flags, setFlags] = useState<Record<string, SavedFlag>>({});

  // Load saved flags from this browser once on arrival.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) setFlags(JSON.parse(raw));
    } catch {
      // storage unavailable or corrupted: start with no flags
    }
  }, []);

  function saveFlags(next: Record<string, SavedFlag>) {
    setFlags(next);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      // storage unavailable: flags still work until the page is closed
    }
  }

  function setFlag(key: string, flag: SavedFlag | null) {
    const next = { ...flags };
    if (flag) next[key] = flag;
    else delete next[key];
    saveFlags(next);
  }

  function downloadCsv() {
    const header = "image,tile_row,tile_col,tile_size_px,pc_score,status";
    const lines = Object.values(flags).map((f) =>
      [`"${f.image.replace(/"/g, '""')}"`, f.row + 1, f.col + 1, f.tileSize, f.score.toFixed(3), f.status].join(",")
    );
    const blob = new Blob([[header, ...lines].join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "upload-flags.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  function chooseFiles(list: FileList | null) {
    if (!list) return;
    const images = Array.from(list).filter((f) => f.type.startsWith("image/"));
    setNotice(
      images.length > MAX_FILES
        ? `Only the first ${MAX_FILES} images are used.`
        : images.length < list.length
        ? "Some files were skipped because they are not images."
        : null
    );
    setFiles(images.slice(0, MAX_FILES));
    setItems([]);
  }

  function update(id: number, patch: Partial<Item>) {
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, ...patch } : it)));
  }

  async function run() {
    if (!files.length || running) return;
    setRunning(true);
    setItems(files.map((f, i) => ({ id: i, name: f.name, size: f.size, status: "waiting" })));
    // One at a time keeps the free server's memory low.
    for (let i = 0; i < files.length; i++) {
      update(i, { status: "running" });
      try {
        const result = await analyzeImage(files[i], tileSize);
        update(i, { status: "done", result });
      } catch (err) {
        update(i, {
          status: "error",
          error: err instanceof Error ? err.message : "Analysis failed",
        });
      }
    }
    setRunning(false);
  }

  const saved = Object.values(flags);
  const count = (s: FlagStatus) => saved.filter((f) => f.status === s).length;

  return (
    <main className="mx-auto max-w-5xl px-6 py-8">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <SonarMark size={44} />
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-white">Analyze your own images</h1>
            <p className="mt-0.5 text-sm text-ocean-300">
              Upload side-scan sonar images and see which parts look anomalous.
            </p>
          </div>
        </div>
        <Link
          href="/"
          className="rounded-full border border-ocean-600 px-3 py-1 text-sm text-ocean-300 hover:border-teal-400"
        >
          Back to review queue
        </Link>
      </header>

      <section className="mb-8 rounded-xl border border-ocean-700 bg-ocean-800 p-5">
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            chooseFiles(e.dataTransfer.files);
          }}
          className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-ocean-600 p-8 text-center"
        >
          <p className="text-sm text-ocean-300">
            Drag images here, or choose up to {MAX_FILES} files (JPG or PNG, 8 MB each).
          </p>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => chooseFiles(e.target.files)}
          />
          <button
            onClick={() => inputRef.current?.click()}
            className="rounded-full border border-teal-400 px-4 py-1.5 text-sm text-teal-300 hover:bg-ocean-700"
          >
            Choose images
          </button>
          {files.length > 0 && (
            <p className="font-mono text-xs text-teal-300">
              {files.length} image{files.length === 1 ? "" : "s"} selected
            </p>
          )}
          {notice && <p className="text-xs text-amber-400">{notice}</p>}
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <label className="text-sm text-ocean-300" htmlFor="tile-size">
            Tile size
          </label>
          <select
            id="tile-size"
            value={tileSize}
            onChange={(e) => setTileSize(Number(e.target.value))}
            className="rounded-lg border border-ocean-600 bg-ocean-900 px-3 py-1.5 text-sm text-white"
          >
            {TILE_SIZES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
          <button
            onClick={run}
            disabled={!files.length || running}
            className="ml-auto rounded-full bg-teal-400 px-5 py-1.5 text-sm font-medium text-black hover:bg-teal-300 disabled:opacity-40"
          >
            {running ? "Analyzing..." : "Analyze"}
          </button>
        </div>

        <p className="mt-4 text-xs text-ocean-300">
          The model was trained on one sonar dataset, where each training tile was a whole image shrunk
          to 64 px. Images that look like that data (640 px works best) give the most reliable
          scores. Very different sonar types or seabeds will give a rougher guide. Scores use the same
          0 to 1 scale as the review queue. The first request after a quiet period can take up to a
          minute while the free server wakes up.
        </p>
      </section>

      {/* Summary of the decisions made so far, with CSV download */}
      {saved.length > 0 && (
        <div className="reveal mb-6 flex flex-wrap items-center gap-x-5 gap-y-2 rounded-xl border border-ocean-700 bg-ocean-800 px-4 py-3 text-sm">
          <span className="text-ocean-300">Your review:</span>
          <span className="text-red-400">{count("confirmed_debris")} debris</span>
          <span className="text-ocean-300">{count("false_positive")} false positive</span>
          <span className="text-amber-400">{count("unsure")} unsure</span>
          <span className="ml-auto flex gap-2">
            <button
              onClick={downloadCsv}
              className="rounded-full border border-teal-400 px-3 py-1 text-xs text-teal-300 hover:bg-ocean-700"
            >
              Download flags (CSV)
            </button>
            <button
              onClick={() => saveFlags({})}
              className="rounded-full border border-ocean-600 px-3 py-1 text-xs text-ocean-300 hover:border-red-400 hover:text-red-400"
            >
              Clear flags
            </button>
          </span>
        </div>
      )}

      <div className="flex flex-col gap-6">
        {items.map((item) => (
          <ResultCard key={item.id} item={item} flags={flags} onFlag={setFlag} />
        ))}
      </div>
    </main>
  );
}