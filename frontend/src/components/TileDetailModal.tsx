// src/components/TileDetailModal.tsx
//
// Overlay shown when an analyst opens a tile: the image with a toggleable
// reconstruction-error heatmap, the score breakdown, and the flag buttons.
// Keyboard: 1 / 2 / 3 flag the tile, Esc closes (arrow keys are handled by the page).

import { useEffect, useState } from "react";
import { Tile, tileImageUrl, tileHeatmapUrl, submitFlag, FlagPayload } from "@/lib/api";

interface TileDetailModalProps {
  tile: Tile;
  onClose: () => void;
  onFlagged: (tileId: string, status: string) => void;
}

const FLAG_OPTIONS: {
  value: FlagPayload["status"];
  label: string;
  key: string;
  color: string;
}[] = [
  { value: "confirmed_debris", label: "Confirm Debris", key: "1", color: "bg-red-400 hover:bg-red-300" },
  { value: "false_positive", label: "False Positive", key: "2", color: "bg-ocean-600 text-white hover:bg-ocean-700" },
  { value: "unsure", label: "Unsure", key: "3", color: "bg-amber-400 hover:bg-amber-300" },
];

function scoreColor(score: number): string {
  if (score > 0.7) return "bg-red-400";
  if (score > 0.4) return "bg-amber-400";
  return "bg-teal-400";
}

export default function TileDetailModal({ tile, onClose, onFlagged }: TileDetailModalProps) {
  const [submitting, setSubmitting] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);

  async function handleFlag(status: FlagPayload["status"]) {
    setSubmitting(true);
    try {
      await submitFlag(tile.tile_id, { status });
      onFlagged(tile.tile_id, status);
    } catch (err) {
      console.error(err);
      alert("Failed to save the flag. The server may be waking up, so try again in a moment.");
    } finally {
      setSubmitting(false);
    }
  }

  // Keyboard shortcuts while the popup is open.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
      const option = FLAG_OPTIONS.find((o) => o.key === e.key);
      if (option && !submitting) handleFlag(option.value);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const scores = [
    { label: "Ensemble", value: tile.ensemble_score },
    { label: "Autoencoder", value: tile.ae_score },
    { label: "Brightness", value: tile.baseline_score },
  ];

  return (
    <div
      className="fade-in fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="reveal max-h-full w-full max-w-lg overflow-y-auto rounded-xl border border-ocean-700 bg-ocean-800 p-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-mono text-sm text-teal-300">Tile #{tile.rank}</h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="rounded px-2 text-ocean-300 hover:bg-ocean-700 hover:text-white"
          >
            ✕
          </button>
        </div>

        {/* Image with the heatmap stacked on top; both are the same square size */}
        <div className="relative mb-2 overflow-hidden rounded-lg border border-ocean-700 bg-black">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={tileImageUrl(tile.tile_id)}
            alt={`Sonar tile ${tile.tile_id}`}
            className="block w-full"
          />
          {showHeatmap && (
            // key makes the heatmap remount (and fade in again) for every new tile
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={tile.tile_id}
              src={tileHeatmapUrl(tile.tile_id)}
              alt="Reconstruction error heatmap"
              className="fade-in pointer-events-none absolute inset-0 h-full w-full"
            />
          )}
        </div>

        <div className="mb-5 flex items-center justify-between gap-3 text-xs text-ocean-300">
          <button
            onClick={() => setShowHeatmap((v) => !v)}
            className="rounded-full border border-ocean-600 px-3 py-1 hover:border-teal-400"
          >
            {showHeatmap ? "Hide heatmap" : "Show heatmap"}
          </button>
          <span className="flex items-center gap-2">
            low
            <span className="h-1.5 w-16 rounded-full bg-gradient-to-r from-teal-400 via-amber-400 to-red-400" />
            high error
          </span>
        </div>

        {/* Score breakdown as bars */}
        <div className="mb-5 flex flex-col gap-3">
          {scores.map((s) => (
            <div key={s.label}>
              <div className="mb-1 flex justify-between text-xs">
                <span className="text-ocean-300">{s.label}</span>
                <span className="font-mono text-teal-300">{s.value.toFixed(3)}</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-ocean-700">
                <div
                  className={`bar-grow h-full rounded-full ${scoreColor(s.value)}`}
                  style={{ width: `${Math.round(s.value * 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        <div className="flex gap-2">
          {FLAG_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              disabled={submitting}
              onClick={() => handleFlag(opt.value)}
              className={`flex flex-1 flex-col items-center gap-0.5 rounded-lg px-2 py-2 text-xs font-medium text-black transition disabled:opacity-50 ${opt.color} ${
                tile.flag_status === opt.value ? "ring-2 ring-white" : ""
              }`}
            >
              {opt.label}
              <span className="font-mono text-[10px] opacity-70">key {opt.key}</span>
            </button>
          ))}
        </div>

        <p className="mt-3 text-center text-xs text-ocean-300">
          {tile.flag_status ? (
            <>
              Current status: <span className="text-teal-300">{tile.flag_status}</span>
            </>
          ) : (
            "Not reviewed yet"
          )}
          <span className="mx-2">·</span>← → step · Esc close
        </p>
      </div>
    </div>
  );
}