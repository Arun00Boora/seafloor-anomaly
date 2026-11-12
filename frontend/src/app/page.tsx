// src/app/page.tsx

"use client"; // uses state, effects and browser events, so it must be a Client Component
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Tile, fetchTiles } from "@/lib/api";
import TileCard from "@/components/TileCard";
import TileDetailModal from "@/components/TileDetailModal";
import SonarMark from "@/components/SonarMark";
// Filter options. "value" is matched against tile.flag_status.
const FILTERS = [
  { key: "all", label: "All" },
  { key: "unreviewed", label: "Unreviewed" },
  { key: "confirmed_debris", label: "Confirmed" },
  { key: "false_positive", label: "False positive" },
  { key: "unsure", label: "Unsure" },
] as const;

type FilterKey = (typeof FILTERS)[number]["key"];

export default function Home() {
  const [tiles, setTiles] = useState<Tile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [filter, setFilter] = useState<FilterKey>("all");

  // Load the whole ranked index (700 tiles) once.
  useEffect(() => {
    fetchTiles(700, 0)
      .then(setTiles)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  // Tiles visible under the current filter, still in rank order.
  const visible = useMemo(() => {
    if (filter === "all") return tiles;
    if (filter === "unreviewed") return tiles.filter((t) => !t.flag_status);
    return tiles.filter((t) => t.flag_status === filter);
  }, [tiles, filter]);

  const reviewedCount = tiles.filter((t) => t.flag_status).length;

  // The selected tile is looked up from the master list, so flag changes show up instantly.
  const selectedTile = tiles.find((t) => t.tile_id === selectedId) ?? null;
  const selectedIndex = visible.findIndex((t) => t.tile_id === selectedId);

  function step(delta: number) {
    if (selectedIndex === -1) return;
    const next = visible[selectedIndex + delta];
    if (next) setSelectedId(next.tile_id);
  }

  // Arrow keys step through the visible list while the modal is open.
  useEffect(() => {
    if (!selectedId) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "ArrowRight") step(1);
      if (e.key === "ArrowLeft") step(-1);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  function handleFlagged(tileId: string, status: string) {
    setTiles((prev) =>
      prev.map((t) => (t.tile_id === tileId ? { ...t, flag_status: status } : t))
    );
  }

  return (
    <main className="mx-auto max-w-7xl px-6 py-8">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
               <div className="flex items-center gap-4">
          <SonarMark size={44} />
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-white">Seafloor Review</h1>
            <p className="mt-0.5 text-sm text-ocean-300">
              Side-scan sonar tiles, ranked most to least anomalous.
            </p>
          </div>
        </div>

        {/* Reviewed counter + progress bar */}
        <div className="min-w-56">
          <p className="text-sm text-ocean-300">
            {reviewedCount} of {tiles.length} reviewed
          </p>
          <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-ocean-700">
            <div
              className="h-full bg-teal-400"
              style={{ width: tiles.length ? `${(reviewedCount / tiles.length) * 100}%` : "0%" }}
            />
          </div>
        </div>
      </header>

      {/* Filter bar */}
      <div className="mb-6 flex flex-wrap gap-2">
        {FILTERS.map((f) => (
          <button
            key={f.key}
            onClick={() => setFilter(f.key)}
            className={`rounded-full border px-3 py-1 text-sm ${
              filter === f.key
                ? "border-teal-400 bg-ocean-700 text-teal-300"
                : "border-ocean-600 text-ocean-300 hover:border-teal-400"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {loading && <p className="text-ocean-300">Loading ranked tiles...</p>}

      {error && (
        <p className="rounded-lg border border-red-400 bg-red-400/10 p-3 text-red-400">
          Failed to load tiles: {error}. Is the backend running at localhost:8000?
        </p>
      )}

      {!loading && !error && visible.length === 0 && (
        <p className="text-ocean-300">No tiles match this filter.</p>
      )}

      {!loading && !error && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
          {visible.map((tile) => (
            <TileCard key={tile.tile_id} tile={tile} onClick={() => setSelectedId(tile.tile_id)} />
          ))}
        </div>
      )}

      {selectedTile && (
        <TileDetailModal
          tile={selectedTile}
          onClose={() => setSelectedId(null)}
          onFlagged={handleFlagged}
        />
      )}
    </main>
  );
}