// src/components/TileCard.tsx
//
// A single tile in the review grid: thumbnail, rank, flag badge, and a score
// bar whose fill/color shows how anomalous the model thinks this tile is.

import { Tile, tileImageUrl } from "@/lib/api";

interface TileCardProps {
  tile: Tile;
  onClick: () => void;
}

const FLAG_BADGES: Record<string, { icon: string; className: string }> = {
  confirmed_debris: { icon: "✓", className: "bg-red-400 text-black" },
  false_positive: { icon: "✕", className: "bg-ocean-600 text-white" },
  unsure: { icon: "?", className: "bg-amber-400 text-black" },
};

function scoreColor(score: number): string {
  // Higher score = more anomalous = warmer color, drawing the analyst's eye.
  if (score > 0.7) return "bg-red-400";
  if (score > 0.4) return "bg-amber-400";
  return "bg-teal-400";
}

export default function TileCard({ tile, onClick }: TileCardProps) {
  const badge = tile.flag_status ? FLAG_BADGES[tile.flag_status] : null;
  const high = tile.ensemble_score > 0.7;

  return (
    <button
      onClick={onClick}
      // Stagger the fade-in a little so the grid "ripples" in instead of popping.
      style={{ animationDelay: `${(tile.rank % 16) * 30}ms` }}
      className={`reveal group relative flex flex-col overflow-hidden rounded-lg border bg-ocean-800 transition duration-200 hover:-translate-y-0.5 hover:border-teal-400 hover:shadow-[0_8px_24px_-8px_rgba(45,212,191,0.35)] focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-400 ${
        high ? "border-red-400/40" : "border-ocean-700"
      }`}
    >
      <div className="relative aspect-square w-full overflow-hidden bg-black">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={tileImageUrl(tile.tile_id)}
          alt={`Sonar tile rank ${tile.rank}`}
          className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
        />
        <span className="absolute left-1.5 top-1.5 rounded bg-black/70 px-1.5 py-0.5 font-mono text-xs text-teal-300">
          #{tile.rank}
        </span>
        {badge && (
          <span
            title={tile.flag_status ?? ""}
            className={`absolute right-1.5 top-1.5 flex h-5 w-5 items-center justify-center rounded-full text-xs font-semibold ${badge.className}`}
          >
            {badge.icon}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 px-2 py-1.5">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ocean-700">
          <div
            className={`bar-grow h-full rounded-full ${scoreColor(tile.ensemble_score)}`}
            style={{ width: `${Math.round(tile.ensemble_score * 100)}%` }}
          />
        </div>
        <span className="font-mono text-xs text-ocean-300">
          {tile.ensemble_score.toFixed(2)}
        </span>
      </div>
    </button>
  );
}