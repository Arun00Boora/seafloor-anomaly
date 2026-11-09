// src/lib/api.ts
//
// Thin client for the FastAPI backend. Keeping all fetch calls here (rather
// than scattered across components) means if the API shape changes, we only
// update one file.

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Tile {
  tile_id: string;
  rank: number;
  ensemble_score: number;
  ae_score: number;
  baseline_score: number;
  flag_status: string | null;
}

export interface FlagPayload {
  status: "confirmed_debris" | "false_positive" | "unsure";
  note?: string;
}

export async function fetchTiles(limit = 50, offset = 0): Promise<Tile[]> {
  const res = await fetch(`${API_BASE}/tiles?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(`Failed to fetch tiles: ${res.status}`);
  return res.json();
}

export function tileImageUrl(tileId: string): string {
  return `${API_BASE}/tiles/${tileId}/image`;
}
export function tileHeatmapUrl(tileId: string): string {
  return `${API_BASE}/tiles/${tileId}/heatmap`;
}
export async function submitFlag(tileId: string, payload: FlagPayload): Promise<void> {
  const res = await fetch(`${API_BASE}/tiles/${tileId}/flag`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Failed to submit flag: ${res.status}`);
}
export interface DepthResult {
  fraction: number;
  tiles_reviewed: number;
  debris_found: number;
  precision: number;
  recall: number;
}

export interface EffortResult {
  recall_target: number;
  tiles_reviewed: number;
  fraction_reviewed: number;
  effort_saved: number;
}

export interface MethodResult {
  key: string;
  label: string;
  at_depth: DepthResult[];
  effort: EffortResult[];
}

export interface Evaluation {
  n_tiles: number;
  n_debris: number;
  debris_rate: number;
  methods: MethodResult[];
}

export async function fetchEvaluation(): Promise<Evaluation> {
  const res = await fetch(`${API_BASE}/evaluation`);
  if (!res.ok) throw new Error(`Failed to fetch evaluation: ${res.status}`);
  return res.json();
}