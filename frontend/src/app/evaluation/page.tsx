// src/app/evaluation/page.tsx
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import SonarMark from "@/components/SonarMark";
import { Evaluation, Prevalence, fetchEvaluation, fetchPrevalence } from "@/lib/api";

// One color per method. Ensemble is the headline method, so it gets the brand teal.
const METHOD_COLORS: Record<string, string> = {
  ensemble: "bg-teal-400",
  ae: "bg-amber-400",
  baseline: "bg-ocean-300",
};

const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

function PrevalenceSection({ data, error }: { data: Prevalence | null; error: string | null }) {
  const [selected, setSelected] = useState(0);

  if (error) {
    return (
      <p className="text-sm text-red-400">
        Could not load the rare-debris results ({error}).
      </p>
    );
  }
  if (!data) return <p className="text-sm text-ocean-300">Loading rare-debris results...</p>;

  const rate = data.rates[selected];
  const depths = rate.methods[0].at_depth;

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-2">
        {data.rates.map((r, i) => (
          <button
            key={r.rate}
            onClick={() => setSelected(i)}
            className={`rounded-full border px-3 py-1 text-sm ${
              i === selected
                ? "border-teal-400 bg-ocean-700 text-teal-300"
                : "border-ocean-600 text-ocean-300 hover:border-teal-400"
            }`}
          >
            {pct(r.rate, 0)} debris
          </button>
        ))}
      </div>

      <p className="mb-5 text-sm text-ocean-300">
        {rate.n_total} tiles: all {data.n_normal} normal tiles plus {rate.n_debris} randomly chosen
        debris tiles. Reviewing in random order would be {pct(rate.rate, 0)} precise.
      </p>

      <div className="flex flex-col gap-6">
        {depths.map((d, i) => (
          <div key={d.fraction}>
            <p className="mb-2 font-mono text-xs text-ocean-300">Review top {pct(d.fraction, 0)}</p>
            <div className="flex flex-col gap-1.5">
              {rate.methods.map((m) => {
                const x = m.at_depth[i];
                return (
                  <div key={m.key} className="flex items-center gap-3">
                    <div className="h-3 flex-1 overflow-hidden rounded-sm bg-ocean-700">
                      <div
                        className={`bar-grow h-full ${METHOD_COLORS[m.key]}`}
                        style={{ width: pct(x.precision, 2) }}
                      />
                    </div>
                    <span className="w-36 text-right font-mono text-xs text-ocean-300">
                      {pct(x.precision)}{" "}
                      <span className="text-[10px]">
                        ({pct(x.precision_low, 0)} to {pct(x.precision_high, 0)})
                      </span>
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 overflow-x-auto">
        <p className="mb-2 text-sm text-ocean-300">
          Review effort saved at this debris rate (average over {data.trials} random draws):
        </p>
        <table className="w-full text-left text-sm">
          <thead className="text-ocean-300">
            <tr>
              <th className="py-2 pr-4 font-normal">Method</th>
              {rate.methods[0].effort.map((e) => (
                <th key={e.recall_target} className="py-2 pr-4 font-mono font-normal">
                  Find {pct(e.recall_target, 0)} of debris
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rate.methods.map((m) => (
              <tr key={m.key} className="border-t border-ocean-700">
                <td className="py-2 pr-4 text-white">{m.label}</td>
                {m.effort.map((e) => (
                  <td key={e.recall_target} className="py-2 pr-4 font-mono text-teal-300">
                    skip {pct(e.effort_saved)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-4 text-xs text-ocean-300">
        Each result is the average of {data.trials} random draws, and the range in brackets is the
        5th to 95th percentile. Only {data.n_normal} normal tiles exist in this test set, so at 2%
        debris just {data.rates[data.rates.length - 1].n_debris} debris tiles remain and results
        swing a lot. This shows the trend when debris is rare, not exact real-world performance.
      </p>
    </div>
  );
}

export default function EvaluationPage() {
  const [data, setData] = useState<Evaluation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [prevalence, setPrevalence] = useState<Prevalence | null>(null);
  const [prevalenceError, setPrevalenceError] = useState<string | null>(null);

  useEffect(() => {
    fetchEvaluation().then(setData).catch((err) => setError(err.message));
    fetchPrevalence().then(setPrevalence).catch((err) => setPrevalenceError(err.message));
  }, []);

  if (error) {
    return (
      <main className="mx-auto max-w-5xl px-6 py-8">
        <p className="rounded-lg border border-red-400 bg-red-400/10 p-3 text-red-400">
          Could not load the evaluation ({error}). The server may be waking up, so wait a few seconds
          and refresh the page.
        </p>
      </main>
    );
  }
  if (!data) {
    return (
      <main className="mx-auto max-w-5xl px-6 py-8">
        <p className="text-ocean-300">
          Loading evaluation. If the server was idle, this can take up to a minute.
        </p>
      </main>
    );
  }

  const depths = data.methods[0].at_depth;

  return (
    <main className="mx-auto max-w-5xl px-6 py-8">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <SonarMark size={44} />
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-white">Evaluation</h1>
            <p className="mt-0.5 text-sm text-ocean-300">
              Does the ranking beat a simple brightness threshold? Measured on {data.n_tiles} test tiles.
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

      {/* The honest caveat, shown up front */}
      <p className="mb-8 rounded-lg border border-amber-400 bg-amber-400/10 p-3 text-sm text-amber-400">
        Read this first: {pct(data.debris_rate, 0)} of these test tiles contain debris ({data.n_debris}{" "}
        of {data.n_tiles}), far more than a real survey would. Reviewing tiles at random would
        already be {pct(data.debris_rate, 0)} precise, so compare the methods to each other rather
        than treating the raw percentages as real-world performance. The last section below shows
        what happens when debris is rare.
      </p>

      {/* Legend */}
      <div className="mb-4 flex flex-wrap gap-5 text-sm text-ocean-300">
        {data.methods.map((m) => (
          <span key={m.key} className="inline-flex items-center gap-2">
            <span className={`h-3 w-3 rounded-sm ${METHOD_COLORS[m.key]}`} />
            {m.label}
          </span>
        ))}
      </div>

      {/* Precision chart: one group of bars per review depth */}
      <section className="reveal mb-10 rounded-xl border border-ocean-700 bg-ocean-800 p-5">
        <h2 className="mb-4 text-sm font-medium text-teal-300">
          Precision: of the tiles an analyst reviews, how many really hold debris
        </h2>
        <div className="flex flex-col gap-6">
          {depths.map((d, i) => (
            <div key={d.fraction}>
              <p className="mb-2 font-mono text-xs text-ocean-300">
                Review top {pct(d.fraction, 0)} ({d.tiles_reviewed} tiles)
              </p>
              <div className="flex flex-col gap-1.5">
                {data.methods.map((m) => (
                  <div key={m.key} className="flex items-center gap-3">
                    <div className="h-3 flex-1 overflow-hidden rounded-sm bg-ocean-700">
                      <div
                        className={`bar-grow h-full ${METHOD_COLORS[m.key]}`}
                        style={{ width: pct(m.at_depth[i].precision, 2) }}
                      />
                    </div>
                    <span className="w-14 text-right font-mono text-xs text-ocean-300">
                      {pct(m.at_depth[i].precision)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Recall table */}
      <section className="reveal mb-10 rounded-xl border border-ocean-700 bg-ocean-800 p-5">
        <h2 className="mb-4 text-sm font-medium text-teal-300">
          Recall: how much of all the debris the reviewed tiles cover
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-ocean-300">
              <tr>
                <th className="py-2 pr-4 font-normal">Method</th>
                {depths.map((d) => (
                  <th key={d.fraction} className="py-2 pr-4 font-mono font-normal">
                    Top {pct(d.fraction, 0)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.methods.map((m) => (
                <tr key={m.key} className="border-t border-ocean-700">
                  <td className="py-2 pr-4 text-white">{m.label}</td>
                  {m.at_depth.map((d) => (
                    <td key={d.fraction} className="py-2 pr-4 font-mono text-teal-300">
                      {pct(d.recall)}
                      <span className="ml-1 text-xs text-ocean-300">
                        ({d.debris_found}/{data.n_debris})
                      </span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Review effort saved */}
      <section className="reveal mb-10 rounded-xl border border-ocean-700 bg-ocean-800 p-5">
        <h2 className="mb-1 text-sm font-medium text-teal-300">Review effort saved</h2>
        <p className="mb-4 text-sm text-ocean-300">
          Share of tiles an analyst can skip while still finding a target share of the debris.
          Higher is better.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-ocean-300">
              <tr>
                <th className="py-2 pr-4 font-normal">Method</th>
                {data.methods[0].effort.map((e) => (
                  <th key={e.recall_target} className="py-2 pr-4 font-mono font-normal">
                    Find {pct(e.recall_target, 0)} of debris
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.methods.map((m) => (
                <tr key={m.key} className="border-t border-ocean-700">
                  <td className="py-2 pr-4 text-white">{m.label}</td>
                  {m.effort.map((e) => (
                    <td key={e.recall_target} className="py-2 pr-4 font-mono text-teal-300">
                      skip {pct(e.effort_saved)}
                      <span className="ml-1 text-xs text-ocean-300">
                        (review {e.tiles_reviewed})
                      </span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Realistic prevalence */}
      <section className="reveal rounded-xl border border-ocean-700 bg-ocean-800 p-5">
        <h2 className="mb-1 text-sm font-medium text-teal-300">What if debris is rare?</h2>
        <p className="mb-4 text-sm text-ocean-300">
          A real survey is mostly empty seabed. Here the test set is rebuilt with far less debris and
          the methods are compared again.
        </p>
        <PrevalenceSection data={prevalence} error={prevalenceError} />
      </section>
    </main>
  );
}