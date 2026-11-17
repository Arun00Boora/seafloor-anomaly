// src/app/evaluation/page.tsx
"use client";

import { useEffect, useState, type ReactNode } from "react";
import Link from "next/link";
import SonarMark from "@/components/SonarMark";
import { Evaluation, Prevalence, fetchEvaluation, fetchPrevalence } from "@/lib/api";

// One color per method. Ensemble is the headline method, so it gets the brand teal.
const METHOD_COLORS: Record<string, string> = {
  ensemble: "bg-teal-400",
  ae: "bg-amber-400",
  baseline: "bg-ocean-300",
  pc: "bg-rose-400",
};

const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

// Small "?" help bubble that shows a plain-English explanation on hover / focus.
function Hint({ text }: { text: string }) {
  return (
    <span className="group relative ml-1.5 inline-flex align-middle">
      <span
        tabIndex={0}
        aria-label={text}
        className="inline-flex h-4 w-4 cursor-help items-center justify-center rounded-full border border-teal-300 text-[10px] text-teal-300 outline-none"
      >
        ?
      </span>
      <span className="pointer-events-none invisible absolute left-0 top-6 z-30 w-64 rounded-lg border border-teal-400 bg-ocean-700 p-2.5 text-xs font-normal leading-snug text-white opacity-0 shadow-xl transition group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100">
        {text}
      </span>
    </span>
  );
}

// A collapsible section. Native <details>, so no extra state is needed.
function Section({
  title,
  hint,
  subtitle,
  defaultOpen = false,
  children,
}: {
  title: string;
  hint?: string;
  subtitle: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  return (
    <details
      open={defaultOpen}
      className="group reveal mb-3 rounded-xl border border-ocean-700 bg-ocean-800"
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-5 py-4 [&::-webkit-details-marker]:hidden">
        <span className="text-sm">
          <span className="mr-2 inline-block text-teal-300 transition group-open:rotate-90">▸</span>
          <span className="font-medium text-teal-300">{title}</span>
          {hint && <Hint text={hint} />}
          <span className="ml-3 text-ocean-300">{subtitle}</span>
        </span>
        <span className="shrink-0 text-xs text-ocean-300">
          <span className="group-open:hidden">Show</span>
          <span className="hidden group-open:inline">Hide</span>
        </span>
      </summary>
      <div className="px-5 pb-5">{children}</div>
    </details>
  );
}

function SummaryCard({
  label,
  big,
  text,
}: {
  label: string;
  big: string;
  text: string;
}) {
  return (
    <div className="reveal rounded-xl border border-ocean-700 bg-ocean-800 p-5">
      <p className="mb-2 text-xs uppercase tracking-wider text-teal-300">{label}</p>
      <p className="text-3xl font-bold leading-tight text-white">{big}</p>
      <p className="mt-2 text-sm text-ocean-300">{text}</p>
    </div>
  );
}

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

  // Summary numbers: use the depth closest to 10% and the recall target closest to 90%.
  const closest = <T,>(items: T[], get: (t: T) => number, target: number) =>
    items.reduce((best, x) => (Math.abs(get(x) - target) < Math.abs(get(best) - target) ? x : best));
  const topIdx = depths.indexOf(closest(depths, (d) => d.fraction, 0.1));
  const best = data.methods.reduce((a, b) =>
    b.at_depth[topIdx].precision > a.at_depth[topIdx].precision ? b : a
  );
  const bestDepth = best.at_depth[topIdx];
  const bestEffort = closest(best.effort, (e) => e.recall_target, 0.9);

  return (
    <main className="mx-auto max-w-5xl px-6 py-8">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <SonarMark size={44} />
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-white">Evaluation</h1>
            <p className="mt-0.5 text-sm text-ocean-300">
              Does the ranking beat a simple brightness check? Measured on {data.n_tiles} test tiles.
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
      <p className="mb-6 rounded-lg border border-amber-400 bg-amber-400/10 p-3 text-sm text-amber-400">
        Read this first: {pct(data.debris_rate, 0)} of these test tiles contain debris ({data.n_debris}{" "}
        of {data.n_tiles}), far more than a real survey would. A random check would already be right{" "}
        {pct(data.debris_rate, 0)} of the time, so compare the methods with each other, not with
        real-world performance. The last section shows what happens when debris is rare.
      </p>

      {/* Three summary cards: the answer at a glance */}
      <div className="mb-6 grid gap-4 sm:grid-cols-3">
        <SummaryCard
          label="Best method"
          big={best.label}
          text="Finds debris more reliably than the other methods."
        />
        <SummaryCard
          label={`Review the top ${pct(bestDepth.fraction, 0)}`}
          big={pct(bestDepth.precision)}
          text={`of those tiles really hold debris. A random check would be right ${pct(
            data.debris_rate,
            0
          )} of the time.`}
        />
        <SummaryCard
          label="Time saved"
          big={`~${pct(bestEffort.effort_saved, 0)}`}
          text={`of tiles can be skipped and you still find ${pct(
            bestEffort.recall_target,
            0
          )} of the debris.`}
        />
      </div>

      {/* Details, collapsed by default except the first */}
      <Section
        title="Precision"
        hint="Out of the tiles you look at, the share that really contain debris. Higher is better."
        subtitle="of the tiles you review, how many are debris"
        defaultOpen
      >
        <div className="mb-4 flex flex-wrap gap-5 text-sm text-ocean-300">
          {data.methods.map((m) => (
            <span key={m.key} className="inline-flex items-center gap-2">
              <span className={`h-3 w-3 rounded-sm ${METHOD_COLORS[m.key]}`} />
              {m.label}
            </span>
          ))}
        </div>
        <div className="flex flex-col gap-6">
          {depths.map((d, i) => (
            <div key={d.fraction}>
              <p className="mb-2 font-mono text-xs text-ocean-300">
                If you review the top {pct(d.fraction, 0)} ({d.tiles_reviewed} tiles)
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
      </Section>

      <Section
        title="Recall"
        hint="Out of all the debris that exists, the share you find in the tiles you reviewed. Higher is better."
        subtitle="how much of all the debris you find"
      >
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
      </Section>

      <Section
        title="Review effort saved"
        hint="How many tiles you can skip and still find a target share of the debris. Higher is better."
        subtitle="how many tiles you can skip"
      >
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
      </Section>

      <Section
        title="What if debris is rare?"
        subtitle="a more realistic test"
        hint="A real survey is mostly empty seabed. This rebuilds the test set with much less debris."
      >
        <p className="mb-4 text-sm text-ocean-300">
          A real survey is mostly empty seabed. Here the test set is rebuilt with far less debris and
          the methods are compared again.
        </p>
        <PrevalenceSection data={prevalence} error={prevalenceError} />
      </Section>
    </main>
  );
}


