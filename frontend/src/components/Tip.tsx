// src/components/Tip.tsx
//
// Small hover / tap tooltip that explains a technical word in plain language.
// <Term term="ensemble">Ensemble</Term> shows a dotted underline and a "?" badge;
// <Tip text="..." hint={false}> wraps a control (like a button) with a hover-only tooltip.

import type { ReactNode } from "react";

export const DEFINITIONS = {
  tile: "A small square piece of the sonar image. The model scores every tile on its own.",
  ensemble:
    "The main score. It combines the autoencoder score and the brightness score into one number. Higher means the tile looks more unusual.",
  autoencoder:
    "A small AI that learned to redraw normal seabed. If it redraws a tile badly, that tile probably holds something unusual.",
  brightness:
    "How bright the brightest spot in the tile is. Rocks, metal and wrecks often send back strong, bright echoes.",
  heatmap:
    "A colour layer on top of the tile. Warm colours mark the spots the model finds most unusual.",
  error:
    "How badly the model could redraw that spot. High error means the spot does not look like normal seabed.",
  debris: "Choose this when you are sure the tile shows an object or debris.",
  falsePositive: "Choose this when the model flagged the tile but it is just ordinary seabed.",
  unsure: "Choose this when you cannot tell. You can come back to it later.",
} as const;

export type TermKey = keyof typeof DEFINITIONS;

interface TipProps {
  text: string;
  children: ReactNode;
  align?: "left" | "right"; // which edge of the word the tooltip lines up with
  up?: boolean; // show the tooltip above instead of below
  hint?: boolean; // dotted underline + "?" badge (true) or a bare hover tooltip (false)
  className?: string; // wrapper classes, e.g. "flex flex-1" around a button
}

export function Tip({ text, children, align = "left", up = false, hint = true, className }: TipProps) {
  const show = hint
    ? "group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100"
    : "group-hover:visible group-hover:opacity-100";
  return (
    <span className={`group relative ${className ?? "inline-block"}`}>
      {hint ? (
        <>
          <span
            tabIndex={0}
            className="cursor-help border-b border-dotted border-teal-300 outline-none focus-visible:ring-1 focus-visible:ring-teal-300"
          >
            {children}
          </span>
          <span
            aria-hidden="true"
            className="ml-1 inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border border-teal-300 align-middle text-[9px] leading-none text-teal-300"
          >
            ?
          </span>
        </>
      ) : (
        children
      )}
      <span
        role="tooltip"
        className={`pointer-events-none invisible absolute z-50 w-56 rounded-lg border border-teal-400 bg-ocean-700 p-2.5 text-left text-xs font-normal leading-snug text-white opacity-0 shadow-xl transition-opacity ${show} ${
          align === "right" ? "right-0" : "left-0"
        } ${up ? "bottom-full mb-1.5" : "top-full mt-1.5"}`}
      >
        {text}
      </span>
    </span>
  );
}

export function Term({
  term,
  children,
  align,
  up,
}: {
  term: TermKey;
  children: ReactNode;
  align?: "left" | "right";
  up?: boolean;
}) {
  return (
    <Tip text={DEFINITIONS[term]} align={align} up={up}>
      {children}
    </Tip>
  );
}