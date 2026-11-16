// src/components/TileThumb.tsx
//
// A small picture of one tile, cut out of the uploaded image in the browser.
// x, y and size are in the same pixel units as imgW and imgH (the analyzed image).

"use client";

import { useEffect, useRef } from "react";

interface TileThumbProps {
  src: string;
  x: number;
  y: number;
  size: number;
  imgW: number;
  imgH: number;
  borderClass: string;
  label: string;
}

export default function TileThumb({ src, x, y, size, imgW, imgH, borderClass, label }: TileThumbProps) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const k = img.naturalWidth / imgW; // in case the shown image is scaled
      const sx = x * k;
      const sy = y * k;
      const sw = Math.min(size, imgW - x) * k;
      const sh = Math.min(size, imgH - y) * k;
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const scale = Math.min(canvas.width / sw, canvas.height / sh);
      const dw = sw * scale;
      const dh = sh * scale;
      ctx.drawImage(img, sx, sy, sw, sh, (canvas.width - dw) / 2, (canvas.height - dh) / 2, dw, dh);
    };
    img.src = src;
    return () => {
      cancelled = true;
    };
  }, [src, x, y, size, imgW, imgH]);

  return (
    <canvas
      ref={ref}
      width={160}
      height={160}
      role="img"
      aria-label={label}
      className={`h-20 w-20 shrink-0 rounded-lg border-2 bg-black ${borderClass}`}
    />
  );
}