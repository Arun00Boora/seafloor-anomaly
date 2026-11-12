// src/components/SonarMark.tsx
//
// The project's logo mark: concentric sonar rings with a slowly rotating sweep.

export default function SonarMark({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <circle cx="16" cy="16" r="14" stroke="#2dd4bf" strokeOpacity="0.35" strokeWidth="1.5" />
      <circle cx="16" cy="16" r="9" stroke="#2dd4bf" strokeOpacity="0.55" strokeWidth="1.5" />
      <circle cx="16" cy="16" r="4" stroke="#2dd4bf" strokeOpacity="0.9" strokeWidth="1.5" />
      <g className="sonar-sweep">
        <path d="M16 16 L16 2 A14 14 0 0 1 28.1 9 Z" fill="#2dd4bf" fillOpacity="0.22" />
        <line x1="16" y1="16" x2="16" y2="2" stroke="#5eead4" strokeWidth="1.5" strokeLinecap="round" />
      </g>
      <circle cx="16" cy="16" r="1.6" fill="#5eead4" />
    </svg>
  );
}