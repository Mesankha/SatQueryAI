import { GitCompare, Waves, Sprout, Building2 } from "lucide-react";
import OrbitalVisual from "./OrbitalVisual";
import HEX from "../../utils/constants";

const SUGGESTIONS = [
  { icon: Sprout, text: "Show vegetation changes around Bengaluru" },
  { icon: GitCompare, text: "Compare satellite imagery from 2024 and 2026" },
  { icon: Waves, text: "Find flood-affected areas in Assam" },
  { icon: Building2, text: "Analyze land-use changes near this location" },
];

export default function EmptyState({ onSuggestion }) {
  return (
    <div className="max-w-160 mx-auto text-center px-5 pt-9 pb-2.5">
      <OrbitalVisual />
      <h1 className="font-display font-semibold text-[clamp(22px,3.2vw,30px)] text-text-primary mb-2.5 tracking-tight">
        Ask anything about Earth.
      </h1>
      <p className="text-sm text-text-secondary leading-relaxed max-w-115 mx-auto mb-7">
        Search satellite imagery, detect changes, analyze terrain and explore geospatial intelligence.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-w-130 mx-auto">
        {SUGGESTIONS.map((s, i) => (
          <button
            key={i}
            onClick={() => onSuggestion(s.text)}
            className="flex items-center gap-2.5 py-3.5 px-3.5 bg-surface border border-border rounded-xl text-text-primary text-[12.8px] text-left cursor-pointer transition-all duration-180 hover:border-cyan/35 hover:bg-surface-hover hover:-translate-y-0.5"
          >
            <span className="w-7 h-7 rounded-lg bg-cyan/8 flex items-center justify-center shrink-0">
              <s.icon size={14} color={HEX.cyan} strokeWidth={1.8} />
            </span>
            {s.text}
          </button>
        ))}
      </div>
    </div>
  );
}