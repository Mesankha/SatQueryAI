import { useState, useEffect } from "react";
import HEX from "../../utils/constants";
import { ACCENT_CLASSES } from "../Common/buttonStyles";

export default function ProgressBar({ label, value, accent = "cyan", mono }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => setWidth(value * 100));
    return () => cancelAnimationFrame(id);
  }, [value]);
  const a = ACCENT_CLASSES[accent] || ACCENT_CLASSES.cyan;
  return (
    <div className="mb-3">
      <div className="flex justify-between mb-1.5">
        <span className="text-[11.5px] text-text-secondary tracking-wide">{label}</span>
        <span className={`text-[11.5px] text-text-primary font-semibold ${mono ? "font-mono" : ""}`}>
          {Math.round(value * 100)}%
        </span>
      </div>
      <div className="h-1.25 rounded-full bg-white/6 overflow-hidden">
        <div
          className={`h-full rounded-full bg-linear-to-r ${a.from} ${a.to} transition-[width] duration-900 ease-[cubic-bezier(0.16,1,0.3,1)]`}
          style={{ width: `${width}%`, boxShadow: `0 0 8px ${HEX[accent]}55` }}
        />
      </div>
    </div>
  );
}