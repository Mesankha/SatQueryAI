import { X } from "lucide-react";
import { hashSeed } from "../../utils/classifyQuery";
import seededFloat from "../../utils/SeededFloat";

export default function AnalysisModal({ result, onClose }) {
  if (!result) return null;
  const seed = hashSeed(result.scene.id + result.taskType);
  const steps = [
    { tool: "M1 · Query Parser", model: result.scores.semantic > 0.9 ? "phi-4 (primary)" : "keyword fallback", latency: Math.round(seededFloat(seed, 80, 420)) },
    { tool: "M2 · Retrieval", model: "RemoteCLIP + SQLite filter", latency: Math.round(seededFloat(seed + 1, 120, 650)) },
    result.taskType === "change"
      ? { tool: "M4 · Change Detection", model: "NDVI / NDWI delta", latency: Math.round(seededFloat(seed + 2, 300, 1400)) }
      : { tool: "M3 · Vision-Language Tool", model: "GeoChat", latency: Math.round(seededFloat(seed + 2, 400, 2200)) },
  ];
  const fallbackUsed = result.scores.composite < 0.82;

  return (
    <div onClick={onClose} className="fixed inset-0 bg-black/70 backdrop-blur-sm z-60 flex items-center justify-center p-5 animate-fade-in">
      <div onClick={(e) => e.stopPropagation()} className="w-full max-w-160 bg-bg-elevated border border-border rounded-2xl p-4.5">
        <div className="flex justify-between items-center mb-3">
          <span className="text-[13px] font-semibold text-text-primary">Execution trace</span>
          <button onClick={onClose} className="bg-transparent border-none text-text-tertiary cursor-pointer"><X size={16} /></button>
        </div>
        <div className="flex flex-col gap-2 mb-3">
          {steps.map((s, i) => (
            <div key={i} className="flex items-center justify-between py-2 px-3 bg-white/2 border border-border rounded-lg">
              <div>
                <div className="text-[12.5px] text-text-primary font-medium">{s.tool}</div>
                <div className="text-[10.5px] text-text-tertiary font-mono">{s.model}</div>
              </div>
              <div className="text-[11px] text-text-secondary font-mono">{s.latency}ms</div>
            </div>
          ))}
        </div>
        {fallbackUsed && (
          <div className="text-[11px] text-amber bg-amber/10 border border-amber/25 rounded-lg py-2 px-3">
            One or more tools used fallback behavior for this query — interpret confidence scores with caution.
          </div>
        )}
      </div>
    </div>
  );
}