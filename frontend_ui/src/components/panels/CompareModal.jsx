import { X } from "lucide-react";
import SatelliteVisual from "./SatelliteVisual";
import { SCENES } from "../Data/mockData";

export default function CompareModal({ scene, onClose }) {
  if (!scene) return null;
  const pair = SCENES.find((s) => s.id === scene.pairOf) || SCENES.find((s) => s.pairOf === scene.id);
  return (
    <div onClick={onClose} className="fixed inset-0 bg-black/70 backdrop-blur-sm z-60 flex items-center justify-center p-5 animate-fade-in">
      <div onClick={(e) => e.stopPropagation()} className="w-full max-w-160 bg-bg-elevated border border-border rounded-2xl p-4.5">
        <div className="flex justify-between items-center mb-3">
          <span className="text-[13px] font-semibold text-text-primary">Compare imagery — {scene.name}</span>
          <button onClick={onClose} className="bg-transparent border-none text-text-tertiary cursor-pointer"><X size={16} /></button>
        </div>
        {pair ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <SatelliteVisual scene={scene} />
              <div className="text-[11px] text-text-secondary mt-1.5 font-mono">{scene.date} &middot; {scene.sensor}</div>
            </div>
            <div>
              <SatelliteVisual scene={pair} />
              <div className="text-[11px] text-text-secondary mt-1.5 font-mono">{pair.date} &middot; {pair.sensor}</div>
            </div>
          </div>
        ) : (
          <div className="py-9 text-center text-text-tertiary text-[12.5px]">
            No paired observation available for this scene in the current catalog window.
          </div>
        )}
      </div>
    </div>
  );
}