import { X, Crosshair } from "lucide-react";
import LeafletMap from "../map/LeafletMap";
import { SCENES } from "../Data/mockData";
import HEX from "../../utils/constants";

export default function MapModal({ scene, onClose, compact }) {
  if (!scene) return null;
  return (
    <div onClick={onClose} className="fixed inset-0 bg-black/70 backdrop-blur-sm z-60 flex items-center justify-center p-5 animate-fade-in">
      <div onClick={(e) => e.stopPropagation()} className="w-full max-w-160 bg-bg-elevated border border-border rounded-2xl p-4.5">
        <div className="flex justify-between items-center mb-3">
          <div>
            <span className="text-[13px] font-semibold text-text-primary">{scene.name}</span>
            <div className="text-[10.5px] text-text-tertiary mt-0.5">{scene.sensor} &middot; {scene.date}</div>
          </div>
          <button onClick={onClose} className="bg-transparent border-none text-text-tertiary cursor-pointer"><X size={16} /></button>
        </div>
        <LeafletMap scenes={SCENES} focusId={scene.id} height={compact ? 260 : 340} zoom={7} />
        <div className="flex gap-2 mt-3 text-[11px] text-text-secondary font-mono">
          <Crosshair size={13} color={HEX.cyan} /> lat {scene.lat.toFixed(4)}, lon {scene.lon.toFixed(4)}
        </div>
      </div>
    </div>
  );
}