import { useState } from "react";
import LeafletMap from "../map/LeafletMap";
import GlassCard from "../Common/GlassCard";
import { SCENES } from "../Data/mockData";

export default function MapPage() {
  const [focusId, setFocusId] = useState(SCENES[0].id);
  const focusScene = SCENES.find((s) => s.id === focusId);

  return (
    <div className="p-6.5 w-full h-full overflow-y-auto">
      <div className="text-[11px] tracking-wide text-text-tertiary mb-1.5">WORKSPACE / GEOGRAPHIC VIEW</div>
      <h1 className="font-display text-2xl text-text-primary mb-1.5">Catalog map</h1>
      <p className="text-[13px] text-text-secondary mb-4.5">All indexed scenes plotted by acquisition geometry. Click a marker or a scene to focus it.</p>

      <div className="flex flex-col lg:flex-row gap-4.5 items-start flex-wrap">
        <div className="flex-[2_1_480px] min-w-[320px] w-full">
          <LeafletMap scenes={SCENES} focusId={focusId} onSelect={(s) => setFocusId(s.id)} height={460} zoom={5} />
          <div className="flex gap-4 mt-2.5 text-[11px] text-text-secondary">
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-cyan" /> Sentinel-2 (optical)</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-blue" /> Sentinel-1 (SAR)</span>
          </div>
        </div>

        <div className="flex-[1_1_260px] min-w-65 w-full">
          <GlassCard className="p-1.5 max-h-115 overflow-y-auto">
            {SCENES.map((s) => (
              <button
                key={s.id}
                onClick={() => setFocusId(s.id)}
                className={`w-full text-left flex flex-col gap-0.5 py-2.5 px-3 rounded-[10px] border-none cursor-pointer mb-0.5
                  ${focusId === s.id ? "bg-cyan/10 border-l-2 border-l-cyan" : "bg-transparent border-l-2 border-l-transparent"}`}
              >
                <span className={`text-[12.5px] font-medium ${focusId === s.id ? "text-cyan" : "text-text-primary"}`}>{s.name}</span>
                <span className="text-[10.5px] text-text-tertiary font-mono">{s.sensor} &middot; {s.date}</span>
              </button>
            ))}
          </GlassCard>

          {focusScene && (
            <GlassCard className="p-3.5 mt-3">
              <div className="text-[10.5px] tracking-wide text-text-tertiary mb-1.5">SELECTED SCENE</div>
              <div className="text-[13px] font-semibold text-text-primary mb-1">{focusScene.name}</div>
              <div className="text-[11px] text-text-secondary font-mono">
                {focusScene.lat.toFixed(4)}, {focusScene.lon.toFixed(4)}
              </div>
              <div className="text-[11px] text-text-secondary mt-1">{focusScene.resolution} &middot; {focusScene.origin.replace(/_/g, " ")}</div>
            </GlassCard>
          )}
        </div>
      </div>
    </div>
  );
}