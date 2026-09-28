import { Layers, MapPin, GitCompare, ShieldCheck, Info } from "lucide-react";
import SatelliteVisual from "./SatelliteVisual";
import ProgressBar from "../chat/ProgressBar";
import { btnGhost } from "../Common/buttonStyles";
import { SCENES } from "../Data/mockData";
import HEX from "../../utils/constants";

export default function SatellitePanel({ activeResult, onOpenMap, onCompare, onAnalysis }) {
  const scene = activeResult?.scene ?? SCENES[0];
  const scores = activeResult?.scores ?? { semantic: 0.91, geo: 0.94, temporal: 0.88, modality: 0.96, change: 0.62 };

  return (
    <div className="h-full flex flex-col overflow-y-auto p-4">
      <div className="flex items-center gap-2 mb-3.5">
        <Layers size={14} color={HEX.cyan} strokeWidth={1.8} />
        <span className="text-[11px] tracking-wide text-text-secondary font-semibold">SATELLITE INTELLIGENCE</span>
      </div>

      <SatelliteVisual scene={scene} />

      <div className="mt-3.5 mb-4">
        <div className="text-sm font-semibold text-text-primary mb-1">{scene.name}</div>
        <div className="flex flex-wrap gap-1.5 text-[11px] text-text-secondary">
          <span className="bg-surface border border-border rounded-md py-0.5 px-2">{scene.sensor}</span>
          <span className="bg-surface border border-border rounded-md py-0.5 px-2">{scene.date}</span>
          <span className="bg-surface border border-border rounded-md py-0.5 px-2">{scene.modality}</span>
        </div>
      </div>

      <div className="mb-1.5">
        <ProgressBar label="Semantic" value={scores.semantic} accent="cyan" />
        <ProgressBar label="Geographic" value={scores.geo} accent="teal" />
        <ProgressBar label="Temporal" value={scores.temporal} accent="blue" />
        <ProgressBar label="Sensor" value={scores.modality} accent="amber" />
        <ProgressBar label="Change" value={scores.change} accent="rose" />
      </div>

      <div className="flex flex-col gap-2 mt-2">
        <button onClick={onOpenMap} className={btnGhost()}>
          <MapPin size={13} /> View on map
        </button>
        <button onClick={onCompare} className={btnGhost()}>
          <GitCompare size={13} /> Compare imagery
        </button>
        <button
          onClick={onAnalysis}
          disabled={!activeResult}
          className={btnGhost(true) + (activeResult ? "" : " opacity-40 cursor-not-allowed")}
        >
          <ShieldCheck size={13} /> Open analysis
        </button>
      </div>

      <div className="mt-4 p-3 rounded-[10px] bg-white/2 border border-border">
        <div className="flex items-center gap-1.5 mb-1.5">
          <Info size={12} className="text-text-tertiary" />
          <span className="text-[10.5px] text-text-tertiary tracking-wide">PROVENANCE</span>
        </div>
        <p className="text-[11px] text-text-secondary leading-relaxed m-0">
          Scene {scene.id} sourced from {scene.origin.replace(/_/g, " ")}. Results are consistent with query criteria and catalog-verified geometry.
        </p>
      </div>
    </div>
  );
}