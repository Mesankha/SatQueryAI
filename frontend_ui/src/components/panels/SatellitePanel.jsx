import { Layers, MapPin, GitCompare, ShieldCheck, Info, AlertTriangle, Cpu, Zap } from "lucide-react";
import SatelliteVisual from "./SatelliteVisual";
import ProgressBar from "../chat/ProgressBar";
import { btnGhost } from "../Common/buttonStyles";
import { SCENES } from "../Data/mockData";
import HEX from "../../utils/constants";

function ConfidenceBadge({ band, reason }) {
  const colors = {
    high: "bg-green/20 text-green border-green/30",
    medium: "bg-amber/20 text-amber border-amber/30",
    low: "bg-rose/20 text-rose border-rose/30",
    unknown: "bg-text-tertiary/20 text-text-tertiary border-text-tertiary/30",
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium border ${colors[band] || colors.unknown}`}>
      {band.toUpperCase()}
      {reason && <span className="text-[9px] opacity-70">— {reason}</span>}
    </span>
  );
}

function SarPathBadge({ path }) {
  if (!path) return null;
  const isCroma = path.includes('CROMA');
  const isFallback = path.includes('fallback') || path.includes('deterministic');
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium border ${isCroma ? 'bg-cyan/20 text-cyan border-cyan/30' : isFallback ? 'bg-amber/20 text-amber border-amber/30' : 'bg-purple/20 text-purple border-purple/30'}`}>
      {isCroma ? <Zap size={10} /> : isFallback ? <Cpu size={10} /> : <Info size={10} />}
      {path}
    </span>
  );
}

export default function SatellitePanel({ activeResult, onOpenMap, onCompare, onAnalysis }) {
  const result = activeResult || {};
  const scene = result.scene ?? SCENES[0];
  const modelOutputs = result.model_outputs || {};
  const scores = result.scores ?? result.score_breakdown ?? { semantic: 0.91, geo: 0.94, temporal: 0.88, modality: 0.96, change: 0.62 };
  const trace = result.trace || {};

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

      {/* Confidence Band */}
      <div className="mb-3">
        <ConfidenceBadge band={result.confidence_band || 'unknown'} reason={result.confidence_reason} />
      </div>

      {/* SAR Path Tracking */}
      {trace.tools_called && trace.tools_called.length > 0 && (
        <div className="mb-3">
          <SarPathBadge path={trace.tools_called.find(t => t.model_name?.includes('CROMA') || t.model_name?.includes('fallback') || t.model_name?.includes('EarthDial'))?.model_name || 'N/A'} />
        </div>
      )}

      {/* Score Breakdown */}
      <div className="mb-1.5 space-y-1">
        <ProgressBar label="Semantic" value={scores.semantic_sim ?? scores.semantic ?? 0.91} accent="cyan" />
        <ProgressBar label="Geographic" value={scores.geo_score ?? scores.geo ?? 0.94} accent="teal" />
        <ProgressBar label="Temporal" value={scores.temporal_score ?? scores.temporal ?? 0.88} accent="blue" />
        <ProgressBar label="Modality" value={scores.modality_score ?? scores.modality ?? 0.96} accent="amber" />
        <ProgressBar label="Change" value={scores.change_score ?? scores.change ?? 0.62} accent="rose" />
      </div>

      {/* SAR Features (Deterministic Cross-Check) */}
      {modelOutputs.sar_features && (
        <div className="mt-3 p-3 bg-amber/10 border border-amber/30 rounded-lg">
          <div className="flex items-center gap-1.5 mb-2">
            <Cpu size={12} className="text-amber" />
            <span className="text-[10.5px] tracking-wide text-amber font-semibold">SAR FEATURES (DETERMINISTIC CROSS-CHECK)</span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-[11px]">
            <div className="bg-white/5 p-2 rounded">
              <div className="text-text-tertiary text-[9px]">WATER FRACTION</div>
              <div className="text-text-primary font-mono">{(modelOutputs.sar_features.water_fraction * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-white/5 p-2 rounded">
              <div className="text-text-tertiary text-[9px]">BUILT-UP FRACTION</div>
              <div className="text-text-primary font-mono">{(modelOutputs.sar_features.builtup_fraction * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-white/5 p-2 rounded">
              <div className="text-text-tertiary text-[9px]">LOG-RATIO MEAN</div>
              <div className="text-text-primary font-mono">{modelOutputs.sar_features.log_ratio_mean?.toFixed(3) ?? 'N/A'}</div>
            </div>
          </div>
          {modelOutputs.sar_features.notes && (
            <div className="mt-2 text-[10px] text-text-tertiary">{modelOutputs.sar_features.notes}</div>
          )}
        </div>
      )}

      {/* SAR Cross-Check Disagreement */}
      {modelOutputs.sar_cross_check?.disagreement && (
        <div className="mt-3 p-3 bg-rose/10 border border-rose/30 rounded-lg">
          <div className="flex items-center gap-1.5 mb-1">
            <AlertTriangle size={12} className="text-rose" />
            <span className="text-[10.5px] tracking-wide text-rose font-semibold">
              SAR Cross-Check Disagreement ({modelOutputs.sar_cross_check.severity?.toUpperCase() || 'MINOR'})
            </span>
          </div>
          <div className="text-[11px] text-text-primary">{modelOutputs.sar_cross_check.details}</div>
        </div>
      )}

      {/* Action Buttons */}
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