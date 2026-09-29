import { useState } from "react";
import { Sparkles, ChevronDown, ChevronRight, ArrowUpRight, MapPin, AlertTriangle } from "lucide-react";
import GlassCard from "../Common/GlassCard";
import ConfidenceRing from "./ConfidenceRing";
import ProgressBar from "./ProgressBar";
import HEX from "../../utils/constants";

const TASK_LABELS = {
  search: "SEARCH RESULT",
  change: "CHANGE ANALYSIS",
  vqa: "VISUAL Q&A",
  caption: "SCENE CAPTION",
  grounding: "GROUNDING",
  fusion: "MULTI-SENSOR FUSION",
  sar_change: "SAR CHANGE DETECTION",
  sar_grounding: "SAR GROUNDING (OPTICAL-GUIDED)",
};

function formatConfidence(band) {
  const map = { high: 0.9, medium: 0.6, low: 0.3, unknown: 0.1 };
  return map[band] ?? 0.5;
}

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

function ModelOutputSection({ label, children, color = "cyan" }) {
  return (
    <div className="mt-3 p-3 bg-white/3 rounded-lg">
      <div className="text-[10px] tracking-wide text-text-tertiary mb-1">{label}</div>
      <div className="text-[12px] text-text-primary">{children}</div>
    </div>
  );
}

export default function AIResponseCard({ result, onFocus }) {
  const [expanded, setExpanded] = useState(true);
  
  const taskType = result.task_type || result.trace?.task_selected || 'search';
  const confidenceBand = result.confidence_band || 'unknown';
  const confidenceReason = result.confidence_reason || '';
  const explanation = result.explanation_text || result.explanation || 'No explanation available';
  const metadata = result.metadata || {};
  const modelOutputs = result.model_outputs || {};
  const trace = result.trace || {};
  const sarCrossCheck = modelOutputs.sar_cross_check || {};

  return (
    <GlassCard className="p-4 max-w-140">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="w-5.5 h-5.5 rounded-md bg-cyan/12 flex items-center justify-center">
            <Sparkles size={12} color={HEX.cyan} />
          </span>
          <span className="text-[10.5px] tracking-wide text-cyan font-semibold">{TASK_LABELS[taskType] || taskType.toUpperCase()}</span>
        </div>
        <button onClick={() => onFocus && onFocus(result)} className="flex items-center gap-1 bg-transparent border-none text-text-tertiary text-[11px] cursor-pointer hover:text-cyan">
          View in panel <ArrowUpRight size={12} />
        </button>
      </div>

      <div className="flex items-start gap-3.5 mb-3.5">
        <ConfidenceRing value={formatConfidence(confidenceBand)} />
        <div className="flex-1 min-w-0">
          <ConfidenceBadge band={confidenceBand} reason={confidenceReason} />
          <p className="text-[13.2px] text-text-primary leading-relaxed m-0 mt-2">{explanation}</p>
        </div>
      </div>

      {/* SAR Cross-Check Disagreement Alert */}
      {sarCrossCheck.disagreement && (
        <div className="mb-3 p-3 bg-rose/10 border border-rose/30 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            <AlertTriangle size={14} className="text-rose" />
            <span className="text-[11px] tracking-wide text-rose font-semibold">
              SAR Cross-Check Disagreement ({sarCrossCheck.severity?.toUpperCase() || 'MINOR'})
            </span>
          </div>
          <div className="text-[12px] text-text-primary">{sarCrossCheck.details}</div>
        </div>
      )}

      <button
        onClick={() => setExpanded((v) => !v)}
        className={`flex items-center gap-1 bg-transparent border-none text-text-secondary text-[11.5px] cursor-pointer py-1 ${expanded ? "mb-2" : "mb-0"}`}
      >
        {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />} Analysis breakdown
      </button>

      {expanded && (
        <div className="pt-1 space-y-2">
          <ProgressBar label="Semantic match" value={result.score_breakdown?.semantic_sim ?? 0.5} accent="cyan" />
          <ProgressBar label="Geographic match" value={result.score_breakdown?.geo_score ?? 0.5} accent="teal" />
          <ProgressBar label="Temporal alignment" value={result.score_breakdown?.temporal_score ?? 0.5} accent="blue" />
          <ProgressBar label="Modality match" value={result.score_breakdown?.modality_score ?? 0.5} accent="purple" />
          <ProgressBar label="Change alignment" value={result.score_breakdown?.change_score ?? 0.5} accent="rose" />
          
          <div className="flex items-center gap-1.5 mt-2.5 py-2 px-2.5 bg-white/3 rounded-lg">
            <MapPin size={12} className="text-text-tertiary" />
            <span className="text-[11px] text-text-tertiary font-mono">
              {metadata.lat ? `${metadata.lat.toFixed(4)}, ${metadata.lon.toFixed(4)}` : 'N/A'} &nbsp;·&nbsp; 
              {metadata.sensor || 'N/A'} &nbsp;·&nbsp; 
              {metadata.acquisition_time || metadata.date || 'N/A'}
            </span>
          </div>
          
          {/* Model outputs */}
          {modelOutputs.caption && (
            <ModelOutputSection label="CAPTION" color="cyan">{modelOutputs.caption}</ModelOutputSection>
          )}
          {modelOutputs.vqa_answer && (
            <ModelOutputSection label="VQA ANSWER" color="teal">{modelOutputs.vqa_answer}</ModelOutputSection>
          )}
          {modelOutputs.change_description && (
            <ModelOutputSection label="CHANGE DESCRIPTION" color="blue">{modelOutputs.change_description}</ModelOutputSection>
          )}
          {modelOutputs.fusion_statement && (
            <ModelOutputSection label="FUSION SYNTHESIS" color="purple">{modelOutputs.fusion_statement}</ModelOutputSection>
          )}
          {modelOutputs.sar_caption && (
            <ModelOutputSection label="SAR CAPTION (CROMA-S1)" color="amber">{modelOutputs.sar_caption}</ModelOutputSection>
          )}
          {modelOutputs.sar_vqa_answer && (
            <ModelOutputSection label="SAR VQA (CROMA-S1)" color="amber">{modelOutputs.sar_vqa_answer}</ModelOutputSection>
          )}
          {modelOutputs.sar_features && (
            <div className="mt-3 p-3 bg-white/3 rounded-lg">
              <div className="text-[10px] tracking-wide text-text-tertiary mb-2">SAR FEATURES (DETERMINISTIC CROSS-CHECK)</div>
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
          
          {/* Three-field separation display */}
          {(modelOutputs.optical_caption || modelOutputs.sar_caption || modelOutputs.sar_features) && (
            <div className="mt-3 p-3 bg-white/3 rounded-lg">
              <div className="text-[10px] tracking-wide text-text-tertiary mb-2">THREE-FIELD SEPARATION</div>
              <div className="grid gap-2 text-[11px]">
                {modelOutputs.factual_metadata && (
                  <div className="bg-cyan/10 border border-cyan/30 p-2 rounded">
                    <div className="text-cyan text-[9px] font-semibold">FACTUAL METADATA</div>
                    <div className="text-text-primary text-[10px]">{modelOutputs.factual_metadata}</div>
                  </div>
                )}
                <div className="bg-teal/10 border border-teal/30 p-2 rounded">
                  <div className="text-teal text-[9px] font-semibold">DETERMINISTIC SENSOR-DERIVED</div>
                  <div className="text-text-primary text-[10px]">
                    {modelOutputs.sar_features ? 
                      `Water: ${(modelOutputs.sar_features.water_fraction * 100).toFixed(1)}% · Built-up: ${(modelOutputs.sar_features.builtup_fraction * 100).toFixed(1)}% · Log-ratio: ${modelOutputs.sar_features.log_ratio_mean?.toFixed(3) ?? 'N/A'}` 
                      : 'NDVI/NDWI delta, SAR log-ratio'}
                  </div>
                </div>
                <div className="bg-purple/10 border border-purple/30 p-2 rounded">
                  <div className="text-purple text-[9px] font-semibold">AI-MODEL-DERIVED</div>
                  <div className="text-text-primary text-[10px]">
                    {modelOutputs.optical_caption ? `Optical: ${modelOutputs.optical_caption.substring(0, 60)}...` : ''}
                    {modelOutputs.sar_caption ? ` SAR: ${modelOutputs.sar_caption.substring(0, 60)}...` : ''}
                    {modelOutputs.vqa_answer ? ` VQA: ${modelOutputs.vqa_answer.substring(0, 60)}...` : ''}
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Trace info */}
          <div className="mt-3 p-2 bg-white/3 rounded-lg">
            <div className="text-[10px] tracking-wide text-text-tertiary mb-1">EXECUTION TRACE</div>
            <div className="text-[10px] text-text-secondary font-mono">
              Tools: {trace.tools_called?.map(t => t.model_name).join(' → ') || 'N/A'}
            </div>
            <div className="text-[10px] text-text-secondary font-mono">
              Fallback: {trace.fallback_used ? 'Yes' : 'No'}
            </div>
            <div className="text-[10px] text-text-secondary font-mono">
              Candidates: {trace.candidates_after_filter || 0} / {trace.candidates_considered || 0}
            </div>
            {trace.tools_called?.some(t => t.model_name?.includes('CROMA') || t.model_name?.includes('EarthDial')) && (
              <div className="text-[10px] text-cyan font-mono mt-1">
                SAR Path: {trace.tools_called?.find(t => t.model_name?.includes('CROMA') || t.model_name?.includes('fallback'))?.model_name || 'N/A'}
              </div>
            )}
          </div>
        </div>
      )}
    </GlassCard>
  );
}