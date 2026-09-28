import { useState } from "react";
import { Sparkles, ChevronDown, ChevronRight, ArrowUpRight, MapPin } from "lucide-react";
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
};

function formatConfidence(band) {
  const map = { high: 0.9, medium: 0.6, low: 0.3, unknown: 0.1 };
  return map[band] ?? 0.5;
}

export default function AIResponseCard({ result, onFocus }) {
  const [expanded, setExpanded] = useState(true);
  
  const taskType = result.task_type || result.trace?.task_selected || 'search';
  const confidenceBand = result.confidence_band || 'unknown';
  const explanation = result.explanation_text || result.explanation || 'No explanation available';
  const metadata = result.metadata || {};
  const modelOutputs = result.model_outputs || {};
  const trace = result.trace || {};

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

      <div className="flex gap-3.5 mb-3.5">
        <ConfidenceRing value={formatConfidence(confidenceBand)} />
        <p className="text-[13.2px] text-text-primary leading-relaxed m-0">{explanation}</p>
      </div>

      <button
        onClick={() => setExpanded((v) => !v)}
        className={`flex items-center gap-1 bg-transparent border-none text-text-secondary text-[11.5px] cursor-pointer py-1 ${expanded ? "mb-2" : "mb-0"}`}
      >
        {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />} Analysis breakdown
      </button>

      {expanded && (
        <div className="pt-1">
          <ProgressBar label="Semantic match" value={result.score_breakdown?.semantic_sim ?? 0.5} accent="cyan" />
          <ProgressBar label="Geographic match" value={result.score_breakdown?.geo_score ?? 0.5} accent="teal" />
          <ProgressBar label="Temporal alignment" value={result.score_breakdown?.temporal_score ?? 0.5} accent="blue" />
          <ProgressBar label="Modality match" value={result.score_breakdown?.modality_score ?? 0.5} accent="purple" />
          
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
            <div className="mt-3 p-3 bg-white/3 rounded-lg">
              <div className="text-[10px] tracking-wide text-text-tertiary mb-1">CAPTION</div>
              <div className="text-[12px] text-text-primary">{modelOutputs.caption}</div>
            </div>
          )}
          {modelOutputs.vqa_answer && (
            <div className="mt-3 p-3 bg-white/3 rounded-lg">
              <div className="text-[10px] tracking-wide text-text-tertiary mb-1">VQA ANSWER</div>
              <div className="text-[12px] text-text-primary">{modelOutputs.vqa_answer}</div>
            </div>
          )}
          {modelOutputs.change_description && (
            <div className="mt-3 p-3 bg-white/3 rounded-lg">
              <div className="text-[10px] tracking-wide text-text-tertiary mb-1">CHANGE DESCRIPTION</div>
              <div className="text-[12px] text-text-primary">{modelOutputs.change_description}</div>
            </div>
          )}
          {modelOutputs.fusion_statement && (
            <div className="mt-3 p-3 bg-white/3 rounded-lg">
              <div className="text-[10px] tracking-wide text-text-tertiary mb-1">FUSION SYNTHESIS</div>
              <div className="text-[12px] text-text-primary">{modelOutputs.fusion_statement}</div>
            </div>
          )}
          {modelOutputs.sar_caption && (
            <div className="mt-3 p-3 bg-white/3 rounded-lg">
              <div className="text-[10px] tracking-wide text-text-tertiary mb-1">SAR CAPTION</div>
              <div className="text-[12px] text-text-primary">{modelOutputs.sar_caption}</div>
            </div>
          )}
          {modelOutputs.sar_features && (
            <div className="mt-3 p-3 bg-white/3 rounded-lg">
              <div className="text-[10px] tracking-wide text-text-tertiary mb-1">SAR FEATURES</div>
              <div className="grid grid-cols-3 gap-2 text-[11px]">
                <div>Water: {(modelOutputs.sar_features.water_fraction * 100).toFixed(1)}%</div>
                <div>Built-up: {(modelOutputs.sar_features.builtup_fraction * 100).toFixed(1)}%</div>
                <div>Log-ratio: {modelOutputs.sar_features.log_ratio_mean?.toFixed(3) ?? 'N/A'}</div>
              </div>
            </div>
          )}
          
          {/* Trace info */}
          <div className="mt-3 p-2 bg-white/3 rounded-lg">
            <div className="text-[10px] tracking-wide text-text-tertiary mb-1">EXECUTION TRACE</div>
            <div className="text-[10px] text-text-secondary font-mono">
              Tools: {trace.tools_called?.map(t => t.tool_name).join(', ') || 'N/A'}
            </div>
            <div className="text-[10px] text-text-secondary font-mono">
              Fallback: {trace.fallback_used ? 'Yes' : 'No'}
            </div>
          </div>
</div>
      )}
    </GlassCard>
  );
}