import { useState } from "react";
import GlassCard from "../Common/GlassCard";
import Toggle from "../Common/Toggle";

export default function SettingsPage() {
  const [prefs, setPrefs] = useState({ notifications: true, autoAnalysis: true, highPrecision: false, telemetry: true });
  const rows = [
    { key: "notifications", label: "Query notifications", desc: "Get notified when a long-running analysis completes." },
    { key: "autoAnalysis", label: "Auto-run analysis breakdown", desc: "Automatically expand confidence breakdowns on new results." },
    { key: "highPrecision", label: "High-precision retrieval", desc: "Prioritize geographic accuracy over response speed." },
    { key: "telemetry", label: "Share anonymous usage telemetry", desc: "Helps improve ranking quality across the catalog." },
  ];
  return (
    <div className="py-6.5 px-7 max-w-160 mx-auto w-full">
      <div className="text-[11px] tracking-wide text-text-tertiary mb-1.5">WORKSPACE / SETTINGS</div>
      <h1 className="font-display text-2xl text-text-primary mb-5">Preferences</h1>
      <GlassCard className="p-1.5">
        {rows.map((r, i) => (
          <div key={r.key} className={`flex items-center justify-between p-3.5 ${i < rows.length - 1 ? "border-b border-border" : ""}`}>
            <div>
              <div className="text-[13px] text-text-primary font-medium">{r.label}</div>
              <div className="text-[11.5px] text-text-tertiary mt-0.5">{r.desc}</div>
            </div>
            <Toggle checked={prefs[r.key]} onChange={(v) => setPrefs((p) => ({ ...p, [r.key]: v }))} />
          </div>
        ))}
      </GlassCard>
    </div>
  );
}