import { Radar } from "lucide-react";
import HEX from "../../utils/constants";

export default function OrbitalVisual() {
  return (
    <div className="relative w-52.5 h-52.5 mx-auto mb-5">
      <svg viewBox="0 0 210 210" width="210" height="210" className="absolute inset-0">
        <defs>
          <radialGradient id="coreGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor={HEX.cyan} stopOpacity="0.35" />
            <stop offset="100%" stopColor={HEX.cyan} stopOpacity="0" />
          </radialGradient>
          <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor={HEX.cyan} />
            <stop offset="100%" stopColor={HEX.blue} />
          </linearGradient>
        </defs>
        <circle cx="105" cy="105" r="70" fill="url(#coreGlow)" />
        {[46, 70, 94].map((r, i) => (
          <circle key={r} cx="105" cy="105" r={r} fill="none" stroke={HEX.border} strokeWidth="1" strokeDasharray={i === 1 ? "2 5" : "none"} />
        ))}
        <circle cx="105" cy="105" r="22" fill="#0F131A" stroke="url(#ringGrad)" strokeWidth="1.5" />
        <g className="origin-[105px_105px] animate-spin" style={{ animationDuration: "14s" }}>
          <circle cx="105" cy="35" r="3" fill={HEX.cyan} />
          <circle cx="105" cy="35" r="7" fill={HEX.cyan} opacity="0.18" />
        </g>
        <g className="origin-[105px_105px] animate-spin-rev">
          <circle cx="199" cy="105" r="2.4" fill={HEX.teal} />
        </g>
        <g className="origin-[105px_105px] animate-spin" style={{ animationDuration: "26s" }}>
          <circle cx="105" cy="11" r="2" fill={HEX.blue} />
        </g>
        <foreignObject x="90" y="90" width="30" height="30">
          <div className="w-7.5 h-7.5 flex items-center justify-center">
            <Radar size={17} color={HEX.cyan} strokeWidth={1.6} />
          </div>
        </foreignObject>
        <line x1="105" y1="105" x2="105" y2="35" stroke={HEX.cyan} strokeWidth="1" opacity="0.5" className="origin-[105px_105px] animate-spin" style={{ animationDuration: "4.5s" }} />
      </svg>
    </div>
  );
}