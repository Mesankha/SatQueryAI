import { useState, useEffect } from "react";
import HEX from "../../utils/constants";

export default function ConfidenceRing({ value }) {
  const r = 20, c = 2 * Math.PI * r;
  const [dash, setDash] = useState(c);
  useEffect(() => {
    const id = requestAnimationFrame(() => setDash(c - value * c));
    return () => cancelAnimationFrame(id);
  }, [value, c]);
  return (
    <svg width="52" height="52" viewBox="0 0 52 52">
      <circle cx="26" cy="26" r={r} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="4" />
      <circle
        cx="26" cy="26" r={r} fill="none" stroke={HEX.cyan} strokeWidth="4" strokeLinecap="round"
        strokeDasharray={c} strokeDashoffset={dash}
        transform="rotate(-90 26 26)"
        style={{ transition: "stroke-dashoffset 1000ms cubic-bezier(0.16,1,0.3,1)" }}
      />
      <text x="26" y="30" textAnchor="middle" fontSize="12" fontFamily="'JetBrains Mono', monospace" fill="#E8EDF3" fontWeight="600">
        {Math.round(value * 100)}
      </text>
    </svg>
  );
}