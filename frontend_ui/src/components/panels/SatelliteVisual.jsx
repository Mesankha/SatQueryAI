export default function SatelliteVisual({ scene }) {
  return (
    <div
      className="relative w-full aspect-16/10 rounded-xl overflow-hidden border border-border"
      style={{ background: "linear-gradient(160deg, #0d2b2e 0%, #0a1a24 45%, #071018 100%)" }}
    >
      <svg width="100%" height="100%" className="absolute inset-0">
        <defs>
          <pattern id="grid" width="18" height="18" patternUnits="userSpaceOnUse">
            <path d="M 18 0 L 0 0 0 18" fill="none" stroke="rgba(79,216,232,0.12)" strokeWidth="0.6" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
        <line x1="0" y1="50%" x2="100%" y2="50%" stroke="rgba(79,216,232,0.35)" strokeWidth="1" className="animate-scan" />
      </svg>
      <div className="absolute left-[48%] top-[42%]">
        <span className="absolute w-6.5 h-6.5 rounded-full border border-cyan -left-3 -top-3 animate-ping" />
        <span className="block w-1.75 h-1.75 rounded-full bg-cyan shadow-[0_0_10px_#4FD8E8]" />
      </div>
      <div className="absolute top-2 left-2.5 text-[9.5px] font-mono text-cyan/80 tracking-wide">
        {scene.lat.toFixed(3)}N / {scene.lon.toFixed(3)}E
      </div>
      <div className="absolute bottom-2 right-2.5 text-[9.5px] font-mono text-text-tertiary">{scene.resolution}</div>
      <div className="absolute top-2 right-2.5 flex items-center gap-1 text-[9px] text-green font-mono">
        <span className="w-1.5 h-1.5 rounded-full bg-green" /> LIVE FEED
      </div>
    </div>
  );
}