export default function StatusDot({ color, label }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="relative inline-flex w-1.5 h-1.5">
        <span className="absolute inset-0 rounded-full opacity-50 animate-ping" style={{ background: color }} />
        <span className="relative w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      </span>
      <span className="text-[11px] text-text-secondary tracking-wide font-medium">{label}</span>
    </div>
  );
}
