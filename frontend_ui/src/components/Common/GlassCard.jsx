export default function GlassCard({ children, className = "" }) {
  return (
    <div className={`bg-linear-to-b from-white/2.5 to-white/1 border border-border rounded-[14px] backdrop-blur-md ${className}`}>
      {children}
    </div>
  );
}