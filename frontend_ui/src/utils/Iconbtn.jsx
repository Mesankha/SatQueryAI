export default function IconBtn({ icon: Icon, onClick, active, title, size = 18, badge }) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`relative w-8.5 h-8.5 rounded-[9px] flex items-center justify-center transition-all duration-150 cursor-pointer border
        ${active ? "bg-cyan/10 border-cyan/25 text-cyan" : "border-transparent text-text-secondary hover:bg-white/6"}`}
    >
      <Icon size={size} strokeWidth={1.8} />
      {badge && <span className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-rose ring-2 ring-bg-elevated" />}
    </button>
  );
}
