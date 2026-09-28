export const ACCENT_CLASSES = {
  cyan: { from: "from-cyan/70", to: "to-cyan", text: "text-cyan" },
  teal: { from: "from-teal/70", to: "to-teal", text: "text-teal" },
  blue: { from: "from-blue/70", to: "to-blue", text: "text-blue" },
  amber: { from: "from-amber/70", to: "to-amber", text: "text-amber" },
  rose: { from: "from-rose/70", to: "to-rose", text: "text-rose" },
};

export function btnGhost(primary) {
  return `flex items-center justify-center gap-1.5 py-2.5 px-3 rounded-[9px] text-xs cursor-pointer transition-all duration-150 border
    ${primary ? "bg-cyan/10 border-cyan/25 text-cyan" : "bg-surface border-border text-text-secondary hover:bg-surface-hover"}`;
}