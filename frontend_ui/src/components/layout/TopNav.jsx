import {
  Menu , 
  User,
} from "lucide-react"

import { useEffect, useState, useMemo } from "react";

import seededFloat from "../../utils/SeededFloat"
import HEX from "../../utils/constants"
import StatusDot from '../../utils/StatusDot'
import IconBtn from '../../utils/Iconbtn'




export default function TopNav({ setMobileOpen, route, setRoute }) {
  const [time, setTime] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setTime(Date.now()), 3000);
    return () => clearInterval(id);
  }, []);
  const latency = useMemo(() => (40 + seededFloat(time, 0, 30)).toFixed(0), [time]);

  return (
    <div className="h-14 shrink-0 border-b border-border flex items-center justify-between px-4.5 bg-bg-elevated">
      <div className="flex items-center gap-3.5 min-w-0">
        <button onClick={() => setMobileOpen(true)} className="md:hidden flex items-center justify-center bg-transparent border-none text-text-secondary cursor-pointer">
          <Menu size={20} />
        </button>
        <div className="flex flex-col leading-tight">
          <span className="font-display font-semibold text-[15px] text-text-primary tracking-wide">SATQUERY AI</span>
          <span className="text-[10.5px] text-text-tertiary tracking-widest">EARTH OBSERVATION INTELLIGENCE</span>
        </div>
        <div className="hidden lg:flex items-center gap-4 ml-4.5 pl-4.5 border-l border-border">
          <StatusDot color={HEX.green} label="SYSTEM ONLINE" />
          <StatusDot color={HEX.cyan} label="SATELLITE DATA" />
          <StatusDot color={HEX.blue} label="AI READY" />
          <span className="text-[10.5px] text-text-tertiary font-mono">{latency}ms</span>
        </div>
      </div>
      <div className="flex items-center gap-1">
        <div className="w-px h-5 bg-border mx-1" />
        <div className="w-7.5 h-7.5 rounded-[9px] bg-linear-to-br from-[#2A3441] to-[#1a212b] border border-border-light flex items-center justify-center">
          <User size={15} className="text-text-secondary" strokeWidth={1.8} />
        </div>
      </div>
    </div>
  );
}