import { useState, useEffect, useMemo } from "react";
import { Search, ArrowUpDown, Loader2 } from "lucide-react";
import GlassCard from "../Common/GlassCard";
import { getCatalog } from "../../api/catalog";

export default function CatalogPage() {
  const [query, setQuery] = useState("");
  const [sensorFilter, setSensorFilter] = useState("all");
  const [sortKey, setSortKey] = useState("date");
  const [scenes, setScenes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadCatalog = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getCatalog({ limit: 200 });
        setScenes(data.scenes || data);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    loadCatalog();
  }, []);

  const rows = useMemo(() => {
    let r = scenes.filter((s) =>
      (sensorFilter === "all" || s.sensor === sensorFilter) &&
      (s.name?.toLowerCase().includes(query.toLowerCase()) || 
       s.tags?.some((t) => t.toLowerCase().includes(query.toLowerCase())) ||
       s.id?.toLowerCase().includes(query.toLowerCase()))
    );
    r = [...r].sort((a, b) => (sortKey === "date" ? 
      (b.acquisition_time || b.date || "").localeCompare(a.acquisition_time || a.date || "") : 
      (a.name || "").localeCompare(b.name || "")));
    return r;
  }, [query, sensorFilter, sortKey, scenes]);

  return (
    <div className="py-6.5 px-7 max-w-245 mx-auto w-full">
      <div className="text-[11px] tracking-wide text-text-tertiary mb-1.5">WORKSPACE / DATA CATALOG</div>
      <h1 className="font-display text-2xl text-text-primary mb-1.5">Scene catalog</h1>
      <p className="text-[13px] text-text-secondary mb-5">{scenes.length} indexed observations across optical and SAR sensors.</p>

      <div className="flex gap-2.5 mb-4 flex-wrap">
        <div className="flex items-center gap-2 flex-1 min-w-50 bg-surface border border-border rounded-[10px] py-2 px-3">
          <Search size={14} className="text-text-tertiary" />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search scenes, tags, locations..." className="bg-transparent border-none outline-none text-text-primary text-[12.5px] flex-1 placeholder:text-text-tertiary" />
        </div>
        <select value={sensorFilter} onChange={(e) => setSensorFilter(e.target.value)} className="bg-surface border border-border rounded-[10px] py-2 px-3 text-text-secondary text-xs outline-none">
          <option value="all">All sensors</option>
          <option value="Sentinel-1">Sentinel-1</option>
          <option value="Sentinel-2">Sentinel-2</option>
        </select>
        <button onClick={() => setSortKey((k) => (k === "date" ? "name" : "date"))} className="flex items-center gap-1.5 bg-surface border border-border rounded-[10px] py-2 px-3 text-text-secondary text-xs cursor-pointer">
          <ArrowUpDown size={13} /> Sort: {sortKey === "date" ? "Newest" : "Name"}
        </button>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-10">
          <Loader2 className="w-6 h-6 animate-spin text-cyan" />
          <span className="ml-2 text-text-secondary">Loading catalog...</span>
        </div>
      )}

      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-[10px] text-red-400 text-sm mb-4">
          Failed to load catalog: {error}
        </div>
      )}

      {!loading && !error && (
        <GlassCard className="overflow-hidden">
          <div className="hidden sm:grid grid-cols-5 py-2.5 px-4 border-b border-border text-[10.5px] tracking-wide text-text-tertiary">
            <span>SCENE</span><span>SENSOR</span><span>DATE</span><span>RES.</span><span>LOCATION</span>
          </div>
          {rows.length === 0 && <div className="p-7 text-center text-text-tertiary text-[12.5px]">No scenes match this filter.</div>}
          {rows.map((s, i) => (
            <div key={s.id}>
              <div className={`hidden sm:grid grid-cols-5 py-3 px-4 text-[12.5px] text-text-primary transition-colors duration-150 hover:bg-white/2 ${i < rows.length - 1 ? "border-b border-border" : ""}`}>
                <span>
                  {s.name}
                  <span className="block text-[10.5px] text-text-tertiary font-mono">{s.id}</span>
                </span>
                <span className="text-text-secondary">{s.sensor}</span>
                <span className="text-text-secondary font-mono">{s.acquisition_time || s.date}</span>
                <span className="text-text-secondary">{s.resolution}</span>
                <span className="text-text-secondary">{s.origin?.replace(/_/g, " ") || s.origin}</span>
              </div>
              <div className={`sm:hidden py-3 px-4 ${i < rows.length - 1 ? "border-b border-border" : ""}`}>
                <div className="text-[12.5px] text-text-primary font-medium">{s.name}</div>
                <div className="text-[10.5px] text-text-tertiary font-mono mb-1.5">{s.id}</div>
                <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-text-secondary">
                  <span>{s.sensor}</span>
                  <span className="font-mono">{s.acquisition_time || s.date}</span>
                  <span>{s.resolution}</span>
                  <span>{s.origin?.replace(/_/g, " ") || s.origin}</span>
                </div>
              </div>
            </div>
          ))}
        </GlassCard>
      )}
    </div>
  );
}