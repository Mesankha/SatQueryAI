import { useRef, useEffect } from "react";
import { useLeaflet } from "../hooks/useLeaflet";
import HEX from "../../utils/constants";

function sensorMarkerColor(sensor) {
  return sensor === "Sentinel-1" ? HEX.blue : HEX.cyan;
}

export default function LeafletMap({ scenes, focusId, onSelect, height = 280, zoom = 6 }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const markersRef = useRef({});
  const { L, error } = useLeaflet();

  useEffect(() => {
    if (!L || !containerRef.current || mapRef.current) return;
    const focusScene = scenes.find((s) => s.id === focusId) || scenes[0];
    const map = L.map(containerRef.current, {
      zoomControl: true,
      attributionControl: true,
      scrollWheelZoom: true,
    }).setView([focusScene.lat, focusScene.lon], zoom);

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 19,
    }).addTo(map);

    scenes.forEach((s) => {
      const color = sensorMarkerColor(s.sensor);
      const icon = L.divIcon({
        className: "",
        html: `<div style="position:relative;width:16px;height:16px;">
                 <div style="position:absolute;inset:-8px;border-radius:50%;border:1.5px solid ${color};opacity:0.55;"></div>
                 <div style="position:absolute;left:4px;top:4px;width:8px;height:8px;border-radius:50%;background:${color};box-shadow:0 0 8px ${color};"></div>
               </div>`,
        iconSize: [16, 16],
        iconAnchor: [8, 8],
      });
      const marker = L.marker([s.lat, s.lon], { icon }).addTo(map);
      marker.bindPopup(
        `<div style="font-family:Inter,sans-serif;min-width:150px">
           <div style="font-weight:600;font-size:12.5px;margin-bottom:3px;color:#0f172a">${s.name}</div>
           <div style="font-size:11px;color:#475569">${s.sensor} &middot; ${s.date}</div>
           <div style="font-size:10.5px;color:#64748b;margin-top:2px">${s.resolution}</div>
         </div>`
      );
      if (onSelect) marker.on("click", () => onSelect(s));
      markersRef.current[s.id] = marker;
    });

    mapRef.current = map;
    setTimeout(() => map.invalidateSize(), 50);
  }, [L, focusId, onSelect, scenes, zoom]);

  useEffect(() => {
    if (!mapRef.current || !focusId) return;
    const scene = scenes.find((s) => s.id === focusId);
    if (!scene) return;
    mapRef.current.flyTo([scene.lat, scene.lon], Math.max(mapRef.current.getZoom(), 8), { duration: 0.9 });
    const marker = markersRef.current[scene.id];
    if (marker) marker.openPopup();
  }, [focusId, scenes]);

  useEffect(() => () => { if (mapRef.current) { mapRef.current.remove(); mapRef.current = null; } }, []);

  if (error) {
    return (
      <div style={{ height }} className="rounded-xl border border-border flex items-center justify-center text-text-tertiary text-xs">
        Map failed to load — check your network connection.
      </div>
    );
  }

  return (
    <div style={{ height }} className="relative rounded-xl overflow-hidden border border-border">
      <div ref={containerRef} className="w-full h-full bg-[#0b0f14]" />
      {!L && (
        <div className="absolute inset-0 flex items-center justify-center gap-2 bg-surface">
          <span className="w-3.5 h-3.5 border-2 border-border border-t-cyan rounded-full animate-spin" />
          <span className="text-xs text-text-tertiary">Loading map…</span>
        </div>
      )}
    </div>
  );
}