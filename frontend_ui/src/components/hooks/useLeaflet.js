import { useState, useEffect } from "react";

let leafletLoadPromise = null;
function loadLeaflet() {
  if (typeof window !== "undefined" && window.L) return Promise.resolve(window.L);
  if (leafletLoadPromise) return leafletLoadPromise;
  leafletLoadPromise = new Promise((resolve, reject) => {
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css";
    document.head.appendChild(css);

    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js";
    script.async = true;
    script.onload = () => resolve(window.L);
    script.onerror = () => reject(new Error("Leaflet failed to load"));
    document.head.appendChild(script);
  });
  return leafletLoadPromise;
}

export function useLeaflet() {
  const [L, setL] = useState(typeof window !== "undefined" ? window.L || null : null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let mounted = true;
    loadLeaflet().then((l) => { if (mounted) setL(l); }).catch(() => { if (mounted) setError(true); });
    return () => { mounted = false; };
  }, []);
  return { L, error };
}