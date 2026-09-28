import { useState, useEffect } from "react";
export function useHashRoute() {
  const getRoute = () => (window.location.hash.replace("#/", "") || "workspace");
  const [route, setRouteState] = useState(getRoute());
  useEffect(() => {
    const onHash = () => setRouteState(getRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const setRoute = (r) => { window.location.hash = `/${r}`; };
  return [route, setRoute];
}