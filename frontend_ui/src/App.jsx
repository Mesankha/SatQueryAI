import { useState, useEffect } from "react";
import { Layers } from "lucide-react";
import { useHashRoute } from "./components/hooks/useHashRoute";
import MainLayout from "./components/layout/MainLayout";
import WorkspacePage from "./components/chat/WorkspacePage";
import CatalogPage from "./components/pages/CatalogPage";
import MapPage from "./components/pages/MapPage";
import SettingsPage from "./components/pages/SettingsPage";
import SatellitePanel from "./components/panels/SatellitePanel";
import MapModal from "./components/panels/MapModal";
import CompareModal from "./components/panels/CompareModal";
import AnalysisModal from "./components/panels/AnalysisModal";
import { SCENES } from "./components/Data/mockData";

export default function SatQueryAI() {
  const [route, setRoute] = useHashRoute();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [activeResult, setActiveResult] = useState(null);
  const [mapScene, setMapScene] = useState(null);
  const [compareScene, setCompareScene] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [mobilePanelOpen, setMobilePanelOpen] = useState(false);
  const [viewport, setViewport] = useState(typeof window !== "undefined" ? window.innerWidth : 1200);

  useEffect(() => {
    const onResize = () => setViewport(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== "Escape") return;
      setMobileOpen(false);
      setMapScene(null);
      setCompareScene(null);
      setAnalysisResult(null);
      setMobilePanelOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const isCompact = viewport < 900;
  const showRightPanel = route === "workspace" && !isCompact;
  const showMobileFab = route === "workspace" && isCompact;

  return (
    <>
      <MainLayout route={route} setRoute={setRoute} mobileOpen={mobileOpen} setMobileOpen={setMobileOpen}>
        {route === "workspace" && <WorkspacePage setActiveResult={setActiveResult} setMapScene={setMapScene} />}
        {route === "catalog" && <div className="flex-1 overflow-y-auto"><CatalogPage /></div>}
        {route === "map" && <div className="flex-1 overflow-y-auto"><MapPage /></div>}
        {route === "settings" && <div className="flex-1 overflow-y-auto"><SettingsPage /></div>}

        {showRightPanel && (
          <div className="w-85 shrink-0 border-l border-border bg-bg-elevated">
            <SatellitePanel
              activeResult={activeResult}
              onOpenMap={() => setMapScene(activeResult?.scene ?? SCENES[0])}
              onCompare={() => setCompareScene(activeResult?.scene ?? SCENES[0])}
              onAnalysis={() => setAnalysisResult(activeResult)}
            />
          </div>
        )}
      </MainLayout>

      {showMobileFab && (
        <button
          onClick={() => setMobilePanelOpen(true)}
          title="Satellite intelligence"
          className="fixed bottom-24 right-4 z-30 w-13 h-13 rounded-full bg-linear-to-br from-cyan to-blue text-[#08111a] flex items-center justify-center shadow-[0_8px_24px_rgba(79,216,232,0.35)] cursor-pointer border-none"
        >
          <Layers size={20} strokeWidth={2} />
        </button>
      )}

      {mobilePanelOpen && (
        <div onClick={() => setMobilePanelOpen(false)} className="fixed inset-0 bg-black/60 z-40 flex items-end animate-fade-in">
          <div onClick={(e) => e.stopPropagation()} className="w-full max-h-[80vh] bg-bg-elevated border-t border-border rounded-t-2xl overflow-hidden flex flex-col">
            <div className="flex justify-center pt-2 pb-1 shrink-0">
              <span className="w-9 h-1 rounded-full bg-border-light" />
            </div>
            <SatellitePanel
              activeResult={activeResult}
              onOpenMap={() => { setMobilePanelOpen(false); setMapScene(activeResult?.scene ?? SCENES[0]); }}
              onCompare={() => { setMobilePanelOpen(false); setCompareScene(activeResult?.scene ?? SCENES[0]); }}
              onAnalysis={() => { setMobilePanelOpen(false); setAnalysisResult(activeResult); }}
            />
          </div>
        </div>
      )}

      <MapModal scene={mapScene} onClose={() => setMapScene(null)} compact={isCompact} />
      <CompareModal scene={compareScene} onClose={() => setCompareScene(null)} />
      <AnalysisModal result={analysisResult} onClose={() => setAnalysisResult(null)} />
    </>
  );
}