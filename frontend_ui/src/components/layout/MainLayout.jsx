import Sidebar from "./Sidebar";
import TopNav from "./TopNav";

export default function MainLayout({ route, setRoute, mobileOpen, setMobileOpen, children }) {
  return (
    <div className="font-sans bg-bg text-text-primary w-full h-screen flex overflow-hidden relative">
      <Sidebar
        route={route}
        setRoute={setRoute}
        mobileOpen={mobileOpen}
        setMobileOpen={setMobileOpen}
      />
      <div className="flex-1 min-w-0 flex flex-col h-full">
        <TopNav setMobileOpen={setMobileOpen} route={route} setRoute={setRoute} />
        <div className="flex-1 min-h-0 flex">{children}</div>
      </div>
    </div>
  );
}