
import {
  MessageSquare,
  Database,
  MapPin,
  Settings as SettingsIcon,
  Satellite,
} from "lucide-react";

const NAV_ITEMS = [
  { id: "workspace", icon: MessageSquare, label: "Workspace" },
  { id: "catalog", icon: Database, label: "Catalog" },
  { id: "map", icon: MapPin, label: "Map" },
  { id: "settings", icon: SettingsIcon, label: "Settings" },
];

function Sidebar({
  route,
  setRoute,
  mobileOpen,
  setMobileOpen,
}) {
  return (
    <>
      {/* Desktop Sidebar */}
      <div className="hidden md:flex w-17 shrink-0 h-full border-r border-border bg-bg-elevated flex-col items-center pt-4 gap-1.5">
        
        {/* Logo */}
        <div className="w-9 h-9 rounded-[10px] mb-3 bg-linear-to-br from-cyan to-blue flex items-center justify-center shadow-[0_0_16px_rgba(79,216,232,0.2)]">
          <Satellite
            size={18}
            color="#08111a"
            strokeWidth={2.2}
          />
        </div>

        {/* Navigation */}
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            onClick={() => setRoute(item.id)}
            title={item.label}
            className={`w-11 h-11 rounded-[11px] flex items-center justify-center transition-all duration-150 border cursor-pointer
              ${
                route === item.id
                  ? "bg-cyan/10 border-cyan/25 text-cyan"
                  : "border-transparent text-text-tertiary hover:text-text-primary"
              }`}
          >
            <item.icon size={19} strokeWidth={1.8} />
          </button>
        ))}
      </div>

      {/* Mobile Sidebar */}
      {mobileOpen && (
        <div
          onClick={() => setMobileOpen(false)}
          className="fixed inset-0 bg-black/55 z-40 md:hidden"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="absolute left-0 top-0 bottom-0 w-55 bg-bg-elevated border-r border-border p-4 flex flex-col gap-1"
          >
            {/* Mobile Logo */}
            <div className="flex items-center gap-2.5 mb-5 px-2 py-1">
              <div className="w-7.5 h-7.5 rounded-lg bg-linear-to-br from-cyan to-blue flex items-center justify-center">
                <Satellite
                  size={15}
                  color="#08111a"
                />
              </div>

              <span className="font-display font-semibold text-sm text-text-primary">
                SatQuery AI
              </span>
            </div>

            {/* Mobile Navigation */}
            {NAV_ITEMS.map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  setRoute(item.id);
                  setMobileOpen(false);
                }}
                className={`flex items-center gap-2.5 py-2.5 px-3 rounded-[9px] text-left text-[13.5px] font-medium cursor-pointer border-none
                  ${
                    route === item.id
                      ? "bg-cyan/10 text-cyan"
                      : "text-text-secondary bg-transparent"
                  }`}
              >
                <item.icon
                  size={17}
                  strokeWidth={1.8}
                />

                {item.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

export default Sidebar;