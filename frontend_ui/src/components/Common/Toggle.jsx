export default function Toggle({ checked, onChange }) {
  return (
    <button onClick={() => onChange(!checked)} className={`w-9.5 h-5.5 rounded-full border-none cursor-pointer relative transition-colors duration-180 ${checked ? "bg-cyan" : "bg-white/12"}`}>
      <span className={`absolute top-0.5 w-4.5 h-4.5 rounded-full bg-[#08111a] transition-all duration-180 ${checked ? "left-4.5" : "left-0.5"}`} />
    </button>
  );
}