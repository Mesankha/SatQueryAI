import { useState, useRef } from "react";
import { Send, Paperclip, Mic, Square, X } from "lucide-react";
import { useAudioRecorder } from "../hooks/useAudioRecorder";
import HEX from "../../utils/constants";

export default function Composer({ onSend, disabled }) {
  const [text, setText] = useState("");
  const [attachment, setAttachment] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef(null);
  const textareaRef = useRef(null);
  const dragCounterRef = useRef(0);
  const rec = useAudioRecorder();

  const handleFiles = (files) => {
    const file = files[0];
    if (!file || !file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    setAttachment({ url, name: file.name, size: `${(file.size / 1024).toFixed(0)} KB` });
  };

  const removeAttachment = () => {
    if (attachment) URL.revokeObjectURL(attachment.url);
    setAttachment(null);
  };

  const submit = () => {
    if (disabled || (!text.trim() && !attachment)) return;
    onSend(text.trim() || "Describe this image", attachment);
    setText("");
    setAttachment(null);
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  return (
    <div
      onDragEnter={(e) => { e.preventDefault(); dragCounterRef.current += 1; setDragging(true); }}
      onDragOver={(e) => { e.preventDefault(); }}
      onDragLeave={(e) => {
        e.preventDefault();
        dragCounterRef.current -= 1;
        if (dragCounterRef.current <= 0) { dragCounterRef.current = 0; setDragging(false); }
      }}
      onDrop={(e) => {
        e.preventDefault();
        dragCounterRef.current = 0;
        setDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
      className={`relative rounded-2xl p-2.5 bg-surface border transition-all duration-180 shadow-[0_8px_24px_rgba(0,0,0,0.25)]
        ${dragging ? "border-cyan ring-4 ring-cyan/15" : "border-border"}`}
    >
      {dragging && (
        <div className="absolute inset-1.5 rounded-xl border border-dashed border-cyan/55 bg-cyan/6 flex items-center justify-center text-[12.5px] text-cyan z-10 pointer-events-none">
          Drop image to attach
        </div>
      )}

      {attachment && (
        <div className="flex items-center gap-2 bg-bg-elevated border border-border rounded-[10px] p-1.5 mb-2 w-fit">
          <img src={attachment.url} alt="" className="w-8.5 h-8.5 rounded-md object-cover" />
          <div className="text-[10.5px] text-text-secondary">
            <div className="text-text-primary">{attachment.name}</div>
            <div>{attachment.size}</div>
          </div>
          <button onClick={removeAttachment} className="bg-transparent border-none text-text-tertiary cursor-pointer p-1">
            <X size={13} />
          </button>
        </div>
      )}

      {rec.recording ? (
        <div className="flex items-center gap-2.5 py-2 px-2.5">
          <span className="w-1.75 h-1.75 rounded-full bg-rose animate-ping" />
          <span className="text-[12.5px] text-text-primary">Listening…</span>
          <div className="flex-1 flex items-center gap-0.5 h-5">
            {Array.from({ length: 28 }).map((_, i) => (
              <span
                key={i}
                className="w-[2.5px] rounded-sm bg-cyan transition-[height] duration-100"
                style={{ height: `${4 + Math.abs(Math.sin(i * 0.8)) * rec.level * 16}px` }}
              />
            ))}
          </div>
          <button
            onClick={rec.stop}
            className="flex items-center gap-1.5 bg-rose/13 border border-rose/35 text-rose rounded-lg py-1.5 px-2.5 text-[11.5px] cursor-pointer"
          >
            <Square size={11} fill={HEX.rose} />
            Stop
          </button>
        </div>
      ) : (
        <div className="flex items-end gap-1.5">
          <button
            onClick={() => fileRef.current?.click()}
            className="w-8.5 h-8.5 shrink-0 flex items-center justify-center bg-transparent border-none text-text-tertiary cursor-pointer rounded-lg hover:text-cyan"
          >
            <Paperclip size={17} strokeWidth={1.8} />
          </button>

          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleFiles(e.target.files)}
            onClick={(e) => { e.target.value = null; }}
          />

          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Ask SatQuery anything about Earth..."
            rows={1}
            disabled={disabled}
            className="flex-1 resize-none bg-transparent border-none outline-none text-text-primary text-[13.5px] leading-relaxed py-2 px-0.5 font-sans max-h-30 placeholder:text-text-tertiary"
          />

          <button
            onClick={rec.start}
            title="Voice input"
            className="w-8.5 h-8.5 shrink-0 flex items-center justify-center bg-transparent border-none text-text-tertiary cursor-pointer rounded-lg hover:text-cyan"
          >
            <Mic size={17} strokeWidth={1.8} />
          </button>

          <button
            onClick={submit}
            disabled={disabled || (!text.trim() && !attachment)}
            className={`w-8.5 h-8.5 shrink-0 rounded-[9px] border-none flex items-center justify-center transition-all duration-150 ${
              disabled || (!text.trim() && !attachment)
                ? "bg-white/6 text-text-tertiary cursor-default"
                : "bg-linear-to-br from-cyan to-blue text-[#08111a] cursor-pointer"
            }`}
          >
            <Send size={15} strokeWidth={2} />
          </button>
        </div>
      )}

      {rec.error && <div className="text-[10.5px] text-rose pt-1 px-1.5">{rec.error}</div>}
    </div>
  );
}