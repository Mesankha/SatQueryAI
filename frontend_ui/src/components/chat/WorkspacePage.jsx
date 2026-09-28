import { useState, useRef, useEffect } from "react";
import EmptyState from "./EmptyState";
import ChatMessage from "./ChatMessage";
import Composer from "./Composer";
import { queryBackend } from "../../api/queryBackend";

export default function WorkspacePage({ setActiveResult, setMapScene }) {
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const send = async (text, attachment) => {
    const userMsg = { role: "user", text, attachment, id: Date.now() };
    const thinkingId = Date.now() + 1;
    const thinkingMsg = { role: "ai", thinking: true, id: thinkingId };
    setMessages((m) => [...m, userMsg, thinkingMsg]);
    setBusy(true);

    try {
      const response = await queryBackend(text);
      const result = response.results?.[0] ?? response;
      setActiveResult(result);
      setMessages((m) => m.map((msg) => (msg.id === thinkingId ? { role: "ai", result, id: msg.id } : msg)));
    } catch (e) {
      const errorResult = { 
        explanation_text: `Error: ${e.message}`,
        confidence_band: "low",
        model_outputs: { caption: "Query failed. Check backend connection." }
      };
      setActiveResult(errorResult);
      setMessages((m) => m.map((msg) => (msg.id === thinkingId ? { role: "ai", result: errorResult, id: msg.id } : msg)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col h-full flex-1 min-w-0">
      <div className="px-6 pt-3.5 shrink-0">
        <div className="text-[11px] tracking-wide text-text-tertiary">WORKSPACE / EARTH OBSERVATION</div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 pb-5 pt-2.5">
        {messages.length === 0 ? (
          <EmptyState onSuggestion={send} />
        ) : (
          <div className="max-w-160 mx-auto">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} msg={msg} onFocusResult={(r) => { setActiveResult(r); setMapScene(null); }} />
            ))}
          </div>
        )}
      </div>

      <div className="px-6 pb-5 shrink-0">
        <div className="max-w-160 mx-auto">
          <Composer onSend={send} disabled={busy} />
        </div>
      </div>
    </div>
  );
}