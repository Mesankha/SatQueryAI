import { Sparkles } from "lucide-react";
import AIResponseCard from "./AIResponseCard";
import TypingDots from "./TypingDot";
import HEX from "../../utils/constants";

export default function ChatMessage({ msg, onFocusResult }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex flex-col mb-4.5 animate-msg-in ${isUser ? "items-end" : "items-start"}`}>
      {isUser ? (
        <div className="max-w-[78%]">
          {msg.attachment && (
            <div className="mb-1.5 inline-flex items-center gap-2 bg-surface border border-border rounded-[10px] p-1.5">
              <img src={msg.attachment.url} alt={msg.attachment.name} className="w-10 h-10 object-cover rounded-md" />
              <div className="text-[10.5px] text-text-secondary pr-1.5">
                <div className="text-text-primary font-medium">{msg.attachment.name}</div>
                <div>{msg.attachment.size}</div>
              </div>
            </div>
          )}
          <div className="bg-linear-to-br from-cyan/13 to-blue/8 border border-cyan/20 rounded-tl-2xl rounded-tr-2xl rounded-bl-2xl rounded-br-[3px] py-2.5 px-3.5 text-[13.5px] text-text-primary leading-relaxed">
            {msg.text}
          </div>
        </div>
      ) : msg.thinking ? (
        <div className="flex items-center gap-2">
          <span className="w-5.5 h-5.5 rounded-md bg-cyan/10 flex items-center justify-center">
            <Sparkles size={12} color={HEX.cyan} />
          </span>
          <TypingDots />
        </div>
      ) : (
        <AIResponseCard result={msg.result} onFocus={onFocusResult} />
      )}
    </div>
  );
}