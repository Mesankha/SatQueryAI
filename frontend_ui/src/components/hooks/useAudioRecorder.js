import { useState, useRef, useCallback } from "react";

export function useAudioRecorder() {
  const [recording, setRecording] = useState(false);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState(null);
  const mediaRef = useRef(null);
  const rafRef = useRef(null);
  const streamRef = useRef(null);
  const ctxRef = useRef(null);

  const stop = useCallback(() => {
    setRecording(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (mediaRef.current && mediaRef.current.state !== "inactive") mediaRef.current.stop();
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    if (ctxRef.current) { ctxRef.current.close(); ctxRef.current = null; }
  }, []);

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      ctxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);
      const tick = () => {
        analyser.getByteFrequencyData(data);
        const avg = data.reduce((a, b) => a + b, 0) / data.length;
        setLevel(Math.min(1, avg / 90));
        rafRef.current = requestAnimationFrame(tick);
      };
      tick();
      const mr = new MediaRecorder(stream);
      mediaRef.current = mr;
      mr.start();
      setRecording(true);
    } catch (e) {
      setError("Microphone access unavailable in this environment.");
    }
  }, []);

  return { recording, level, error, start, stop };
}