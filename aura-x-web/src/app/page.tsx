"use client";

import { useState, useRef, useEffect } from "react";
import {
  UploadCloud, File, Trash, Send, Loader2, Bot, User,
  CheckCircle, Copy, Mic, MicOff, Download, ChevronDown,
  ChevronUp, BookOpen, Clock, X, Rocket, Check,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// ── Types ──────────────────────────────────────────────────────────────────
type Message = {
  id: string;
  role: "user" | "bot";
  content: string;
  sources?: string[];
  score?: number;
};

type Toast = { id: string; message: string; type: "success" | "error" | "info" };
type DocStatus = "loading" | "indexed" | "error";
type PreloadedDoc = { name: string; status: DocStatus; chunks?: number };

// ── Animated Star Canvas ───────────────────────────────────────────────────
function StarCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    let raf: number;

    type Star = { x: number; y: number; r: number; alpha: number; speed: number; dx: number; dy: number };
    let stars: Star[] = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const init = () => {
      stars = Array.from({ length: 220 }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        r: Math.random() * 1.4 + 0.2,
        alpha: Math.random(),
        speed: (Math.random() * 0.004 + 0.001) * (Math.random() > 0.5 ? 1 : -1),
        dx: (Math.random() - 0.5) * 0.08,
        dy: (Math.random() - 0.5) * 0.08,
      }));
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const s of stars) {
        s.alpha += s.speed;
        if (s.alpha > 1 || s.alpha < 0) s.speed *= -1;
        s.x = (s.x + s.dx + canvas.width) % canvas.width;
        s.y = (s.y + s.dy + canvas.height) % canvas.height;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${Math.max(0, Math.min(1, s.alpha))})`;
        ctx.fill();
      }

      // Nebula glows
      const glow1 = ctx.createRadialGradient(canvas.width * 0.75, canvas.height * 0.2, 0, canvas.width * 0.75, canvas.height * 0.2, 320);
      glow1.addColorStop(0, "rgba(102,126,234,0.06)");
      glow1.addColorStop(1, "transparent");
      ctx.fillStyle = glow1;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const glow2 = ctx.createRadialGradient(canvas.width * 0.1, canvas.height * 0.8, 0, canvas.width * 0.1, canvas.height * 0.8, 260);
      glow2.addColorStop(0, "rgba(118,75,162,0.07)");
      glow2.addColorStop(1, "transparent");
      ctx.fillStyle = glow2;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      raf = requestAnimationFrame(draw);
    };

    resize();
    init();
    draw();
    window.addEventListener("resize", () => { resize(); init(); });
    return () => { cancelAnimationFrame(raf); };
  }, []);

  return <canvas ref={ref} className="fixed inset-0 w-full h-full z-0 pointer-events-none" />;
}

// ── Toast System ──────────────────────────────────────────────────────────
function ToastContainer({ toasts, remove }: { toasts: Toast[]; remove: (id: string) => void }) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
      <AnimatePresence>
        {toasts.map((t) => (
          <motion.div
            key={t.id}
            initial={{ opacity: 0, x: 60, scale: 0.9 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 60 }}
            className="pointer-events-auto flex items-center gap-3 px-4 py-3 rounded-xl shadow-2xl text-sm font-medium"
            style={{
              background: t.type === "success" ? "rgba(20,83,45,0.9)" : t.type === "error" ? "rgba(127,29,29,0.9)" : "rgba(30,27,75,0.9)",
              border: `1px solid ${t.type === "success" ? "rgba(74,222,128,0.4)" : t.type === "error" ? "rgba(248,113,113,0.4)" : "rgba(165,180,252,0.4)"}`,
              backdropFilter: "blur(16px)",
              color: t.type === "success" ? "#86efac" : t.type === "error" ? "#fca5a5" : "#c4b5fd",
            }}
          >
            {t.type === "success" ? <Check size={15} /> : t.type === "error" ? <X size={15} /> : <Rocket size={15} />}
            <span>{t.message}</span>
            <button onClick={() => remove(t.id)} style={{ opacity: 0.5 }} className="hover:opacity-100 ml-1"><X size={13} /></button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

// ── Constants ─────────────────────────────────────────────────────────────
const SUGGESTED = [
  "What is the role of IN-SPACe?",
  "What are ISRO's key missions in 2024-25?",
  "How does India assign satellite spectrum?",
  "What activities require authorization from IN-SPACe?",
  "What is NSIL's role in commercial launches?",
  "What are the spectrum bands for satellite internet in India?",
  "What is India's vision for space by 2047?",
  "How does India regulate private space companies?",
];

const INIT_DOCS: PreloadedDoc[] = [
  { name: "Indian Space Policy 2023", status: "loading" },
  { name: "TRAI Satellite Spectrum 2025", status: "loading" },
  { name: "IN-SPACe NGP 2024", status: "loading" },
  { name: "ISRO Annual Report 2024-25", status: "loading" },
  { name: "NSIL Annual Report 2023-24", status: "loading" },
];

// ── Main Component ────────────────────────────────────────────────────────
export default function Home() {
  const [messages, setMessages] = useState<Message[]>([{
    id: "init",
    role: "bot",
    content: "🚀 Welcome to Nova-X! I am pre-trained on India's key space policy documents. You can ask me anything or try one of the suggested questions below. You can also upload additional documents to expand my knowledge.",
  }]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [docs, setDocs] = useState<PreloadedDoc[]>(INIT_DOCS);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Seed pre-loaded docs on mount
  useEffect(() => {
    fetch("/api/seed")
      .then((r) => r.json())
      .then((data) => {
        if (data.docs) {
          setDocs(data.docs);
          const indexed = data.docs.filter((d: PreloadedDoc) => d.status === "indexed").length;
          if (indexed > 0) {
            addToast(`✅ ${indexed} document${indexed > 1 ? "s" : ""} indexed into knowledge base!`, "success");
          } else {
            addToast("⚠️ Documents found but indexing failed. Check API key.", "error");
          }
        } else {
          setDocs(INIT_DOCS.map((d) => ({ ...d, status: "error" as DocStatus })));
          addToast("❌ Knowledge base failed to load.", "error");
        }
      })
      .catch(() => setDocs(INIT_DOCS.map((d) => ({ ...d, status: "error" as DocStatus }))));
  }, []);

  const addToast = (message: string, type: Toast["type"] = "info") => {
    const id = Math.random().toString(36).slice(2);
    setToasts((p) => [...p, { id, message, type }]);
    setTimeout(() => setToasts((p) => p.filter((t) => t.id !== id)), 4000);
  };

  const removeToast = (id: string) => setToasts((p) => p.filter((t) => t.id !== id));

  const submitFile = async () => {
    if (!file) return;
    setIsUploading(true);
    setUploadStatus("Indexing...");
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus("✓ Indexed!");
        addToast(`"${file.name}" indexed!`, "success");
        setMessages((p) => [...p, { id: Date.now().toString(), role: "bot", content: `✅ Successfully indexed "${file.name}". Ask me anything about it!` }]);
        setDocs((prev) => [...prev, { name: file.name, status: "indexed", chunks: data.chunks }]);
      } else {
        setUploadStatus("Error");
        addToast(data.error || "Upload failed", "error");
      }
    } catch {
      addToast("Network error", "error");
      setUploadStatus("Error");
    } finally {
      setIsUploading(false);
    }
  };

  const handleSend = async (q?: string) => {
    const query = (q || input).trim();
    if (!query || isLoading) return;
    setInput("");
    const uid = Date.now().toString();
    setMessages((p) => [...p, { id: uid + "u", role: "user", content: query }]);
    setQueryHistory((p) => [query, ...p.slice(0, 4)]);
    setIsLoading(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      if (res.ok) {
        setMessages((p) => [...p, { id: uid + "b", role: "bot", content: data.answer, sources: data.sources, score: data.score }]);
      } else {
        setMessages((p) => [...p, { id: uid + "b", role: "bot", content: `Error: ${data.error}` }]);
        addToast(data.error || "Error", "error");
      }
    } catch {
      setMessages((p) => [...p, { id: uid + "b", role: "bot", content: "Network error." }]);
    } finally {
      setIsLoading(false);
    }
  };

  const copyMsg = (id: string, content: string) => {
    navigator.clipboard.writeText(content);
    setCopiedId(id);
    addToast("Copied!", "success");
    setTimeout(() => setCopiedId(null), 2000);
  };

  const toggleSources = (id: string) => {
    setExpandedSources((p) => {
      const n = new Set(p);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  };

  const exportChat = () => {
    const text = messages.map((m) => `${m.role === "user" ? "You" : "AURA-X"}: ${m.content}`).join("\n\n---\n\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
    a.download = `aura-x-${Date.now()}.txt`;
    a.click();
    addToast("Chat exported!", "success");
  };

  const toggleVoice = () => {
    if (isListening) { recognitionRef.current?.stop(); setIsListening(false); return; }
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) { addToast("Voice not supported in this browser", "error"); return; }
    const recognition = new SR();
    recognition.continuous = false;
    recognition.lang = "en-US";
    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onresult = (e: any) => { setInput(e.results[0][0].transcript); addToast("Voice captured!", "success"); };
    recognition.onerror = () => { setIsListening(false); addToast("Voice error", "error"); };
    recognitionRef.current = recognition;
    recognition.start();
  };

  const isDark = (role: string) => role === "bot";

  return (
    <div className="flex h-screen w-full font-sans overflow-hidden relative" style={{ background: "#050505" }}>
      <StarCanvas />
      <ToastContainer toasts={toasts} remove={removeToast} />

      {/* ── Sidebar ── */}
      <aside className="w-72 flex flex-col p-5 z-10 shrink-0 relative" style={{ background: "rgba(5,5,8,0.85)", backdropFilter: "blur(24px)", borderRight: "1px solid rgba(255,255,255,0.08)" }}>
        
        {/* Logo */}
        <div className="mb-7">
          <div className="flex items-center gap-2.5 mb-1">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: "linear-gradient(135deg,#667eea,#764ba2)" }}>
              <Rocket size={16} className="text-white" />
            </div>
            <span className="text-xl font-bold" style={{ background: "linear-gradient(to right,#667eea,#a78bfa)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Nova-X</span>
          </div>
          <p className="text-white/35 text-xs pl-10">Space Policy Intelligence</p>
        </div>

        {/* Knowledge Base */}
        <div className="mb-5">
          <p className="text-xs font-semibold text-white/40 uppercase tracking-widest mb-2.5 flex items-center gap-1.5">
            <BookOpen size={11} /> Knowledge Base
          </p>
          <div className="space-y-1.5">
            {docs.map((doc, i) => (
              <div key={i} className="flex items-center gap-2.5 p-2.5 rounded-lg" style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)" }}>
                <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${doc.status === "indexed" ? "bg-emerald-400" : doc.status === "loading" ? "bg-amber-400 animate-pulse" : "bg-red-400"}`} />
                <span className="text-xs text-white/65 truncate flex-1">{doc.name}</span>
                {doc.chunks && <span className="text-xs text-white/25 shrink-0">{doc.chunks}c</span>}
              </div>
            ))}
          </div>
        </div>

        {/* Upload */}
        <div className="mb-5">
          <p className="text-xs font-semibold text-white/40 uppercase tracking-widest mb-2.5">Upload Document</p>
          <div className="rounded-xl p-4 flex flex-col items-center text-center" style={{ background: "rgba(255,255,255,0.03)", border: "1px dashed rgba(255,255,255,0.15)" }}>
            <UploadCloud size={24} style={{ color: "#667eea" }} className="mb-2" />
            <p className="text-white/40 text-xs mb-3">PDF or TXT · Max 10MB</p>
            <input type="file" accept=".pdf,.txt" className="hidden" id="fu" onChange={(e) => { if (e.target.files?.[0]) setFile(e.target.files[0]); }} />
            <label htmlFor="fu" className="text-xs px-3 py-1.5 rounded-lg cursor-pointer text-white font-medium transition-opacity hover:opacity-90" style={{ background: "#667eea" }}>
              Select File
            </label>
          </div>
          <AnimatePresence>
            {file && (
              <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="mt-2 flex items-center gap-2 p-2 rounded-lg" style={{ background: "rgba(255,255,255,0.05)" }}>
                <File size={13} style={{ color: "#a78bfa" }} />
                <span className="text-xs flex-1 truncate text-white/60">{file.name}</span>
                <button onClick={() => { setFile(null); setUploadStatus(""); }} className="text-white/25 hover:text-red-400"><Trash size={12} /></button>
              </motion.div>
            )}
          </AnimatePresence>
          {file && (
            <button onClick={submitFile} disabled={isUploading} className="mt-2 w-full text-xs py-2 rounded-lg text-white font-medium flex items-center justify-center gap-1.5 transition-opacity disabled:opacity-50" style={{ background: "#764ba2" }}>
              {isUploading ? <Loader2 size={12} className="animate-spin" /> : <CheckCircle size={12} />}
              {isUploading ? "Indexing..." : "Index Document"}
            </button>
          )}
          {uploadStatus && <p className="text-xs text-center mt-1.5 text-white/35">{uploadStatus}</p>}
        </div>

        {/* Recent Queries */}
        {queryHistory.length > 0 && (
          <div className="flex-1 overflow-y-auto smooth-scrollbar">
            <p className="text-xs font-semibold text-white/40 uppercase tracking-widest mb-2.5 flex items-center gap-1.5">
              <Clock size={11} /> Recent Queries
            </p>
            <div className="space-y-1">
              {queryHistory.map((q, i) => (
                <button key={i} onClick={() => handleSend(q)} className="w-full text-left text-xs text-white/45 hover:text-white/80 py-1.5 px-2 rounded-lg hover:bg-white/5 truncate transition-colors">
                  ↩ {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </aside>

      {/* ── Main Chat ── */}
      <main className="flex-1 flex flex-col relative h-full z-10">
        
        {/* Header */}
        <header className="h-14 flex items-center justify-between px-6 shrink-0" style={{ background: "rgba(5,5,8,0.7)", backdropFilter: "blur(20px)", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-white/75">🛰️ Space Policy Terminal</span>
          </div>
          <button onClick={exportChat} className="flex items-center gap-1.5 text-xs text-white/40 hover:text-white/75 transition-colors px-3 py-1.5 rounded-lg hover:bg-white/5">
            <Download size={13} /> Export Chat
          </button>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 md:px-10 py-6 smooth-scrollbar space-y-5">
          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div key={msg.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}
                className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse ml-auto max-w-2xl" : "max-w-2xl"}`}>

                {/* Avatar */}
                <div className="w-8 h-8 rounded-full shrink-0 flex items-center justify-center text-white shadow-lg"
                  style={{ background: msg.role === "user" ? "linear-gradient(135deg,#667eea,#764ba2)" : "rgba(118,75,162,0.6)", border: "1px solid rgba(255,255,255,0.1)" }}>
                  {msg.role === "user" ? <User size={15} /> : <Rocket size={14} />}
                </div>

                <div className={`flex flex-col gap-1.5 flex-1 ${msg.role === "user" ? "items-end" : "items-start"}`}>
                  
                  {/* Relevance Score */}
                  {msg.score !== undefined && msg.role === "bot" && (
                    <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ background: "rgba(102,126,234,0.15)", color: "#a5b4fc", border: "1px solid rgba(102,126,234,0.25)" }}>
                      {Math.round(msg.score * 100)}% relevance
                    </span>
                  )}

                  {/* Bubble */}
                  <div className={`px-4 py-3 rounded-2xl text-sm whitespace-pre-wrap leading-relaxed ${msg.role === "user" ? "rounded-tr-sm" : "rounded-tl-sm"}`}
                    style={{
                      background: msg.role === "user" ? "linear-gradient(135deg,#667eea,#764ba2)" : "rgba(255,255,255,0.055)",
                      border: msg.role === "bot" ? "1px solid rgba(255,255,255,0.09)" : "none",
                      color: "#f1f5f9",
                    }}>
                    {msg.content}
                  </div>

                  {/* Bot controls */}
                  {msg.role === "bot" && (
                    <div className="flex items-center gap-3 pl-1">
                      <button onClick={() => copyMsg(msg.id, msg.content)} className="flex items-center gap-1 text-xs transition-colors" style={{ color: copiedId === msg.id ? "#86efac" : "rgba(255,255,255,0.3)" }}>
                        {copiedId === msg.id ? <Check size={12} /> : <Copy size={12} />}
                        <span>{copiedId === msg.id ? "Copied" : "Copy"}</span>
                      </button>
                      {msg.sources && msg.sources.length > 0 && (
                        <button onClick={() => toggleSources(msg.id)} className="flex items-center gap-1 text-xs transition-colors hover:text-white/60" style={{ color: "rgba(255,255,255,0.3)" }}>
                          {expandedSources.has(msg.id) ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                          <span>{Array.from(new Set(msg.sources)).length} source(s)</span>
                        </button>
                      )}
                    </div>
                  )}

                  {/* Citation Expander */}
                  <AnimatePresence>
                    {msg.role === "bot" && expandedSources.has(msg.id) && msg.sources && (
                      <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
                        className="rounded-xl p-3 text-xs space-y-1.5 overflow-hidden w-full" style={{ background: "rgba(102,126,234,0.08)", border: "1px solid rgba(102,126,234,0.2)" }}>
                        <p className="text-white/40 font-medium uppercase tracking-wider text-xs mb-2">Referenced Sources</p>
                        {Array.from(new Set(msg.sources)).map((s, i) => (
                          <div key={i} className="flex items-center gap-2">
                            <BookOpen size={11} style={{ color: "#a5b4fc" }} />
                            <span style={{ color: "#a5b4fc" }}>{s}</span>
                          </div>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Typing Indicator */}
          {isLoading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3 max-w-2xl">
              <div className="w-8 h-8 rounded-full shrink-0 flex items-center justify-center text-white" style={{ background: "rgba(118,75,162,0.6)", border: "1px solid rgba(255,255,255,0.1)" }}>
                <Rocket size={14} />
              </div>
              <div className="px-4 py-3.5 rounded-2xl rounded-tl-sm flex items-center gap-1.5" style={{ background: "rgba(255,255,255,0.055)", border: "1px solid rgba(255,255,255,0.09)" }}>
                {[0, 0.18, 0.36].map((delay, i) => (
                  <span key={i} className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: i % 2 === 0 ? "#667eea" : "#764ba2", animationDelay: `${delay}s` }} />
                ))}
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Suggested Questions */}
        <AnimatePresence>
          {messages.length <= 1 && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="px-6 py-3 flex flex-wrap gap-2">
              {SUGGESTED.map((q, i) => (
                <motion.button key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }}
                  onClick={() => handleSend(q)}
                  className="text-xs px-3.5 py-2 rounded-full text-white/65 hover:text-white transition-all"
                  style={{ background: "rgba(102,126,234,0.12)", border: "1px solid rgba(102,126,234,0.25)" }}>
                  {q}
                </motion.button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input Bar */}
        <div className="px-4 py-4 shrink-0" style={{ background: "rgba(5,5,8,0.75)", backdropFilter: "blur(20px)", borderTop: "1px solid rgba(255,255,255,0.08)" }}>
          <div className="max-w-3xl mx-auto flex items-center gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
              placeholder="Ask about space policies, ISRO, IN-SPACe..."
              disabled={isLoading}
              className="flex-1 text-sm py-3 px-4 rounded-xl text-white/90 placeholder-white/25 outline-none transition-all disabled:opacity-50"
              style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", caretColor: "#667eea" }}
            />
            <button onClick={toggleVoice}
              className="p-3 rounded-xl transition-all"
              style={{
                background: isListening ? "rgba(239,68,68,0.2)" : "rgba(255,255,255,0.06)",
                border: `1px solid ${isListening ? "rgba(239,68,68,0.4)" : "rgba(255,255,255,0.1)"}`,
                color: isListening ? "#f87171" : "rgba(255,255,255,0.45)",
              }}>
              {isListening ? <MicOff size={17} /> : <Mic size={17} />}
            </button>
            <button onClick={() => handleSend()} disabled={!input.trim() || isLoading}
              className="p-3 rounded-xl text-white transition-all disabled:opacity-35"
              style={{ background: "linear-gradient(135deg,#667eea,#764ba2)" }}>
              <Send size={17} />
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
