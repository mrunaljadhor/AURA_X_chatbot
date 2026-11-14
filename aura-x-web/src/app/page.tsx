"use client";

import { useState, useRef, useEffect } from "react";
import {
  UploadCloud, FileText, Trash2, Send, Loader2, Bot, User,
  CheckCircle2, Copy, Mic, MicOff, Download, ChevronDown,
  ChevronUp, ShieldCheck, AlertTriangle, Scale, Cpu, Sparkles,
  ExternalLink, Layers, Terminal, BookMarked, Globe, Check
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { ChatApiResponse, ClaimAttribution, HybridSearchResult, ReasoningStep } from "@/lib/types";

// ── Starfield Canvas Background ──────────────────────────────────────────────
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
      stars = Array.from({ length: 180 }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        r: Math.random() * 1.3 + 0.3,
        alpha: Math.random() * 0.8 + 0.2,
        speed: (Math.random() * 0.003 + 0.001) * (Math.random() > 0.5 ? 1 : -1),
        dx: (Math.random() - 0.5) * 0.06,
        dy: (Math.random() - 0.5) * 0.06,
      }));
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const s of stars) {
        s.alpha += s.speed;
        if (s.alpha > 0.95 || s.alpha < 0.15) s.speed *= -1;
        s.x = (s.x + s.dx + canvas.width) % canvas.width;
        s.y = (s.y + s.dy + canvas.height) % canvas.height;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(186, 230, 253, ${Math.max(0, Math.min(1, s.alpha))})`;
        ctx.fill();
      }

      raf = requestAnimationFrame(draw);
    };

    resize();
    init();
    draw();
    window.addEventListener("resize", () => { resize(); init(); });
    return () => { cancelAnimationFrame(raf); };
  }, []);

  return <canvas ref={ref} className="fixed inset-0 w-full h-full z-0 pointer-events-none opacity-50" />;
}

// ── Types ──────────────────────────────────────────────────────────────────
interface MessageItem {
  id: string;
  role: "user" | "assistant";
  content: string;
  responsePayload?: ChatApiResponse;
  timestamp: string;
}

const SAMPLE_QUESTIONS = [
  "What are the FDI thresholds for satellite manufacturing & launch vehicles under the 2023 policy?",
  "What is IN-SPACe's single-window authorization mandate for private spaceports?",
  "What are the mandatory third-party liability insurance caps (MPL) for private launches?",
  "How does TRAI handle satcom spectrum allocation under the Telecommunications Act 2023?",
  "What is ISRO's designated role following the commercial transition to NSIL?",
];

export default function Home() {
  const [messages, setMessages] = useState<MessageItem[]>([
    {
      id: "welcome-1",
      role: "assistant",
      content: "Welcome to **NOVA-X**, an autonomous compliance intelligence platform for the Indian Space Regulatory Framework.\n\nI parse and cross-reference statutory instruments including the **Indian Space Policy 2023**, **IN-SPACe Norms 2024**, and **TRAI Satellite Spectrum Recommendations 2025** using hybrid search (BM25 + Dense RRF) and verifiable Explainable AI (XAI) claim attribution.\n\nSelect a preset regulatory scenario below or type your inquiry to initiate an audit.",
      timestamp: "Just now",
    },
  ]);
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isAuditorMode, setIsAuditorMode] = useState(true);
  const [activeCitation, setActiveCitation] = useState<ClaimAttribution | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Speech to text initialization
  useEffect(() => {
    if (typeof window !== "undefined") {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = "en-IN";
        recognition.onresult = (event: any) => {
          const text = event.results[0][0].transcript;
          setQuery(text);
          setIsRecording(false);
        };
        recognition.onerror = () => setIsRecording(false);
        recognition.onend = () => setIsRecording(false);
        recognitionRef.current = recognition;
      }
    }
  }, []);

  const toggleRecording = () => {
    if (!recognitionRef.current) {
      alert("Speech recognition is not supported in this browser. Please use Google Chrome or Microsoft Edge.");
      return;
    }
    if (isRecording) {
      recognitionRef.current.stop();
      setIsRecording(false);
    } else {
      recognitionRef.current.start();
      setIsRecording(true);
    }
  };

  const handleSend = async (textToSend?: string) => {
    const q = (textToSend || query).trim();
    if (!q || isLoading) return;

    const userMsg: MessageItem = {
      id: `user-${Date.now()}`,
      role: "user",
      content: q,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setQuery("");
    setIsLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });

      if (!res.ok) {
        throw new Error(`API responded with status ${res.status}`);
      }

      const data: ChatApiResponse = await res.json();

      const assistantMsg: MessageItem = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: data.answer,
        responsePayload: data,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: "assistant",
          content: `⚠️ Compliance Engine Error: ${err.message || "Failed to process statutory audit."}`,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadStatus("Indexing statutory document...");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus(`✓ Indexed ${data.clausesCount} clauses from "${file.name}"`);
      } else {
        setUploadStatus(`❌ ${data.error || "Upload failed"}`);
      }
    } catch (err: any) {
      setUploadStatus(`❌ Error: ${err.message}`);
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const exportAuditReport = (payload: ChatApiResponse, queryText: string) => {
    const markdown = `# NOVA-X STATUTORY COMPLIANCE AUDIT REPORT
Generated: ${new Date().toISOString()}
Inquiry: ${queryText}

## 1. COMPLIANCE VERDICT
Status: ${payload.verdict.status}
Risk Rating: ${payload.verdict.riskRating}
Primary Governing Authority: ${payload.verdict.primaryAuthority}
Governing Provision: ${payload.verdict.governingSection}

### Summary:
${payload.verdict.executiveSummary}

### Mandatory Prerequisites:
${payload.verdict.mandatoryPrerequisites.map((p) => `- [ ] ${p}`).join("\n")}

## 2. STATUTORY FINDINGS & ANALYSIS
${payload.answer}

## 3. EXPLAINABLE AI (XAI) CLAIM ATTRIBUTION MATRIX
${payload.claims
  .map(
    (c, i) => `### Claim ${i + 1} [${c.status} - ${(c.confidence * 100).toFixed(0)}% Confidence]
- **Assertion:** "${c.claim}"
- **Statutory Instrument:** ${c.sourceDocument} (${c.section})
- **Verbatim Excerpt:** "${c.verbatimExcerpt}"
`
  )
  .join("\n")}

## 4. RETRIEVAL & ML DIAGNOSTICS
- Total Statutory Clauses Scored: ${payload.diagnostics.totalClausesIndexed}
- Hybrid Fusion Strategy: BM25 Okapi + 64-D Dense Vector + Reciprocal Rank Fusion (k=60)
- Execution Latency: ${payload.diagnostics.latencyMs}ms
- Synthesizer: ${payload.diagnostics.model}
`;

    const blob = new Blob([markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `nova-x-compliance-audit-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="relative min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans overflow-hidden">
      <StarCanvas />

      {/* ── Top Navigation Bar ────────────────────────────────────────── */}
      <header className="relative z-10 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-gradient-to-tr from-cyan-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Cpu className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold tracking-wider text-base text-white">NOVA-X</span>
              <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-cyan-950/90 text-cyan-400 border border-cyan-800/60 font-semibold">
                Autonomous Compliance RAG
              </span>
            </div>
            <p className="text-xs text-slate-400 flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Hybrid Search (BM25 + Dense RRF) • Statutory XAI Verification
            </p>
          </div>
        </div>

        {/* Mode Toggle & System Status */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 bg-slate-900/90 border border-slate-800 p-1 rounded-lg">
            <button
              onClick={() => setIsAuditorMode(true)}
              className={`text-xs px-3 py-1.5 rounded-md font-medium transition flex items-center gap-1.5 ${
                isAuditorMode
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              Agentic Auditor Mode
            </button>
            <button
              onClick={() => setIsAuditorMode(false)}
              className={`text-xs px-3 py-1.5 rounded-md font-medium transition flex items-center gap-1.5 ${
                !isAuditorMode
                  ? "bg-slate-800 text-slate-200 border border-slate-700 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              Direct RAG
            </button>
          </div>

          <a
            href="https://github.com/mrunaljadhor/AURA_X_chatbot"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs px-3 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-slate-300 flex items-center gap-1.5 transition"
          >
            <Globe className="w-3.5 h-3.5" />
            GitHub
          </a>
        </div>
      </header>

      {/* ── Main Workspace Layout ─────────────────────────────────────── */}
      <div className="relative z-10 flex-1 flex overflow-hidden">
        {/* Left Sidebar: Knowledge Base & Statutory Instruments */}
        <aside className="w-80 border-r border-slate-800/80 bg-slate-950/60 backdrop-blur-sm p-4 hidden md:flex flex-col gap-4 overflow-y-auto">
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
              <BookMarked className="w-3.5 h-3.5 text-cyan-400" />
              Statutory Knowledge Base
            </h2>
            <div className="space-y-2 text-xs">
              {[
                { name: "Indian Space Policy 2023", authority: "IN-SPACe / DoS", status: "Indexed" },
                { name: "IN-SPACe NGP Guidelines 2024", authority: "IN-SPACe", status: "Indexed" },
                { name: "TRAI Satcom Spectrum 2025", authority: "TRAI", status: "Indexed" },
                { name: "ISRO Strategic Annual Report", authority: "ISRO", status: "Indexed" },
                { name: "NSIL Commercial Contracts", authority: "NSIL", status: "Indexed" },
              ].map((doc, idx) => (
                <div
                  key={idx}
                  className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800/80 flex items-start justify-between gap-2 hover:border-slate-700 transition"
                >
                  <div>
                    <p className="font-medium text-slate-200">{doc.name}</p>
                    <p className="text-[11px] text-slate-400">{doc.authority}</p>
                  </div>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-800/50">
                    {doc.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Upload Custom Statutory Instrument */}
          <div className="p-3 rounded-lg bg-slate-900/40 border border-slate-800/80">
            <label className="text-xs font-semibold text-slate-300 mb-1.5 flex items-center gap-1.5 cursor-pointer">
              <UploadCloud className="w-3.5 h-3.5 text-indigo-400" />
              Ingest Policy PDF / TXT
            </label>
            <input
              type="file"
              accept=".pdf,.txt"
              onChange={handleFileUpload}
              className="text-xs text-slate-400 file:mr-2 file:py-1 file:px-2.5 file:rounded file:border-0 file:text-[11px] file:font-medium file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 cursor-pointer w-full"
            />
            {uploadStatus && (
              <p className="text-[11px] mt-2 text-slate-300 font-mono">{uploadStatus}</p>
            )}
          </div>

          {/* Regulatory Scenario Templates */}
          <div className="flex-1">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
              Regulatory Inquiry Scenarios
            </h3>
            <div className="space-y-1.5">
              {SAMPLE_QUESTIONS.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(q)}
                  className="w-full text-left text-xs p-2 rounded-lg bg-slate-900/30 hover:bg-slate-900 border border-slate-800/50 hover:border-cyan-500/40 text-slate-300 transition line-clamp-2"
                >
                  "{q}"
                </button>
              ))}
            </div>
          </div>

          {/* Engine Specs */}
          <div className="text-[11px] text-slate-400 font-mono p-2.5 rounded bg-slate-950 border border-slate-900">
            <div>Engine: BM25 + Dense RRF</div>
            <div>XAI Metric: Sentence-NLI Entailment</div>
            <div>Architecture: Stateless Serverless</div>
          </div>
        </aside>

        {/* Center Chat Area */}
        <main className="flex-1 flex flex-col justify-between overflow-hidden">
          {/* Messages Stream */}
          <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex gap-3 max-w-4xl mx-auto ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {msg.role === "assistant" && (
                  <div className="h-8 w-8 rounded-lg bg-gradient-to-tr from-cyan-600 to-indigo-600 flex-shrink-0 flex items-center justify-center shadow-md">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                )}

                <div className={`space-y-3 max-w-3xl ${msg.role === "user" ? "items-end" : "items-start"}`}>
                  {/* Bubble Content */}
                  <div
                    className={`rounded-xl p-4 text-sm leading-relaxed border ${
                      msg.role === "user"
                        ? "bg-cyan-600/20 border-cyan-500/40 text-cyan-100 rounded-tr-none ml-auto"
                        : "bg-slate-900/90 border-slate-800 text-slate-200 rounded-tl-none shadow-xl"
                    }`}
                  >
                    {/* Compliance Verdict Banner (If present) */}
                    {msg.responsePayload?.verdict && (
                      <div className="mb-4 p-3.5 rounded-lg border bg-slate-950/90 flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                          <span
                            className={`text-xs font-mono font-bold px-2 py-0.5 rounded border uppercase tracking-wider flex items-center gap-1.5 ${
                              msg.responsePayload.verdict.status === "COMPLIANT"
                                ? "bg-emerald-950/80 border-emerald-600/60 text-emerald-300"
                                : msg.responsePayload.verdict.status === "REQUIRES_INSPACE_AUTHORIZATION"
                                ? "bg-amber-950/80 border-amber-600/60 text-amber-300"
                                : "bg-red-950/80 border-red-600/60 text-red-300"
                            }`}
                          >
                            <Scale className="w-3.5 h-3.5" />
                            {msg.responsePayload.verdict.status.replace(/_/g, " ")}
                          </span>

                          <span className="text-[11px] font-mono text-slate-400">
                            Risk Level:{" "}
                            <strong className="text-slate-200">{msg.responsePayload.verdict.riskRating}</strong>
                          </span>
                        </div>
                        <p className="text-xs text-slate-300">
                          {msg.responsePayload.verdict.executiveSummary}
                        </p>
                      </div>
                    )}

                    {/* Agent Multi-Step Reasoning Accordion */}
                    {msg.responsePayload?.reasoningSteps && (
                      <details className="mb-4 rounded-lg bg-slate-950/60 border border-slate-800/80 p-2.5 text-xs group">
                        <summary className="font-mono text-cyan-400 cursor-pointer flex items-center justify-between select-none">
                          <span className="flex items-center gap-1.5">
                            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
                            Agentic Multi-Step Reasoning Trace ({msg.responsePayload.reasoningSteps.length} steps)
                          </span>
                          <ChevronDown className="w-3.5 h-3.5 text-slate-400 group-open:rotate-180 transition-transform" />
                        </summary>
                        <div className="mt-2.5 pt-2 border-t border-slate-800/60 space-y-2">
                          {msg.responsePayload.reasoningSteps.map((step) => (
                            <div key={step.id} className="font-mono text-[11px] flex items-start gap-2">
                              <span className="text-emerald-400">✓</span>
                              <div>
                                <span className="font-semibold text-slate-200">{step.title}</span>
                                <p className="text-slate-400">{step.detail}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    {/* Text Body */}
                    <div className="whitespace-pre-line prose prose-invert prose-sm max-w-none">
                      {msg.content}
                    </div>

                    {/* Explainable AI (XAI) Atomic Claims Matrix */}
                    {msg.responsePayload?.claims && msg.responsePayload.claims.length > 0 && (
                      <div className="mt-4 pt-3 border-t border-slate-800/80 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                            Explainable AI (XAI) Source Attribution
                          </span>
                          <span className="text-[10px] font-mono text-slate-400">
                            NLI Entailment Verified
                          </span>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {msg.responsePayload.claims.map((claim) => (
                            <button
                              key={claim.id}
                              onClick={() => setActiveCitation(claim)}
                              className="text-left p-2.5 rounded-lg bg-slate-950/80 border border-slate-800 hover:border-cyan-500/50 transition flex flex-col justify-between gap-1 group"
                            >
                              <div className="flex items-center justify-between w-full">
                                <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/60">
                                  {claim.sourceDocument} • {claim.section}
                                </span>
                                <span className="text-[10px] font-mono text-emerald-400 font-semibold">
                                  {(claim.confidence * 100).toFixed(0)}% Match
                                </span>
                              </div>
                              <p className="text-[11px] text-slate-300 line-clamp-2">
                                "{claim.claim}"
                              </p>
                              <span className="text-[10px] text-cyan-400 flex items-center gap-1 group-hover:underline">
                                View verbatim statute <ChevronDown className="w-2.5 h-2.5 rotate-[-90deg]" />
                              </span>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Diagnostic Metrics & Export Footer */}
                    {msg.responsePayload && (
                      <div className="mt-3 pt-2.5 border-t border-slate-800/60 flex items-center justify-between text-[11px] font-mono text-slate-400">
                        <div className="flex items-center gap-3">
                          <span>Latency: {msg.responsePayload.diagnostics.latencyMs}ms</span>
                          <span>Clauses Scored: {msg.responsePayload.diagnostics.totalClausesIndexed}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => copyToClipboard(msg.content, msg.id)}
                            className="hover:text-slate-200 transition flex items-center gap-1"
                          >
                            {copiedId === msg.id ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                            Copy
                          </button>
                          <button
                            onClick={() => exportAuditReport(msg.responsePayload!, "Statutory Policy Query")}
                            className="hover:text-cyan-400 transition flex items-center gap-1"
                          >
                            <Download className="w-3 h-3" />
                            Export Audit Report
                          </button>
                        </div>
                      </div>
                    )}
                  </div>

                  <span className="text-[10px] font-mono text-slate-400 px-1">{msg.timestamp}</span>
                </div>

                {msg.role === "user" && (
                  <div className="h-8 w-8 rounded-lg bg-cyan-600 flex-shrink-0 flex items-center justify-center shadow-md">
                    <User className="w-4 h-4 text-white" />
                  </div>
                )}
              </motion.div>
            ))}

            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex gap-3 max-w-4xl mx-auto items-center text-sm text-cyan-400 font-mono"
              >
                <div className="h-8 w-8 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center">
                  <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
                </div>
                <span>Executing Hybrid Retrieval & NLI Attribution Verification...</span>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Control Console */}
          <div className="p-4 border-t border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
            <div className="max-w-4xl mx-auto flex items-center gap-2">
              <button
                type="button"
                onClick={toggleRecording}
                className={`p-2.5 rounded-lg border transition ${
                  isRecording
                    ? "bg-red-500/20 border-red-500 text-red-400 animate-pulse"
                    : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
                title="Voice Query (Web Speech API)"
              >
                {isRecording ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
              </button>

              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                placeholder="Ask about IN-SPACe authorization, FDI limits, satellite spectrum, or ISRO mandates..."
                className="flex-1 bg-slate-900/90 border border-slate-800 focus:border-cyan-500/60 focus:ring-1 focus:ring-cyan-500/30 rounded-lg px-4 py-2.5 text-sm text-slate-100 placeholder-slate-400 outline-none transition"
              />

              <button
                onClick={() => handleSend()}
                disabled={isLoading || !query.trim()}
                className="px-4 py-2.5 rounded-lg bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white font-medium text-xs flex items-center gap-1.5 shadow-lg shadow-cyan-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition"
              >
                <Send className="w-3.5 h-3.5" />
                Audit
              </button>
            </div>
          </div>
        </main>
      </div>

      {/* ── Verbatim Citation Modal ───────────────────────────────────── */}
      <AnimatePresence>
        {activeCitation && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-slate-900 border border-slate-800 rounded-xl max-w-xl w-full p-5 shadow-2xl relative"
            >
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Scale className="w-4 h-4 text-cyan-400" />
                  <span className="text-sm font-semibold text-slate-100">
                    Statutory Source Citation
                  </span>
                </div>
                <button
                  onClick={() => setActiveCitation(null)}
                  className="text-slate-400 hover:text-slate-200 text-xs px-2 py-1 rounded bg-slate-800"
                >
                  ✕ Close
                </button>
              </div>

              <div className="mt-4 space-y-3">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-cyan-400 font-semibold">
                    {activeCitation.sourceDocument} • {activeCitation.section}
                  </span>
                  <span className="text-emerald-400">
                    Authority: {activeCitation.regulatoryAuthority}
                  </span>
                </div>

                <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 text-xs leading-relaxed text-slate-300 font-serif">
                  "{activeCitation.verbatimExcerpt}"
                </div>

                <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 text-xs text-slate-400">
                  <span className="font-semibold text-slate-200 block mb-1">Synthesized Claim:</span>
                  "{activeCitation.claim}"
                  <div className="mt-2 flex items-center gap-2 font-mono text-[11px] text-emerald-400">
                    <span>NLI Entailment Confidence: {(activeCitation.confidence * 100).toFixed(0)}%</span>
                    <span>• Status: {activeCitation.status}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
