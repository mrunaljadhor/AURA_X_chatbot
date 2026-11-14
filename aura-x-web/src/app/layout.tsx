import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NOVA-X | Autonomous Space Policy Compliance & Explainable RAG Engine",
  description: "An autonomous, serverless RAG framework for parsing complex space regulations, delivering Explainable AI (XAI) source attribution and statutory compliance auditing.",
  keywords: ["RAG", "Space Policy", "Explainable AI", "IN-SPACe", "ISRO", "LangChain", "Next.js", "AI Compliance"],
  authors: [{ name: "NOVA-X Systems" }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased dark">
      <body className="min-h-full flex flex-col bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
