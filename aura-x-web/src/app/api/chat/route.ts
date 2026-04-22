import { NextRequest, NextResponse } from "next/server";
import { similaritySearch } from "@/lib/vectorStore";

export const maxDuration = 60;

export async function POST(req: NextRequest) {
  try {
    const { query } = await req.json();
    if (!query) return NextResponse.json({ error: "No query provided" }, { status: 400 });

    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) return NextResponse.json({ error: "OPENAI_API_KEY not set." }, { status: 500 });

    const results = await similaritySearch(query, 3);

    if (!results.length) {
      return NextResponse.json({
        answer: "No documents are currently loaded. Please wait for the knowledge base to initialize or upload a document.",
        sources: [],
        score: 0,
      });
    }

    const context = results.map((r) => r.pageContent).join("\n\n");
    const sources = results.map((r) => r.metadata.source);
    const avgScore = results.reduce((s, r) => s + r.score, 0) / results.length;

    const prompt = `You are Nova-X, an intelligent assistant specializing in Indian space policy and regulations.
Answer the question based strictly on the context provided below.
If the context does not contain the answer, say "I cannot find the answer to this in the available documents."
Be concise, accurate, and professional.

Context:
${context}

Question: ${query}

Answer:`;

    // Direct OpenAI REST API call — no SDK issues
    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        messages: [{ role: "user", content: prompt }],
        temperature: 0.3,
        max_tokens: 500,
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      console.error("OpenAI error:", JSON.stringify(data));
      return NextResponse.json(
        { error: data.error?.message || "OpenAI API error" },
        { status: res.status }
      );
    }

    const answer = data.choices?.[0]?.message?.content?.trim() || "Could not generate answer.";

    return NextResponse.json({ answer, sources, score: avgScore });
  } catch (error: any) {
    console.error("Chat error:", error);
    return NextResponse.json(
      { error: error.message || "Failed to generate answer" },
      { status: 500 }
    );
  }
}
