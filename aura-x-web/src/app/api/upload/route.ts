import { NextRequest, NextResponse } from "next/server";
import { addDocuments } from "@/lib/vectorStore";
import { extractText, getDocumentProxy } from "unpdf";

export const maxDuration = 60;

function chunkText(text: string, size = 800, overlap = 150): string[] {
  const chunks: string[] = [];
  let i = 0;
  while (i < text.length) {
    const chunk = text.slice(i, i + size).trim();
    if (chunk.length > 60) chunks.push(chunk);
    i += size - overlap;
  }
  return chunks;
}

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File | null;

    if (!file) return NextResponse.json({ error: "No file provided" }, { status: 400 });
    if (!process.env.GOOGLE_API_KEY) return NextResponse.json({ error: "GOOGLE_API_KEY not set." }, { status: 500 });

    const buffer = new Uint8Array(await file.arrayBuffer());
    let text = "";

    if (file.type === "application/pdf") {
      const pdf = await getDocumentProxy(buffer);
      const result = await extractText(pdf, { mergePages: true });
      text = result.text;
    } else if (file.type === "text/plain") {
      text = new TextDecoder().decode(buffer);
    } else {
      return NextResponse.json({ error: "Unsupported file type. Please upload PDF or TXT." }, { status: 400 });
    }

    if (!text.trim()) return NextResponse.json({ error: "No readable text found in file." }, { status: 400 });

    const chunks = chunkText(text);
    await addDocuments(chunks.map((c) => ({ pageContent: c, metadata: { source: file.name } })));

    return NextResponse.json({ success: true, message: `Indexed "${file.name}"`, chunks: chunks.length });
  } catch (error: any) {
    return NextResponse.json({ error: error.message || "Upload failed" }, { status: 500 });
  }
}
