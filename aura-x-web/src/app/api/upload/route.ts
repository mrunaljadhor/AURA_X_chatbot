import { NextRequest, NextResponse } from "next/server";
import { addCustomClauses } from "@/lib/hybridEngine";
import { PolicyClause } from "@/lib/types";
import { extractText, getDocumentProxy } from "unpdf";

export const maxDuration = 60;

function chunkTextHierarchically(text: string, docName: string): PolicyClause[] {
  const clauses: PolicyClause[] = [];
  const paragraphs = text.split(/\n\s*\n/).filter((p) => p.trim().length > 60);

  let currentSection = "General Provision";
  let sectionIdx = 1;

  for (let i = 0; i < paragraphs.length; i++) {
    const p = paragraphs[i].trim();

    // Check for section or article patterns
    const match = p.match(/(?:Section|Article|Clause|Recommendation|Chapter)\s+([0-9A-Za-z\.\-]+)/i);
    if (match) {
      currentSection = match[0];
    } else {
      currentSection = `Clause ${sectionIdx++}`;
    }

    const title = p.slice(0, 60).replace(/[\r\n]+/g, " ") + "...";
    clauses.push({
      id: `custom-${Date.now()}-${i}`,
      documentName: docName,
      authority: "IN-SPACe",
      section: currentSection,
      title,
      content: p,
      category: "AUTHORIZATION",
      keywords: ["custom", "uploaded", ...p.toLowerCase().split(/\W+/).filter(w => w.length > 4).slice(0, 5)],
      effectiveYear: new Date().getFullYear(),
    });
  }

  return clauses;
}

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File | null;

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    const buffer = new Uint8Array(await file.arrayBuffer());
    let text = "";

    if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
      const pdf = await getDocumentProxy(buffer);
      const result = await extractText(pdf, { mergePages: true });
      text = result.text || "";
    } else if (file.type.startsWith("text/") || file.name.endsWith(".txt")) {
      text = new TextDecoder().decode(buffer);
    } else {
      return NextResponse.json(
        { error: "Unsupported file type. Please upload a PDF or TXT statutory document." },
        { status: 400 }
      );
    }

    if (!text.trim()) {
      return NextResponse.json({ error: "Extracted document content is empty." }, { status: 400 });
    }

    const clauses = chunkTextHierarchically(text, file.name);
    addCustomClauses(clauses);

    return NextResponse.json({
      success: true,
      message: `Successfully indexed ${clauses.length} statutory clauses from "${file.name}".`,
      clausesCount: clauses.length,
      sampleSections: clauses.slice(0, 3).map((c) => c.section),
    });
  } catch (error: any) {
    console.error("Upload error:", error);
    return NextResponse.json({ error: error.message || "Failed to process statutory document." }, { status: 500 });
  }
}
