import { NextResponse } from "next/server";
import { clearStore, addDocuments } from "@/lib/vectorStore";
import path from "path";
import fs from "fs";
import { extractText, getDocumentProxy } from "unpdf";

export const maxDuration = 60;

const DOCS_DIR = path.join(process.cwd(), "public", "docs");

const POLICY_DOCS = [
  { file: "IndianSpacePolicy2023.pdf",     name: "Indian Space Policy 2023" },
  { file: "TRAISatelliteSpectrum2025.pdf", name: "TRAI Satellite Spectrum Recommendations 2025" },
  { file: "INSPACENorms2024.pdf",          name: "IN-SPACe NGP Authorization Guidelines 2024" },
  { file: "ISROAnnualReport2024-25.pdf",   name: "ISRO Annual Report 2024-25" },
  { file: "NSILAnnualReport2023-24.pdf",   name: "NSIL Annual Report 2023-24" },
];

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

export async function GET() {
  // No API key needed for indexing — TF-IDF is fully local!
  clearStore();
  const results = [];

  for (const docDef of POLICY_DOCS) {
    const filePath = path.join(DOCS_DIR, docDef.file);

    if (!fs.existsSync(filePath)) {
      console.log(`[seed] NOT FOUND: ${filePath}`);
      results.push({ name: docDef.name, status: "error", chunks: 0 });
      continue;
    }

    try {
      const fileSizeMB = fs.statSync(filePath).size / (1024 * 1024);
      console.log(`[seed] Processing: ${docDef.file} (${fileSizeMB.toFixed(1)}MB)`);

      const buffer = new Uint8Array(fs.readFileSync(filePath));
      const pdf = await getDocumentProxy(buffer);
      const { text } = await extractText(pdf, { mergePages: true });

      if (!text?.trim()) {
        console.log(`[seed] EMPTY: ${docDef.file}`);
        results.push({ name: docDef.name, status: "error", chunks: 0 });
        continue;
      }

      // Limit to 60k chars for large PDFs
      const limitedText = text.trim().slice(0, 60000);
      const chunks = chunkText(limitedText);

      await addDocuments(
        chunks.map((c) => ({ pageContent: c, metadata: { source: docDef.name } }))
      );

      console.log(`[seed] ✓ ${docDef.name}: ${chunks.length} chunks`);
      results.push({ name: docDef.name, status: "indexed", chunks: chunks.length });
    } catch (err: any) {
      console.error(`[seed] ERROR: ${docDef.file}`, err?.message);
      results.push({ name: docDef.name, status: "error", chunks: 0 });
    }
  }

  console.log("[seed] Complete:", results);
  return NextResponse.json({ docs: results });
}
