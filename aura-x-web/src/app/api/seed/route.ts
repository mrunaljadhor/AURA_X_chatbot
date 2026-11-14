import { NextResponse } from "next/server";
import { getAllClauses } from "@/lib/hybridEngine";

export const maxDuration = 60;

export async function GET() {
  const clauses = getAllClauses();
  const docCounts: Record<string, number> = {};

  clauses.forEach((c) => {
    docCounts[c.documentName] = (docCounts[c.documentName] || 0) + 1;
  });

  const docs = Object.entries(docCounts).map(([name, chunks]) => ({
    name,
    status: "indexed" as const,
    chunks,
  }));

  return NextResponse.json({
    status: "ok",
    totalClauses: clauses.length,
    docs,
  });
}
