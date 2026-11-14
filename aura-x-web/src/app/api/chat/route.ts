import { NextRequest, NextResponse } from "next/server";
import { executeComplianceAudit } from "@/lib/complianceAgent";

export const maxDuration = 60;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const query = body.query?.trim();

    if (!query) {
      return NextResponse.json({ error: "Missing required query string" }, { status: 400 });
    }

    const apiKey = process.env.OPENAI_API_KEY;
    const auditResponse = await executeComplianceAudit(query, apiKey);

    return NextResponse.json(auditResponse);
  } catch (error: any) {
    console.error("NOVA-X compliance execution error:", error);
    return NextResponse.json(
      { error: error?.message || "Internal compliance evaluation error" },
      { status: 500 }
    );
  }
}
