export type RegulatoryAuthority = "IN-SPACe" | "ISRO" | "TRAI" | "DoS" | "NSIL" | "Union Cabinet";

export type PolicyCategory = 
  | "AUTHORIZATION"
  | "SPECTRUM_ALLOCATION"
  | "LIABILITY_AND_INSURANCE"
  | "FDI_AND_COMMERCE"
  | "COMMERCIAL"
  | "INFRASTRUCTURE_ACCESS"
  | "RESEARCH_AND_DEVELOPMENT"
  | "EXPORT_CONTROL";

export interface PolicyClause {
  id: string;
  documentName: string;
  authority: RegulatoryAuthority;
  section: string;
  title: string;
  content: string;
  parentSection?: string;
  category: PolicyCategory;
  keywords: string[];
  effectiveYear: number;
}

export interface HybridSearchResult {
  clause: PolicyClause;
  bm25Score: number;
  denseScore: number;
  rrfScore: number;
  bm25Rank: number;
  denseRank: number;
  rerankScore: number;
  isExactStatutoryMatch: boolean;
}

export type ClaimVerificationStatus = "ENTAILED" | "SUPPORTED" | "CONDITIONAL" | "AMBIGUOUS";

export interface ClaimAttribution {
  id: string;
  claim: string;
  sourceDocument: string;
  section: string;
  verbatimExcerpt: string;
  confidence: number;
  status: ClaimVerificationStatus;
  regulatoryAuthority: RegulatoryAuthority;
}

export interface ReasoningStep {
  id: string;
  step: "QUERY_DECOMPOSITION" | "HYBRID_RETRIEVAL" | "RELEVANCE_GRADING" | "NLI_CLAIM_VERIFICATION" | "COMPLIANCE_SYNTHESIS";
  title: string;
  detail: string;
  status: "pending" | "running" | "completed";
  durationMs?: number;
  dataSummary?: Record<string, any>;
}

export type VerdictStatus = 
  | "COMPLIANT" 
  | "REQUIRES_INSPACE_AUTHORIZATION" 
  | "RESTRICTED_OR_PROHIBITED" 
  | "CONDITIONAL_APPROVAL" 
  | "INFORMATIONAL";

export interface ComplianceVerdict {
  status: VerdictStatus;
  primaryAuthority: RegulatoryAuthority;
  governingSection: string;
  executiveSummary: string;
  mandatoryPrerequisites: string[];
  riskRating: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
}

export interface ChatApiResponse {
  answer: string;
  verdict: ComplianceVerdict;
  claims: ClaimAttribution[];
  retrievalResults: HybridSearchResult[];
  reasoningSteps: ReasoningStep[];
  diagnostics: {
    totalClausesIndexed: number;
    bm25ClausesScored: number;
    denseClausesScored: number;
    rrfTopK: number;
    latencyMs: number;
    model: string;
    temperature: number;
  };
}
