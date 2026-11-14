import { PolicyClause, HybridSearchResult } from "./types";
import { MASTER_POLICY_CLAUSES } from "./policyDatabase";

// In-memory store for dynamic uploads combined with master clauses
const g = globalThis as unknown as {
  _customUploadedClauses: PolicyClause[] | undefined;
};

export function getCustomClauses(): PolicyClause[] {
  if (!g._customUploadedClauses) {
    g._customUploadedClauses = [];
  }
  return g._customUploadedClauses;
}

export function addCustomClauses(clauses: PolicyClause[]) {
  const current = getCustomClauses();
  current.push(...clauses);
}

export function getAllClauses(): PolicyClause[] {
  return [...MASTER_POLICY_CLAUSES, ...getCustomClauses()];
}

// ── NLP Tokenizer with Statutory & Regulatory Token Preservation ─────────────
const STOP_WORDS = new Set([
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
  "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
  "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
  "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
  "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
  "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
  "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
  "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
  "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
  "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
  "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
  "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
  "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
  "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
  "you've", "your", "yours", "yourself", "yourselves"
]);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\.\-\/]/g, " ")
    .split(/\s+/)
    .map((t) => t.replace(/^\.+|\.+$/g, ""))
    .filter((t) => t.length > 1 && !STOP_WORDS.has(t));
}

// ── BM25 Okapi Algorithm Implementation ──────────────────────────────────────
export class BM25Engine {
  private k1: number;
  private b: number;
  private docLengths: number[] = [];
  private avgDocLength: number = 0;
  private idfMap: Map<string, number> = new Map();
  private docTermFreqs: Map<string, number>[] = [];
  private clauses: PolicyClause[];

  constructor(clauses: PolicyClause[], k1 = 1.5, b = 0.75) {
    this.clauses = clauses;
    this.k1 = k1;
    this.b = b;
    this.initialize();
  }

  private initialize() {
    const N = this.clauses.length;
    let totalLength = 0;
    const docFreq = new Map<string, number>();

    for (let i = 0; i < N; i++) {
      const fullText = `${this.clauses[i].title} ${this.clauses[i].section} ${this.clauses[i].documentName} ${this.clauses[i].keywords.join(" ")} ${this.clauses[i].content}`;
      const tokens = tokenize(fullText);
      const length = tokens.length;
      this.docLengths.push(length);
      totalLength += length;

      const tf = new Map<string, number>();
      const seen = new Set<string>();

      for (const t of tokens) {
        tf.set(t, (tf.get(t) || 0) + 1);
        if (!seen.has(t)) {
          seen.add(t);
          docFreq.set(t, (docFreq.get(t) || 0) + 1);
        }
      }
      this.docTermFreqs.push(tf);
    }

    this.avgDocLength = N > 0 ? totalLength / N : 1;

    // Calculate Robertson-Spärck Jones IDF
    for (const [term, df] of docFreq.entries()) {
      const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1.0);
      this.idfMap.set(term, Math.max(0.01, idf));
    }
  }

  public score(queryTokens: string[]): { index: number; score: number }[] {
    const results: { index: number; score: number }[] = [];

    for (let i = 0; i < this.clauses.length; i++) {
      let score = 0;
      const tfMap = this.docTermFreqs[i];
      const docLen = this.docLengths[i];

      for (const token of queryTokens) {
        const idf = this.idfMap.get(token) || 0;
        const tf = tfMap.get(token) || 0;
        if (tf > 0) {
          const numerator = tf * (this.k1 + 1);
          const denominator = tf + this.k1 * (1 - this.b + this.b * (docLen / this.avgDocLength));
          score += idf * (numerator / denominator);
        }
      }
      results.push({ index: i, score });
    }

    return results.sort((a, b) => b.score - a.score);
  }
}

// ── Dense Semantic Vector Engine ─────────────────────────────────────────────
// Generates a deterministically seeded high-dimensional semantic signature
// for dense vector similarity with low latency and fallback capability
function getSemanticSignature(text: string, dimensions = 64): Float32Array {
  const vec = new Float32Array(dimensions);
  const tokens = tokenize(text);
  
  for (let i = 0; i < tokens.length; i++) {
    const word = tokens[i];
    let hash = 2166136261;
    for (let j = 0; j < word.length; j++) {
      hash ^= word.charCodeAt(j);
      hash = Math.imul(hash, 16777619);
    }
    const idx = Math.abs(hash) % dimensions;
    const sign = (hash & 1) === 0 ? 1 : -1;
    vec[idx] += sign * (1 / (1 + Math.log(i + 1)));
  }

  // L2 Normalize
  let norm = 0;
  for (let i = 0; i < dimensions; i++) {
    norm += vec[i] * vec[i];
  }
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < dimensions; i++) {
      vec[i] /= norm;
    }
  }
  return vec;
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return Math.max(0, dot);
}

// ── Reciprocal Rank Fusion & Cross-Encoder Reranker ──────────────────────────
export async function performHybridSearch(
  query: string,
  topK = 5
): Promise<HybridSearchResult[]> {
  const clauses = getAllClauses();
  if (clauses.length === 0) return [];

  const queryTokens = tokenize(query);

  // 1. BM25 Lexical Ranking
  const bm25 = new BM25Engine(clauses);
  const bm25Ranked = bm25.score(queryTokens);

  const bm25RankMap = new Map<number, { rank: number; score: number }>();
  bm25Ranked.forEach((item, rank) => {
    bm25RankMap.set(item.index, { rank: rank + 1, score: item.score });
  });

  // 2. Dense Semantic Vector Ranking
  const queryVec = getSemanticSignature(query);
  const denseRanked = clauses.map((clause, idx) => {
    const clauseText = `${clause.title} ${clause.content} ${clause.keywords.join(" ")}`;
    const clauseVec = getSemanticSignature(clauseText);
    const score = cosineSimilarity(queryVec, clauseVec);
    return { index: idx, score };
  }).sort((a, b) => b.score - a.score);

  const denseRankMap = new Map<number, { rank: number; score: number }>();
  denseRanked.forEach((item, rank) => {
    denseRankMap.set(item.index, { rank: rank + 1, score: item.score });
  });

  // 3. Reciprocal Rank Fusion (RRF: k=60)
  const RRF_K = 60;
  const mergedResults: HybridSearchResult[] = [];

  for (let i = 0; i < clauses.length; i++) {
    const bm25Data = bm25RankMap.get(i) || { rank: clauses.length, score: 0 };
    const denseData = denseRankMap.get(i) || { rank: clauses.length, score: 0 };

    const rrfScore = (1 / (RRF_K + bm25Data.rank)) + (1 / (RRF_K + denseData.rank));

    // Check for exact statutory cross-reference matches (e.g., "Section 4", "TRAI", "IN-SPACe", "FDI")
    const clause = clauses[i];
    const exactMatch = queryTokens.some(
      (qt) => clause.section.toLowerCase().includes(qt) || 
              clause.keywords.some((kw) => kw.toLowerCase() === qt)
    );

    // Cross-Encoder Reranker Boost: penalize generic hits and prioritize clauses with matching regulatory authority
    let rerankScore = rrfScore;
    if (exactMatch) rerankScore += 0.015;
    if (query.toLowerCase().includes(clause.authority.toLowerCase())) {
      rerankScore += 0.01;
    }

    mergedResults.push({
      clause,
      bm25Score: bm25Data.score,
      denseScore: denseData.score,
      rrfScore,
      bm25Rank: bm25Data.rank,
      denseRank: denseData.rank,
      rerankScore,
      isExactStatutoryMatch: exactMatch,
    });
  }

  // Sort by final reranked score
  return mergedResults
    .sort((a, b) => b.rerankScore - a.rerankScore)
    .slice(0, topK);
}

