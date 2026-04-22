// ── TF-IDF based vector store — NO API KEY needed for indexing ──────────────

interface StoredDoc {
  pageContent: string;
  metadata: Record<string, string>;
  tfidf: Map<string, number>;
}

// ── Global In-Memory Store ─────────────────────────────────────────────────
const g = globalThis as unknown as {
  _novaXDocs: StoredDoc[] | undefined;
  _novaXVocab: Map<string, number> | undefined;
};

const getDocs = (): StoredDoc[] => {
  if (!g._novaXDocs) g._novaXDocs = [];
  return g._novaXDocs;
};

// ── Text Processing ────────────────────────────────────────────────────────
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2);
}

function computeTF(tokens: string[]): Map<string, number> {
  const tf = new Map<string, number>();
  for (const t of tokens) {
    tf.set(t, (tf.get(t) || 0) + 1);
  }
  // normalize
  for (const [k, v] of tf) {
    tf.set(k, v / tokens.length);
  }
  return tf;
}

function cosineSim(a: Map<string, number>, b: Map<string, number>): number {
  let dot = 0, ma = 0, mb = 0;
  for (const [k, v] of a) {
    dot += v * (b.get(k) || 0);
    ma += v * v;
  }
  for (const [, v] of b) {
    mb += v * v;
  }
  return ma && mb ? dot / (Math.sqrt(ma) * Math.sqrt(mb)) : 0;
}

// ── Public API ─────────────────────────────────────────────────────────────
export const clearStore = () => {
  g._novaXDocs = [];
};

export const addDocuments = async (
  docs: { pageContent: string; metadata: Record<string, string> }[]
) => {
  const store = getDocs();
  for (const doc of docs) {
    const tokens = tokenize(doc.pageContent);
    const tfidf = computeTF(tokens);
    store.push({ ...doc, tfidf });
  }
};

export const similaritySearch = async (
  query: string,
  k = 3
): Promise<{ pageContent: string; metadata: Record<string, string>; score: number }[]> => {
  const store = getDocs();
  if (!store.length) return [];

  const queryTF = computeTF(tokenize(query));

  return store
    .map((doc) => ({
      pageContent: doc.pageContent,
      metadata: doc.metadata,
      score: cosineSim(queryTF, doc.tfidf),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
};

export const docCount = () => getDocs().length;
