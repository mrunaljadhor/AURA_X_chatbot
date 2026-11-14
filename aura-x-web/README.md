# NOVA-X Web: Autonomous Space Policy Compliance Platform

Production Next.js 16 web application for NOVA-X, deployed on Vercel.

## Core Features
- **Dual-Channel Hybrid Search:** Combines BM25 lexical token scoring with dense semantic projections using Reciprocal Rank Fusion ($k=60$).
- **Explainable AI (XAI) Attribution Inspector:** Clickable citations with sentence-level Natural Language Inference (NLI) confidence badges and verbatim statutory quotes.
- **Agentic Multi-Step Reasoning Trace:** Live inspectable trace of query decomposition, retrieval scoring, context grading, and compliance synthesis.
- **Compliance Verdict Pill:** Instant status categorization (`COMPLIANT`, `REQUIRES IN-SPACE AUTHORIZATION`, `RESTRICTED / PROHIBITED`) with risk rating.
- **Voice UI:** Web Speech API integration with reactive microphone state.
- **Audit Export:** One-click compliance report generation (Markdown / Audit Log).

## Getting Started

```bash
npm install
npm run build
npm run start
```
