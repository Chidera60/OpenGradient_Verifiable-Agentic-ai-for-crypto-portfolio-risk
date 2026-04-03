// ============================================================
// server.ts — Express REST API server
// ============================================================

import 'dotenv/config';
import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import * as path from 'path';
import { CryptoRiskAdvisorAgent } from './agent/react-agent';
import { validatePortfolioInput, buildPortfolioConfig } from './utils/asset-registry';
import type { AnalyzeRequest, AnalyzeResponse } from './types';

const app = express();
const PORT = parseInt(process.env.PORT ?? '3000', 10);

// ── Middleware ────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, '../public')));

// Request logger
app.use((req: Request, _res: Response, next: NextFunction) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// ── Singleton agent ───────────────────────────────────────────

const agent = new CryptoRiskAdvisorAgent();

// ── Routes ────────────────────────────────────────────────────

/**
 * GET /health
 * Health check + system status
 */
app.get('/health', (_req: Request, res: Response) => {
  res.json({
    status: 'ok',
    service: 'Verifiable AI Crypto Risk Advisor',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    features: {
      onnxInference: true,
      teeAttestation: true,
      onChainAnchoring: true,
      reactAgent: true,
    },
  });
});

/**
 * GET /assets
 * List supported assets and their metadata
 */
app.get('/assets', (_req: Request, res: Response) => {
  const { ASSET_REGISTRY } = require('./utils/asset-registry');
  res.json({
    assets: Object.entries(ASSET_REGISTRY).map(([symbol, meta]) => ({
      symbol,
      ...(meta as object),
    })),
  });
});

/**
 * POST /analyze
 * Main analysis endpoint — runs the full ReAct agent pipeline
 *
 * Body: AnalyzeRequest
 */
app.post('/analyze', async (req: Request<{}, AnalyzeResponse, AnalyzeRequest>, res: Response) => {
  const input = {
    assets: req.body.assets ?? [],
    timeframe: req.body.timeframe ?? '30d',
    riskTolerance: req.body.riskTolerance ?? 5,
    walletAddress: req.body.walletAddress,
  };

  // Validate
  const validation = validatePortfolioInput(input);
  if (!validation.valid) {
    res.status(400).json({ success: false, error: validation.errors.join(' | ') });
    return;
  }

  try {
    const portfolioConfig = buildPortfolioConfig(input);
    const result = await agent.analyze(portfolioConfig);

    res.json({ success: true, data: result });
  } catch (err) {
    console.error('[/analyze] Error:', err);
    res.status(500).json({
      success: false,
      error: err instanceof Error ? err.message : 'Internal server error',
    });
  }
});

/**
 * GET /verify/:sessionId
 * Fetch verifiable proof for a completed session
 * In production this would query the OpenGradient explorer API
 */
app.get('/verify/:sessionId', (req: Request, res: Response) => {
  res.json({
    sessionId: req.params.sessionId,
    verifyUrl: `https://explorer.opengradient.ai/sessions/${req.params.sessionId}`,
    message: 'Visit the OpenGradient explorer to verify this session on-chain.',
  });
});

// ── SPA fallback ──────────────────────────────────────────────
app.get('*', (_req: Request, res: Response) => {
  res.sendFile(path.join(__dirname, '../public/index.html'));
});

// ── Error handler ─────────────────────────────────────────────
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  console.error('[ERROR]', err.message);
  res.status(500).json({ success: false, error: err.message });
});

// ── Start ─────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🟢 Verifiable AI Crypto Risk Advisor`);
  console.log(`   Server running at http://localhost:${PORT}`);
  console.log(`   POST /analyze  — run risk analysis`);
  console.log(`   GET  /assets   — list supported assets`);
  console.log(`   GET  /health   — system status\n`);
});

export default app;
