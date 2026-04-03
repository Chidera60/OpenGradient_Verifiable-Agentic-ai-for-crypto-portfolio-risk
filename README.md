# Verifiable AI Crypto Risk Advisor

A TypeScript demo agent that analyzes crypto portfolio volatility using **verifiable on-chain inference** through the [OpenGradient](https://opengradient.ai) blockchain stack.

## How It Works

Each analysis request runs a structured 4-step pipeline:

```
Portfolio Config
      │
      ▼
① Exposure Analysis          ← portfolio weights, beta, sector clustering
      │
      ▼
② ONNX Inference             ← crypto_vol_transformer_v2.onnx inside TEE
      │
      ▼
③ TEE-Secured Reasoning      ← SGX enclave, DCAP attestation, ReAct loop
      │
      ▼
④ Risk Recommendations       ← anchored on-chain, verifiable by anyone
```

Every output is:
- **Tamper-evident** — ONNX inference result is SHA-256 hashed
- **TEE-attested** — reasoning runs inside SGX enclave with DCAP quote
- **On-chain anchored** — block hash + tx hash on OpenGradient mainnet
- **Reproducible** — session ID lets third parties verify results

## Tech Stack

| Layer | Technology |
|---|---|
| Language | TypeScript 5.4 |
| Verifiable Inference | OpenGradient SDK |
| ONNX Runtime | `onnxruntime-node` |
| Agent Pattern | ReAct (Reasoning + Acting) |
| TEE | Intel SGX v2 / DCAP |
| Blockchain | OpenGradient Mainnet (EVM-compatible) |
| HTTP Server | Express 4 |

## Project Structure

```
src/
├── index.ts                      # CLI entry point
├── server.ts                     # Express HTTP server
├── types/
│   └── index.ts                  # All TypeScript types
├── agent/
│   └── react-agent.ts            # ReAct pipeline orchestrator
├── inference/
│   └── opengradient-client.ts    # OpenGradient SDK wrapper
├── models/
│   └── onnx-volatility-model.ts  # ONNX model runner + feature engineering
└── utils/
    └── asset-registry.ts         # Asset metadata + input validation
```

## Quick Start

### 1. Install dependencies

```bash
npm install
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your OpenGradient private key and RPC URL
```

### 3. Run CLI demo

```bash
npm run dev
```

### 4. Start HTTP server

```bash
npm run serve
```

## API Reference

### `POST /analyze`

Run a full verifiable risk analysis.

**Request body:**
```json
{
  "assets": [
    { "symbol": "BTC",  "allocationPct": 40 },
    { "symbol": "ETH",  "allocationPct": 30 },
    { "symbol": "SOL",  "allocationPct": 20 },
    { "symbol": "ARB",  "allocationPct": 10 }
  ],
  "timeframe": "30d",
  "riskTolerance": 6,
  "walletAddress": "0x..."
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "sessionId": "og-a1b2c3d4e5f6g7h8",
    "metrics": {
      "aggregateRiskScore": 42,
      "riskLevel": "MEDIUM",
      "portfolioVol": 0.84,
      "portfolioBeta": 1.26,
      "var95": 0.0867,
      "sharpeEstimate": 0.312
    },
    "recommendations": [...],
    "verifiableProof": {
      "onnxHash": "0xabc123...",
      "teeAttestation": { "nodeId": "...", "attestationKey": "0x..." },
      "blockAnchor": { "blockNumber": 19852341, "txHash": "0x..." }
    }
  }
}
```

### `GET /assets`
List all supported asset symbols and metadata.

### `GET /health`
System health check.

### `GET /verify/:sessionId`
Fetch on-chain verification link for a session.

## Supported Assets

BTC, ETH, SOL, BNB, ARB, AVAX, MATIC, LINK, OP, UNI, AAVE, INJ, ATOM, DOT, USDC

## Using a Real ONNX Model

Place your ONNX model file at `models/crypto_vol_transformer_v2.onnx`.

Expected input tensors:
- `portfolio_weights` — `[1, n_assets]` float32
- `market_features` — `[n_assets, 32]` float32  
- `timeframe_index` — `[1]` int64

Expected output tensors:
- `predicted_vol` — `[1]` float32 (annualized volatility)
- `var_95` — `[1]` float32 (1-day 95% VaR)
- `sharpe` — `[1]` float32 (Sharpe estimate)

## OpenGradient SDK Integration

In production, replace the simulation in `opengradient-client.ts` with the real SDK:

```typescript
import { OpenGradient } from '@opengradient/sdk';

const og = new OpenGradient({
  privateKey: process.env.OPENGRADIENT_PRIVATE_KEY,
  rpcUrl: process.env.OPENGRADIENT_RPC_URL,
});

const result = await og.runOnnxModel({
  modelId: 'crypto_vol_transformer_v2',
  inputs: { portfolio_weights, market_features },
  teeEnabled: true,
});
```

## Architecture Notes

### ReAct Agent Loop

The agent follows the **Reason → Act → Observe** pattern across 4 tools:
1. `portfolio_exposure_analyzer` — computes weighted vol, beta, sector maps
2. `onnx_volatility_model` — runs ONNX inference in TEE
3. `tee_reasoning_engine` — scores risk, seals reasoning trace
4. `on_chain_anchor` — publishes proof to OpenGradient chain

### TEE Attestation

The SGX enclave produces a DCAP quote containing:
- Measurement of the inference code and weights
- Hash of the input portfolio
- Signature by the enclave's private key

This quote is verifiable by any Intel-compatible attestation service.

## License

MIT
