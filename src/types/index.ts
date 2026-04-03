// ============================================================
// types/index.ts — Core domain types for Verifiable AI Risk Advisor
// ============================================================

export interface Asset {
  symbol: string;
  allocationPct: number;   // 0–100
  historicalVol: number;   // annualized decimal
  beta: number;            // vs BTC
  correlation: number;     // vs BTC
  marketCap: string;
  sector: AssetSector;
}

export type AssetSector =
  | 'Store of Value'
  | 'Smart Contract'
  | 'Exchange'
  | 'L2'
  | 'Oracle'
  | 'DeFi'
  | 'Stablecoin';

export interface PortfolioConfig {
  assets: Asset[];
  timeframe: '7d' | '30d' | '90d' | '1y';
  riskTolerance: number;   // 1–10
  walletAddress?: string;
}

// ── ONNX Inference ──────────────────────────────────────────

export interface OnnxModelInput {
  portfolioWeights: Float32Array;    // shape [n_assets]
  marketFeatures: Float32Array;      // shape [n_assets, 32]
  timeframeIndex: number;
}

export interface OnnxModelOutput {
  predictedVol: number;
  var95: number;
  sharpeEstimate: number;
  inferenceHash: string;
}

// ── TEE / Attestation ────────────────────────────────────────

export interface TeeAttestation {
  enclaveId: string;
  attestationKey: string;    // hex
  quote: string;             // SGX DCAP quote hex
  timestamp: number;
  nodeId: string;
}

export interface VerifiableInferenceResult {
  sessionId: string;
  onnxHash: string;
  teeAttestation: TeeAttestation;
  blockAnchor: BlockAnchor;
  reasoningTrace: ReasoningStep[];
}

export interface BlockAnchor {
  blockNumber: number;
  blockHash: string;
  txHash: string;
  chainId: number;
}

// ── ReAct Agent ──────────────────────────────────────────────

export type ReasoningStepType = 'thought' | 'action' | 'observation' | 'final';

export interface ReasoningStep {
  type: ReasoningStepType;
  content: string;
  timestamp: number;
}

export interface AgentAction {
  tool: AgentTool;
  input: Record<string, unknown>;
  output?: Record<string, unknown>;
}

export type AgentTool =
  | 'portfolio_exposure_analyzer'
  | 'onnx_volatility_model'
  | 'tee_reasoning_engine'
  | 'on_chain_anchor'
  | 'sector_correlation_map';

// ── Risk Analysis Output ──────────────────────────────────────

export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'EXTREME';

export interface RiskMetrics {
  aggregateRiskScore: number;       // 0–100
  riskLevel: RiskLevel;
  portfolioVol: number;             // annualized
  portfolioBeta: number;
  var95: number;                    // 1-day 95% VaR
  sharpeEstimate: number;
  concentrationRisk: number;        // max single asset pct
  sectorConcentration: SectorMap;
  correlationRisk: number;          // avg pairwise correlation
}

export interface SectorMap {
  [sector: string]: number;  // % allocation by sector
}

export interface Recommendation {
  level: 'info' | 'warning' | 'critical';
  category: RecommendationCategory;
  title: string;
  detail: string;
  action?: string;
}

export type RecommendationCategory =
  | 'CONCENTRATION'
  | 'VOLATILITY'
  | 'CORRELATION'
  | 'TOLERANCE_MISMATCH'
  | 'SECTOR_RISK'
  | 'DIVERSIFICATION'
  | 'VERIFICATION';

export interface RiskAnalysisResult {
  sessionId: string;
  timestamp: number;
  portfolioConfig: PortfolioConfig;
  onnxOutput: OnnxModelOutput;
  metrics: RiskMetrics;
  recommendations: Recommendation[];
  verifiableProof: VerifiableInferenceResult;
  pipelineLog: PipelineLogEntry[];
}

export interface PipelineLogEntry {
  step: PipelineStep;
  status: 'pending' | 'running' | 'complete' | 'error';
  message: string;
  durationMs?: number;
  data?: Record<string, unknown>;
}

export type PipelineStep =
  | 'EXPOSURE_ANALYSIS'
  | 'ONNX_INFERENCE'
  | 'TEE_REASONING'
  | 'RECOMMENDATION_OUTPUT';

// ── HTTP API ─────────────────────────────────────────────────

export interface AnalyzeRequest {
  assets: Array<{ symbol: string; allocationPct: number }>;
  timeframe: PortfolioConfig['timeframe'];
  riskTolerance: number;
  walletAddress?: string;
}

export interface AnalyzeResponse {
  success: boolean;
  data?: RiskAnalysisResult;
  error?: string;
}
