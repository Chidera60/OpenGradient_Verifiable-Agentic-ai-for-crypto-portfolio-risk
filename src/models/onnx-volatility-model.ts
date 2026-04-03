// ============================================================
// models/onnx-volatility-model.ts
// ONNX model runner for crypto portfolio volatility prediction
// ============================================================

import * as ort from 'onnxruntime-node';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import type { OnnxModelInput, OnnxModelOutput, PortfolioConfig } from '../types';

// Model config
const MODEL_NAME = 'crypto_vol_transformer_v2.onnx';
const MODEL_PATH = path.join(__dirname, '../../models', MODEL_NAME);
const N_MARKET_FEATURES = 32;

// Market feature indices (what each column represents in the feature matrix)
const FEATURE_SCHEMA = [
  'log_return_1d', 'log_return_7d', 'log_return_30d',
  'realized_vol_7d', 'realized_vol_30d', 'realized_vol_90d',
  'rsi_14', 'macd_signal', 'bb_width',
  'volume_zscore_7d', 'volume_zscore_30d',
  'open_interest_delta', 'funding_rate',
  'btc_correlation_30d', 'eth_correlation_30d',
  'market_cap_rank_normalized',
  'sector_momentum_7d', 'sector_momentum_30d',
  'macro_vix_30d', 'macro_dxy_30d',
  'on_chain_active_addresses_7d', 'on_chain_tx_count_7d',
  'on_chain_exchange_inflow_7d', 'on_chain_exchange_outflow_7d',
  'sentiment_score_7d', 'fear_greed_index',
  'options_iv_atm_30d', 'options_skew_25d',
  'liquidations_24h', 'open_interest_usd',
  'dominance_pct', 'beta_rolling_30d',
] as const;

export type FeatureKey = typeof FEATURE_SCHEMA[number];

// ── ONNX Session Manager ──────────────────────────────────────

export class OnnxVolatilityModel {
  private session: ort.InferenceSession | null = null;
  private modelPath: string;

  constructor(modelPath: string = MODEL_PATH) {
    this.modelPath = modelPath;
  }

  async initialize(): Promise<void> {
    if (this.session) return;

    if (fs.existsSync(this.modelPath)) {
      // Load real ONNX model
      this.session = await ort.InferenceSession.create(this.modelPath, {
        executionProviders: ['cpu'],
        graphOptimizationLevel: 'all',
      });
      console.log(`[ONNX] Loaded model: ${MODEL_NAME}`);
    } else {
      // Model not found — use synthetic inference (demo mode)
      console.warn(`[ONNX] Model file not found at ${this.modelPath}. Running in simulation mode.`);
    }
  }

  /**
   * Run forward pass on the volatility model.
   * Falls back to closed-form approximation if ONNX model file not present.
   */
  async infer(
    input: OnnxModelInput,
    portfolio: PortfolioConfig
  ): Promise<OnnxModelOutput> {
    const startMs = Date.now();

    let predictedVol: number;
    let var95: number;
    let sharpeEstimate: number;

    if (this.session) {
      // Real ONNX inference
      const feeds = this.buildFeeds(input);
      const results = await this.session.run(feeds);

      predictedVol = (results['predicted_vol'].data as Float32Array)[0];
      var95 = (results['var_95'].data as Float32Array)[0];
      sharpeEstimate = (results['sharpe'].data as Float32Array)[0];
    } else {
      // Simulation: closed-form Markowitz-style approximation
      ({ predictedVol, var95, sharpeEstimate } = this.simulateInference(input, portfolio));
    }

    const elapsedMs = Date.now() - startMs;
    const inferenceHash = this.hashOutput({ predictedVol, var95, sharpeEstimate, input });

    console.log(`[ONNX] Inference complete in ${elapsedMs}ms | hash: ${inferenceHash.slice(0, 18)}…`);

    return { predictedVol, var95, sharpeEstimate, inferenceHash };
  }

  // ── Private helpers ──────────────────────────────────────────

  private buildFeeds(input: OnnxModelInput): Record<string, ort.Tensor> {
    const nAssets = input.portfolioWeights.length;
    return {
      portfolio_weights: new ort.Tensor('float32', input.portfolioWeights, [1, nAssets]),
      market_features: new ort.Tensor('float32', input.marketFeatures, [nAssets, N_MARKET_FEATURES]),
      timeframe_index: new ort.Tensor('int64', [BigInt(input.timeframeIndex)], [1]),
    };
  }

  /**
   * Closed-form simulation used when ONNX model file is absent.
   * Implements a Markowitz-style variance formula with correlation adjustment.
   */
  private simulateInference(
    input: OnnxModelInput,
    portfolio: PortfolioConfig
  ): { predictedVol: number; var95: number; sharpeEstimate: number } {
    const weights = Array.from(input.portfolioWeights);
    const vols = portfolio.assets.map(a => a.historicalVol);
    const corrs = portfolio.assets.map(a => a.correlation);

    // Simplified covariance: diagonal + off-diagonal via avg correlation
    let portfolioVar = 0;
    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < weights.length; j++) {
        const rho = i === j ? 1.0 : (corrs[i] + corrs[j]) / 2;
        portfolioVar += weights[i] * weights[j] * vols[i] * vols[j] * rho;
      }
    }

    const predictedVol = Math.sqrt(portfolioVar) * 0.91;  // model correction factor
    const var95 = predictedVol * 1.645 * Math.sqrt(1 / 252);
    const riskFreeRate = 0.05;
    const expectedReturn = portfolio.assets.reduce((sum, a, i) => {
      return sum + weights[i] * (a.beta * 0.12);  // CAPM estimate
    }, 0);
    const sharpeEstimate = (expectedReturn - riskFreeRate) / predictedVol;

    return { predictedVol, var95, sharpeEstimate };
  }

  private hashOutput(data: object): string {
    const bytes = Buffer.from(JSON.stringify(data));
    return '0x' + crypto.createHash('sha256').update(bytes).digest('hex');
  }
}

// ── Feature Engineering ────────────────────────────────────────

/**
 * Builds the market feature matrix for a portfolio.
 * In production, this pulls live on-chain and CEX data.
 * Here it uses synthetic features seeded from asset metadata.
 */
export function buildMarketFeatures(portfolio: PortfolioConfig): Float32Array {
  const nAssets = portfolio.assets.length;
  const features = new Float32Array(nAssets * N_MARKET_FEATURES);

  for (let i = 0; i < nAssets; i++) {
    const asset = portfolio.assets[i];
    const base = i * N_MARKET_FEATURES;
    const vol = asset.historicalVol;
    const beta = asset.beta;

    // Fill feature row with synthetic but plausible values
    features[base + 0]  = -0.02 + (Math.random() - 0.5) * 0.04;  // log_return_1d
    features[base + 1]  = -0.05 + (Math.random() - 0.5) * 0.10;  // log_return_7d
    features[base + 2]  = 0.02 + (Math.random() - 0.5) * 0.15;   // log_return_30d
    features[base + 3]  = vol * 0.95;                              // realized_vol_7d
    features[base + 4]  = vol;                                     // realized_vol_30d
    features[base + 5]  = vol * 1.05;                              // realized_vol_90d
    features[base + 6]  = 40 + Math.random() * 40;                // rsi_14
    features[base + 7]  = (Math.random() - 0.5) * 0.02;           // macd_signal
    features[base + 8]  = vol * 2.2;                               // bb_width
    features[base + 9]  = (Math.random() - 0.5) * 2;              // volume_zscore_7d
    features[base + 10] = (Math.random() - 0.5) * 1.5;            // volume_zscore_30d
    features[base + 11] = (Math.random() - 0.5) * 0.05;           // open_interest_delta
    features[base + 12] = (Math.random() - 0.5) * 0.001;          // funding_rate
    features[base + 13] = asset.correlation;                       // btc_correlation_30d
    features[base + 14] = asset.correlation * 0.9;                 // eth_correlation_30d
    features[base + 15] = Math.random();                           // market_cap_rank_normalized
    features[base + 16] = beta * 0.02 + (Math.random() - 0.5) * 0.01; // sector_momentum_7d
    features[base + 17] = beta * 0.05 + (Math.random() - 0.5) * 0.03; // sector_momentum_30d
    features[base + 18] = 18 + Math.random() * 10;                // macro_vix_30d
    features[base + 19] = 102 + (Math.random() - 0.5) * 4;        // macro_dxy_30d
    features[base + 20] = 0.5 + Math.random() * 0.5;              // on_chain_active_addresses_7d
    features[base + 21] = 0.5 + Math.random() * 0.5;              // on_chain_tx_count_7d
    features[base + 22] = Math.random() * 0.3;                    // on_chain_exchange_inflow_7d
    features[base + 23] = Math.random() * 0.3;                    // on_chain_exchange_outflow_7d
    features[base + 24] = 0.3 + Math.random() * 0.4;              // sentiment_score_7d
    features[base + 25] = 30 + Math.random() * 40;                // fear_greed_index
    features[base + 26] = vol * 1.2 + Math.random() * 0.1;       // options_iv_atm_30d
    features[base + 27] = (Math.random() - 0.5) * 0.1;            // options_skew_25d
    features[base + 28] = Math.random() * 0.2;                    // liquidations_24h
    features[base + 29] = Math.random() * 0.8;                    // open_interest_usd
    features[base + 30] = 0.05 + Math.random() * 0.4;             // dominance_pct
    features[base + 31] = beta;                                    // beta_rolling_30d
  }

  return features;
}

export function timeframeToIndex(tf: PortfolioConfig['timeframe']): number {
  return { '7d': 0, '30d': 1, '90d': 2, '1y': 3 }[tf];
}

// ── Singleton ────────────────────────────────────────────────

let _modelInstance: OnnxVolatilityModel | null = null;

export async function getOnnxModel(): Promise<OnnxVolatilityModel> {
  if (!_modelInstance) {
    _modelInstance = new OnnxVolatilityModel();
    await _modelInstance.initialize();
  }
  return _modelInstance;
}
