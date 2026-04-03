// ============================================================
// utils/asset-registry.ts
// On-chain asset metadata registry for supported tokens
// ============================================================

import type { Asset, PortfolioConfig, AssetSector } from '../types';

// ── Asset Metadata ─────────────────────────────────────────

interface AssetMeta {
  historicalVol: number;   // 30-day realized vol, annualized
  beta: number;            // rolling 30-day beta vs BTC
  correlation: number;     // rolling 30-day correlation vs BTC
  marketCap: string;
  sector: AssetSector;
}

export const ASSET_REGISTRY: Record<string, AssetMeta> = {
  BTC:  { historicalVol: 0.65, beta: 1.00, correlation: 1.00, marketCap: '$1.2T', sector: 'Store of Value' },
  ETH:  { historicalVol: 0.82, beta: 1.18, correlation: 0.88, marketCap: '$380B', sector: 'Smart Contract' },
  SOL:  { historicalVol: 1.42, beta: 1.85, correlation: 0.74, marketCap: '$78B',  sector: 'Smart Contract' },
  BNB:  { historicalVol: 0.91, beta: 1.22, correlation: 0.82, marketCap: '$88B',  sector: 'Exchange' },
  ARB:  { historicalVol: 1.55, beta: 1.95, correlation: 0.70, marketCap: '$3.2B', sector: 'L2' },
  AVAX: { historicalVol: 1.28, beta: 1.65, correlation: 0.76, marketCap: '$14B',  sector: 'Smart Contract' },
  MATIC:{ historicalVol: 1.38, beta: 1.78, correlation: 0.72, marketCap: '$9.2B', sector: 'L2' },
  LINK: { historicalVol: 1.12, beta: 1.45, correlation: 0.68, marketCap: '$9.8B', sector: 'Oracle' },
  OP:   { historicalVol: 1.62, beta: 2.01, correlation: 0.69, marketCap: '$2.8B', sector: 'L2' },
  UNI:  { historicalVol: 1.25, beta: 1.58, correlation: 0.71, marketCap: '$7.4B', sector: 'DeFi' },
  AAVE: { historicalVol: 1.35, beta: 1.68, correlation: 0.66, marketCap: '$3.2B', sector: 'DeFi' },
  INJ:  { historicalVol: 1.88, beta: 2.22, correlation: 0.62, marketCap: '$2.1B', sector: 'DeFi' },
  ATOM: { historicalVol: 1.18, beta: 1.52, correlation: 0.70, marketCap: '$4.8B', sector: 'Smart Contract' },
  DOT:  { historicalVol: 1.08, beta: 1.38, correlation: 0.75, marketCap: '$11B',  sector: 'Smart Contract' },
  USDC: { historicalVol: 0.01, beta: 0.00, correlation: 0.00, marketCap: '$32B',  sector: 'Stablecoin' },
};

export const SUPPORTED_SYMBOLS = Object.keys(ASSET_REGISTRY);

// ── Validation & Construction ─────────────────────────────

export interface PortfolioInput {
  assets: Array<{ symbol: string; allocationPct: number }>;
  timeframe: PortfolioConfig['timeframe'];
  riskTolerance: number;
  walletAddress?: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

export function validatePortfolioInput(input: PortfolioInput): ValidationResult {
  const errors: string[] = [];

  if (!input.assets || input.assets.length === 0) {
    errors.push('Portfolio must contain at least one asset.');
  }

  if (input.assets.length > 15) {
    errors.push('Maximum 15 assets supported per analysis.');
  }

  const unknownSymbols = input.assets
    .map(a => a.symbol.toUpperCase())
    .filter(s => !ASSET_REGISTRY[s]);

  if (unknownSymbols.length > 0) {
    errors.push(`Unknown asset symbols: ${unknownSymbols.join(', ')}. Supported: ${SUPPORTED_SYMBOLS.join(', ')}`);
  }

  const totalAlloc = input.assets.reduce((sum, a) => sum + a.allocationPct, 0);
  if (Math.abs(totalAlloc - 100) > 0.5) {
    errors.push(`Allocations must sum to 100% (current total: ${totalAlloc.toFixed(2)}%).`);
  }

  for (const asset of input.assets) {
    if (asset.allocationPct < 0 || asset.allocationPct > 100) {
      errors.push(`Allocation for ${asset.symbol} must be between 0 and 100%.`);
    }
  }

  if (input.riskTolerance < 1 || input.riskTolerance > 10) {
    errors.push('Risk tolerance must be between 1 and 10.');
  }

  if (!['7d', '30d', '90d', '1y'].includes(input.timeframe)) {
    errors.push('Timeframe must be one of: 7d, 30d, 90d, 1y.');
  }

  if (input.walletAddress && !/^0x[a-fA-F0-9]{40}$/.test(input.walletAddress)) {
    errors.push('Invalid Ethereum wallet address format.');
  }

  return { valid: errors.length === 0, errors };
}

export function buildPortfolioConfig(input: PortfolioInput): PortfolioConfig {
  const assets: Asset[] = input.assets.map(a => {
    const symbol = a.symbol.toUpperCase();
    const meta = ASSET_REGISTRY[symbol];
    if (!meta) throw new Error(`Unknown symbol: ${symbol}`);
    return { symbol, allocationPct: a.allocationPct, ...meta };
  });

  return {
    assets,
    timeframe: input.timeframe,
    riskTolerance: input.riskTolerance,
    walletAddress: input.walletAddress,
  };
}
