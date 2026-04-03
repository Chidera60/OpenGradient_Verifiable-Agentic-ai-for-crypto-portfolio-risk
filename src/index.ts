// ============================================================
// index.ts — CLI entry point
// Usage: npx ts-node src/index.ts
// ============================================================

import 'dotenv/config';
import chalk from 'chalk';
import ora from 'ora';
import { CryptoRiskAdvisorAgent } from './agent/react-agent';
import { buildPortfolioConfig, validatePortfolioInput } from './utils/asset-registry';
import type { RiskAnalysisResult } from './types';

// ── Demo portfolio ────────────────────────────────────────────

const DEMO_INPUT = {
  assets: [
    { symbol: 'BTC',  allocationPct: 40 },
    { symbol: 'ETH',  allocationPct: 30 },
    { symbol: 'SOL',  allocationPct: 15 },
    { symbol: 'ARB',  allocationPct: 10 },
    { symbol: 'LINK', allocationPct: 5  },
  ],
  timeframe: '30d' as const,
  riskTolerance: 6,
};

// ── Formatting helpers ────────────────────────────────────────

function printHeader(): void {
  console.log('\n' + chalk.cyan('╔══════════════════════════════════════════════════════╗'));
  console.log(chalk.cyan('║') + chalk.bold.white('   Verifiable AI Crypto Risk Advisor v1.0            ') + chalk.cyan('║'));
  console.log(chalk.cyan('║') + chalk.gray('   OpenGradient · ONNX · TEE · ReAct Agent           ') + chalk.cyan('║'));
  console.log(chalk.cyan('╚══════════════════════════════════════════════════════╝\n'));
}

function printPortfolio(): void {
  console.log(chalk.bold('Portfolio:'));
  DEMO_INPUT.assets.forEach(a => {
    const bar = '█'.repeat(Math.round(a.allocationPct / 5));
    console.log(
      `  ${chalk.cyan(a.symbol.padEnd(6))} ${chalk.white(String(a.allocationPct).padStart(3))}%  ${chalk.gray(bar)}`
    );
  });
  console.log(`  ${chalk.gray('Timeframe:')} ${DEMO_INPUT.timeframe}  ${chalk.gray('Risk tolerance:')} ${DEMO_INPUT.riskTolerance}/10\n`);
}

function printResult(result: RiskAnalysisResult): void {
  const { metrics, recommendations, verifiableProof, onnxOutput } = result;

  const riskColor = metrics.riskLevel === 'LOW' ? chalk.green
    : metrics.riskLevel === 'MEDIUM' ? chalk.yellow
    : metrics.riskLevel === 'HIGH' ? chalk.red
    : chalk.bold.red;

  console.log('\n' + chalk.bold('─── Risk Metrics ─────────────────────────────────────'));
  console.log(`  Risk Score:     ${riskColor(metrics.aggregateRiskScore + '/100')} (${riskColor(metrics.riskLevel)})`);
  console.log(`  Portfolio Vol:  ${chalk.white((metrics.portfolioVol * 100).toFixed(2) + '%')} annualized`);
  console.log(`  Portfolio Beta: ${chalk.white(metrics.portfolioBeta.toFixed(3))} vs BTC`);
  console.log(`  VaR (95%, 1d):  ${chalk.white((metrics.var95 * 100).toFixed(2) + '%')}`);
  console.log(`  Sharpe Est.:    ${chalk.white(metrics.sharpeEstimate.toFixed(3))}`);

  console.log('\n' + chalk.bold('─── Verifiable Proof ─────────────────────────────────'));
  console.log(`  Session:     ${chalk.cyan(result.sessionId)}`);
  console.log(`  ONNX Hash:   ${chalk.cyan(onnxOutput.inferenceHash.slice(0, 26) + '…')}`);
  console.log(`  TEE Node:    ${chalk.cyan(verifiableProof.teeAttestation.nodeId)}`);
  console.log(`  Block:       ${chalk.cyan('#' + verifiableProof.blockAnchor.blockNumber)}`);
  console.log(`  Tx:          ${chalk.cyan(verifiableProof.blockAnchor.txHash.slice(0, 26) + '…')}`);
  console.log(`  Verify at:   ${chalk.underline.blue('https://explorer.opengradient.ai/sessions/' + result.sessionId)}`);

  console.log('\n' + chalk.bold('─── Recommendations ──────────────────────────────────'));
  recommendations.forEach(rec => {
    const icon = rec.level === 'critical' ? chalk.red('✖')
      : rec.level === 'warning' ? chalk.yellow('⚠')
      : chalk.green('✓');
    console.log(`\n  ${icon}  ${chalk.bold(rec.title)}`);
    console.log(`     ${chalk.gray(rec.detail)}`);
    if (rec.action) console.log(`     ${chalk.cyan('→')} ${rec.action}`);
  });

  console.log('\n' + chalk.green('Analysis complete ✓\n'));
}

// ── Main ──────────────────────────────────────────────────────

async function main(): Promise<void> {
  printHeader();
  printPortfolio();

  const validation = validatePortfolioInput(DEMO_INPUT);
  if (!validation.valid) {
    console.error(chalk.red('Validation error:'), validation.errors.join('\n'));
    process.exit(1);
  }

  const spinner = ora({
    text: 'Initializing OpenGradient TEE agent...',
    color: 'cyan',
  }).start();

  try {
    const portfolio = buildPortfolioConfig(DEMO_INPUT);
    const agent = new CryptoRiskAdvisorAgent();

    spinner.text = 'Running ReAct pipeline...';
    const result = await agent.analyze(portfolio);

    spinner.succeed('Pipeline complete');
    printResult(result);
  } catch (err) {
    spinner.fail('Analysis failed');
    console.error(chalk.red(err instanceof Error ? err.message : String(err)));
    process.exit(1);
  }
}

main();
