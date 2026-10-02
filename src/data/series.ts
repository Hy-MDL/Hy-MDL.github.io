export const SERIES: Record<string, { label: string; blurb: string }> = {
  'score-to-flow': { label: 'Score → Flow', blurb: 'From NCSN to flow matching' },
  'normalizing-flows': { label: 'Normalizing flows', blurb: 'Change of variables, coupling and autoregressive flows, exact likelihood' },
  'stochastic-modeling': { label: 'Stochastic modeling', blurb: 'Neural ODE/SDEs, rough volatility, diffusion for time series' },
  'sequential-monte-carlo': { label: 'Sequential Monte Carlo', blurb: 'Particle filters, resampling, particle MCMC' },
  'submodular-optimization': { label: 'Submodular optimization', blurb: 'Diminishing returns, greedy guarantees, subset selection' },
  'surrogates-bo': { label: 'Surrogates, priors & Bayesian optimisation', blurb: 'Learned priors and surrogates, and optimising expensive objectives through them' },
  'generative-finance': { label: 'Generative finance', blurb: 'GANs and diffusion for market scenarios' },
  'eswa-finance': { label: 'ESWA finance', blurb: 'Finance papers from Expert Systems with Applications' },
  'ee-timeseries': { label: 'Signals & device health', blurb: 'Time series and physics-informed ML on electronic devices — degradation, lifetime, prognostics' },
  'vision': { label: 'Vision', blurb: 'Backbones, representation, detection, 3D' },
  'industrial-vision': { label: 'Industrial vision', blurb: 'Anomaly detection and inspection — MVTec-style data, CT, device quality control' },
};
export const SERIES_ORDER = Object.keys(SERIES);
export const fmt = (d: Date) => d.toISOString().slice(0, 10).replace(/-/g, '.');
