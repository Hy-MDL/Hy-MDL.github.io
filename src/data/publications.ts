// Single source of truth for submitted papers.
// The home page, the publications page and the paper project pages all read this,
// so a status change is edited once here and nowhere else.

export type Ev = {
  date: string;        // ISO, the date the stage was reached
  stage: string;       // shown on the axis label and the timeline row
  short?: string;      // axis-only abbreviation when the venue changes (e.g. "ESWA")
  note?: string;       // the detail line on the publications page
  bad?: boolean;       // a decline — drawn in the negative colour
};

export type Pub = {
  id: string;
  year: string;
  title: string;
  authorsHtml: string;   // pre-marked because the author lists differ in shape
  role: string;          // chip text
  roleClass?: string;    // "first" gives the chip the emphasised border
  filterRole: 'first' | 'co';
  venueHtml: string;
  stage: string;         // current stage, shown as the status chip
  stageShort?: string;   // shorter form for the home page chip
  stageClass: string;    // s-rev | s-plan | s-acc | s-wip
  filterStage: 'submitted' | 'review' | 'rrc' | 'pending' | 'presented';
  events: Ev[];
  page?: string;         // the write-up, when one is public
  lk?: string;           // availability line
};

export const PUBS: Pub[] = [
  {
    id: 'case',
    year: '2026',
    title: 'Conditional Diffusion Transformers for Multi-Asset Scenario Generation and Portfolio Risk Measurement',
    authorsHtml: '<b>Hyeon Min Jeon</b>, Hee Soo Lee<sup>*</sup>, Kyong Joo Oh<sup>*</sup>',
    role: '1st author', roleClass: 'first', filterRole: 'first',
    venueHtml: 'Submitted to <b>Expert Systems with Applications</b> · ESWA-D-26-39029 · earlier title: <i>Diffusion transformers for generative modeling of joint asset return distributions and portfolio risk scenarios</i>',
    stage: 'Under review', stageClass: 's-rev', filterStage: 'review',
    events: [
      { date: '2026-09-17', stage: 'Submitted', short: 'ESWA', note: 'Expert Systems with Applications · ESWA-D-26-39029' },
      { date: '2026-09-20', stage: 'With editor', note: 'handling editor assigned · desk &amp; format check' },
      { date: '2026-09-22', stage: 'Under review', note: 'reviewers secured · external peer review started' },
    ],
    page: '/research/diffusion-scenarios/',
    lk: 'manuscript not public',
  },
  {
    id: 'finphasor',
    year: '2026',
    title: 'FinPhasor: Phase-Preserving Measurement of Cross-Sectional Timing in Equity Markets',
    authorsHtml: '<b>Hyeon Min Jeon</b>, Hee Soo Lee<sup>*</sup>, Kyong Joo Oh<sup>*</sup>',
    role: '1st author', roleClass: 'first', filterRole: 'first',
    venueHtml: 'Submitted to <b>Expert Systems with Applications</b> · ESWA-D-26-35395 · earlier title: <i>DAMP: Damped Adaptive Market Physics…</i>',
    stage: 'Required Reviews Completed', stageShort: 'RRC', stageClass: 's-rev', filterStage: 'rrc',
    events: [
      { date: '2026-08-22', stage: 'Submitted', short: 'ESWA', note: 'Expert Systems with Applications · ESWA-D-26-35395 · PDF build approved' },
      { date: '2026-08-24', stage: 'With editor', note: 'handling editor assigned · desk &amp; format check' },
      { date: '2026-08-31', stage: 'Under review', note: 'reviewers secured · external peer review started' },
      { date: '2026-09-17', stage: 'Reviews completed', short: 'RRC', note: 'all reviewer reports in · awaiting editor decision' },
    ],
    page: '/research/finphasor/',
    lk: 'manuscript not public',
  },
  {
    id: 'convfactornet',
    year: '2025',
    title: 'Deep Convolutional Factor Prediction with Explainable AI for Enhanced Investment Strategy',
    authorsHtml: 'Sang Hoe Kim, <b>Hyeon Min Jeon</b>, Dae Hyuk You, Jiyoung Jeon, Seungho Baek, Hee Soo Lee, Kyong Joo Oh<sup>*</sup>',
    role: '2nd author', filterRole: 'co',
    venueHtml: 'Last submitted to <b>Applied Soft Computing</b> · previously Expert Systems with Applications · next venue pending',
    stage: 'Pending resubmission', stageClass: 's-plan', filterStage: 'pending',
    events: [
      { date: '2025-10-29', stage: 'Submitted', short: 'ESWA', note: 'Expert Systems with Applications · ESWA-D-25-29858' },
      { date: '2026-06-09', stage: 'Declined', bad: true, note: 'Expert Systems with Applications' },
      { date: '2026-07-16', stage: 'Resubmitted', short: 'ASOC', note: 'Applied Soft Computing · ASOC-D-26-13102' },
      { date: '2026-09-01', stage: 'Declined', bad: true, note: 'Applied Soft Computing — next venue pending' },
    ],
    lk: 'manuscript not public',
  },
  {
    id: 'neif',
    year: '2025',
    title: 'Benchmark-Aware Enhanced Indexing Guided by the Volume–Price Index',
    authorsHtml: 'Dae Hyuk You, Jiyoung Jeon, <b>Hyeon Min Jeon</b>, Sang Hoe Kim, Sang Hyuk Yoo, Tae Yoon Kim, Hee Soo Lee, Kyong Joo Oh<sup>*</sup>',
    role: '3rd author', filterRole: 'co',
    venueHtml: 'Last submitted to <b>JCIS</b> · previously Expert Systems with Applications (ESWA-D-25-33272) · next venue pending · earlier title: <i>Hybrid Expert System for Enhanced Index Fund Framework via Genetic Algorithm, Volume-Price Index, and LSTM Dueling DQN</i>',
    stage: 'Pending resubmission', stageClass: 's-plan', filterStage: 'pending',
    events: [
      { date: '2025-11-28', stage: 'Submitted', short: 'ESWA', note: 'Expert Systems with Applications · ESWA-D-25-33272' },
      { date: '2025-12-03', stage: 'Declined', bad: true, note: 'Expert Systems with Applications — next venue pending' },
    ],
    lk: 'manuscript not public',
  },
];

export const byId = (id: string) => PUBS.find((p) => p.id === id)!;

/** Whole days between two ISO dates. */
export const gapDays = (a: string, b: string) =>
  Math.round((Date.parse(b) - Date.parse(a)) / 86400000);

/** "3d" for short waits, "2mo 11d" once a gap runs past a month. */
export function gapLabel(a: string, b: string): string {
  const d = gapDays(a, b);
  if (d < 31) return d + 'd';
  const mo = Math.floor(d / 30.44), rem = Math.round(d - mo * 30.44);
  return rem ? `${mo}mo ${rem}d` : `${mo}mo`;
}

/** Days since the most recent event — how long it has been sitting where it is. */
export function daysWaiting(p: Pub, today = new Date()): number {
  return Math.round((today.getTime() - Date.parse(p.events[p.events.length - 1].date)) / 86400000);
}
