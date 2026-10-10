// Single source of truth for submitted papers.
// The publications page, the CV's publication list, the home page's Publications and News lines
// and the Research chips read this. Still written by hand when a stage changes: the home intro and
// Selected-work chips, the Research summary lines, each paper page, the CV research-experience
// bullets and the About page.

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
  news?: string;         // home News text after the paper's name, when "<stage> at ESWA" no longer fits
  next?: { stage: string; short?: string; venueHtml: string };  // after a decline: the step now in progress (home page shows only this)
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
    venueHtml: 'Submitted to <b>Applied Soft Computing</b> · previously Expert Systems with Applications (ESWA-D-26-35395) · earlier title: <i>DAMP: Damped Adaptive Market Physics…</i>',
    stage: 'Submitted', stageClass: 's-rev', filterStage: 'submitted',
    events: [
      { date: '2026-08-22', stage: 'Submitted', short: 'ESWA', note: 'Expert Systems with Applications · ESWA-D-26-35395 · PDF build approved' },
      { date: '2026-08-24', stage: 'With editor', note: 'handling editor assigned · desk &amp; format check' },
      { date: '2026-08-31', stage: 'Under review', note: 'reviewers secured · external peer review started' },
      { date: '2026-09-17', stage: 'Reviews completed', short: 'RRC', note: 'all reviewer reports in · awaiting editor decision' },
      { date: '2026-10-07', stage: 'Declined', bad: true, note: 'Expert Systems with Applications, after review — next: Applied Soft Computing' },
      { date: '2026-10-10', stage: 'Submitted', short: 'ASOC', note: 'Applied Soft Computing' },
    ],
    page: '/research/finphasor/',
    lk: 'manuscript not public',
    news: 'submitted to Applied Soft Computing after review at Expert Systems with Applications',
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
    next: { stage: 'Next venue pending', venueHtml: 'Next venue pending' },
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
    next: { stage: 'Next venue pending', venueHtml: 'Next venue pending' },
    lk: 'manuscript not public',
  },
];

export const byId = (id: string) => PUBS.find((p) => p.id === id)!;

/** The submission now in progress, for the home page: events from the latest
 *  (re)submission on. After a decline: when the next venue is already chosen
 *  (next.short), only that next step shows; while the next venue is still open,
 *  the last run stays, followed by "next venue pending". Every earlier venue
 *  stays on Publications. */
export function currentRun(p: Pub): Ev[] {
  const last = p.events[p.events.length - 1];
  let i = p.events.length - 1;
  while (i > 0 && !/submitted/i.test(p.events[i].stage)) i--;
  const run = p.events.slice(i);
  if (!(last.bad && p.next)) return run;
  const step = { date: last.date, stage: p.next.stage, short: p.next.short };
  return p.next.short ? [step] : [...run, step];
}

/** The venue line for the home page: only the chosen next venue once one is set. */
export const currentVenue = (p: Pub) =>
  p.events[p.events.length - 1].bad && p.next?.short ? p.next.venueHtml : p.venueHtml.split(' · earlier title')[0];

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
