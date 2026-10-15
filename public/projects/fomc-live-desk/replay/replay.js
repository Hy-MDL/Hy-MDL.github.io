// "Replay the broadcast": the recorded 16 Sept 2026 run, replayed on its own wall clock.
// All numbers come from replay.json (built by scripts/make_replay_data.py from the run's logs).
(function () {
  const root = document.getElementById('fomc-replay');
  if (!root) return;
  const BASE = '/projects/fomc-live-desk/replay/';
  const T_END = 53.0;
  const NS = 'http://www.w3.org/2000/svg';
  const TICKERS = ['SPY', 'TLT', 'UUP'];
  const NODE_CLASS = { Bull: 'bull', Bear: 'bear', 'Research mgr': 'mgr', Trader: 'trader',
    Aggressive: 'risk', Conservative: 'risk', Neutral: 'risk', PM: 'pm' };
  const RATING_CLASS = { Buy: 'pos', Overweight: 'pos', Sell: 'neg', Underweight: 'neg', Hold: '' };
  const STANCE = { 1: 'very dovish', 2: 'dovish', 3: 'neutral', 4: 'hawkish', 5: 'very hawkish' };

  const h = (tag, attrs = {}, kids = []) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) k === 'text' ? (e.textContent = v) : e.setAttribute(k, v);
    for (const c of [].concat(kids)) if (c != null) e.append(c);
    return e;
  };
  const S = (tag, attrs = {}, text) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  };
  const sgn = (x, d = 2) => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(d);
  const fp = (p) => (p > 0 && p < 0.005 ? '<0.01' : p > 0.995 && p < 1 ? '>0.99' : p.toFixed(2));
  const ICON_PLAY = 'M2 1 L11 6 L2 11 Z';
  const ICON_PAUSE = 'M2 1 H5 V11 H2 Z M7 1 H10 V11 H7 Z';

  let D = null;
  const st = { t: T_END, playing: false, speed: 1, desk: 1, tab: 'SPY', last: 0, fired: new Set(), lastAudio: -1 };
  let audio = null, audioOk = true, raf = 0;
  const el = {};

  // waveform, drawn at the container's pixel width so text and marks keep their size
  function buildWave() {
    const W = Math.round(Math.max(300, Math.min(1000, root.clientWidth - 28))), WH = 150, cy = 62, amp = 40, spY = 112;
    const x = (t) => 10 + (W - 20) * t / T_END;
    el.x = x;
    const svg = S('svg', { viewBox: `0 0 ${W} ${WH}`, class: 'rp-wave', role: 'img',
      'aria-label': 'Waveform of the statement audio with sentence spans and desk triggers' });
    el.env = [];
    const n = D.envelope.length, dt = D.audio_s / n;
    D.envelope.forEach((v, i) => {
      const hh = Math.max(0.8, v * amp);
      const r = S('rect', { class: 'env', x: x(i * dt).toFixed(1), y: (cy - hh).toFixed(1),
        width: Math.max(1.2, (x(dt) - x(0)) * 0.62).toFixed(2), height: (2 * hh).toFixed(1) });
      r._t = i * dt; el.env.push(r); svg.append(r);
    });
    el.spans = D.sentences.map((s) => {
      const r = S('rect', { class: 'sp', x: (x(s.audio_start) + 1.5).toFixed(1), y: spY,
        width: Math.max(2, x(s.audio_end) - x(s.audio_start) - 3).toFixed(1), height: 7, rx: 3.5 });
      r.append(S('title', {}, `Sentence ${s.idx + 1}: spoken ${s.audio_start.toFixed(1)}–${s.audio_end.toFixed(1)} s, committed +${s.emit_wall.toFixed(1)} s`));
      svg.append(r); return r;
    });
    for (let s = 0; s <= 50; s += 10) svg.append(S('text', { class: 'ax', x: x(s), y: WH - 4, 'text-anchor': 'middle' }, s + ' s'));
    el.tg = D.triggers.map((tr, i) => {
      const g = S('g', { class: 'tg', tabindex: '0', role: 'button',
        'aria-label': `Trigger #${i} at ${tr.wall.toFixed(1)} s: open the desk run` });
      const xx = x(tr.wall);
      g.append(S('line', { x1: xx, x2: xx, y1: 12, y2: spY + 8 }),
        S('path', { d: `M${xx - 7} 2 H${xx + 7} L${xx} 12 Z` }),
        S('title', {}, `Trigger #${i} (${tr.kind === 'reprice' ? 're-evaluate p crossed 0.5' : 'end of broadcast'}) at +${tr.wall.toFixed(1)} s`));
      const open = (e) => { e.stopPropagation(); openDesk(i); };
      g.addEventListener('click', open);
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(e); } });
      svg.append(g); return g;
    });
    el.ph = S('line', { class: 'ph', y1: 14, y2: spY + 12 });
    svg.append(el.ph);
    svg.addEventListener('click', (e) => {
      const r = svg.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width * W;
      seek(Math.max(0, Math.min(T_END, (px - 10) / (W - 20) * T_END)));
    });
    el.W = W;
    return svg;
  }

  // ---------------------------------------------------------------- build
  function build() {
    root.textContent = '';
    root.setAttribute('tabindex', '0');
    root.setAttribute('role', 'region');
    root.setAttribute('aria-label', 'Replay of the recorded FOMC broadcast run. Space plays or pauses.');

    // controls
    el.play = h('button', { class: 'rp-play', type: 'button', 'aria-label': 'Play the broadcast' });
    el.playIcon = S('path', { d: ICON_PLAY });
    const ic = S('svg', { viewBox: '0 0 12 12', 'aria-hidden': 'true' }); ic.append(el.playIcon);
    el.playTxt = h('span', { text: 'Play' });
    el.play.append(ic, el.playTxt);
    el.play.addEventListener('click', toggle);
    el.clock = h('span', { class: 'rp-clock', 'aria-live': 'off' });
    el.speed = h('div', { class: 'rp-seg', role: 'group', 'aria-label': 'Playback speed' });
    for (const sp of [1, 2]) {
      const b = h('button', { type: 'button', text: sp + '×', 'aria-pressed': String(sp === 1) });
      if (sp === 1) b.classList.add('on');
      b.addEventListener('click', () => setSpeed(sp));
      el.speed.append(b);
    }
    el.scrub = h('input', { class: 'rp-scrub', type: 'range', min: '0', max: String(T_END), step: '0.1',
      value: String(T_END), 'aria-label': 'Broadcast time in seconds' });
    el.scrub.addEventListener('input', () => seek(+el.scrub.value));

    el.wave = buildWave();
    const legend = h('div', { class: 'rp-legend' }, [
      h('span', {}, [h('i', { style: 'background:var(--rule2)' }), 'sentence spoken']),
      h('span', {}, [h('i', { style: 'background:var(--accent)' }), 'committed by streamed STT']),
      h('span', {}, [h('i', { style: 'background:var(--warn);width:3px;height:10px' }), 'desk trigger (click)']),
    ]);

    // transcript + decision
    el.tx = h('div', { class: 'rp-tx', 'aria-live': 'polite' });
    const txPanel = h('div', { class: 'rp-panel' }, [h('div', { class: 'rp-h' }, [h('span', { text: 'Transcript (committed sentences)' })]), el.tx]);
    el.dec = h('div', { class: 'rp-dec' });
    const decPanel = h('div', { class: 'rp-panel' }, [h('div', { class: 'rp-h' }, [h('span', { text: 'Decision on the latest sentence' })]), el.dec,
      h('div', { class: 'rp-emu', text: 'emulated SystemOne (Jev not called): qwen3:14b, 4 fixed questions' })]);
    el.ev = h('div', { class: 'rp-ev' });
    el.desk = h('div', { class: 'rp-desk rp-panel' });

    root.append(h('div', { class: 'rp-bar' }, [el.play, el.clock, el.speed]), el.scrub, el.wave, legend,
      h('div', { class: 'rp-grid' }, [txPanel, decPanel]), el.ev, el.desk,
      h('p', { class: 'rp-foot', text: 'Recorded run on 16 Sept 2026 statement audio (macOS say, 49.9 s), replayed on the run’s wall clock. Prices: yfinance 5-minute bars, returns from the 13:55 bar close.' }));

    root.addEventListener('keydown', (e) => {
      if ((e.key === ' ' || e.code === 'Space') && !e.target.closest('button,input,summary,[role="button"]')) {
        e.preventDefault(); toggle();
      }
    });

    if (window.ResizeObserver) new ResizeObserver(() => {
      if (Math.abs(Math.max(300, Math.min(1000, root.clientWidth - 28)) - el.W) < 24) return;
      const old = el.wave; el.wave = buildWave(); old.replaceWith(el.wave); render();
    }).observe(root);

    audio = new Audio();
    audio.preload = 'none';
    audio.src = BASE + 'statement.m4a';
    audio.addEventListener('error', () => { audioOk = false; });
  }

  // ---------------------------------------------------------------- playback
  function toggle() { st.playing ? pause() : play(); }
  function play() {
    if (st.t >= T_END - 0.05) { st.t = 0; st.desk = null; st.fired.clear(); }
    st.playing = true; st.last = performance.now();
    syncAudio(true);
    raf = requestAnimationFrame(tick);
    render();
  }
  function pause() {
    st.playing = false; cancelAnimationFrame(raf);
    if (audio) audio.pause();
    render();
  }
  function setSpeed(sp) {
    st.speed = sp;
    [...el.speed.children].forEach((b, i) => { const on = [1, 2][i] === sp; b.classList.toggle('on', on); b.setAttribute('aria-pressed', String(on)); });
    if (audio) audio.playbackRate = sp;
  }
  function seek(t) {
    st.t = t;
    st.desk = t >= T_END - 0.05 ? 1 : null;
    st.fired = new Set(D.triggers.map((tr, i) => (tr.wall <= t ? i : -1)).filter((i) => i >= 0));
    if (audio && audioOk && t < D.audio_s) { try { audio.currentTime = t; } catch (e) { /* not loaded yet */ } }
    if (st.playing) syncAudio(true);
    render();
  }
  function syncAudio(force) {
    if (!audio || !audioOk) return;
    if (!st.playing || st.t >= D.audio_s) { if (!audio.paused) audio.pause(); return; }
    audio.playbackRate = st.speed;
    if (audio.paused) {
      try { audio.currentTime = st.t; } catch (e) { /* ignore */ }
      const p = audio.play();
      if (p && p.catch) p.catch(() => { audioOk = false; });
    } else if (force || Math.abs(audio.currentTime - st.t) > 0.35) {
      try { audio.currentTime = st.t; } catch (e) { /* ignore */ }
    }
  }
  function tick(now) {
    const dt = Math.min(0.25, (now - st.last) / 1000) * st.speed;
    st.last = now;
    // follow the audio clock while it is playing, otherwise the timer
    const at = audio && audioOk && !audio.paused ? audio.currentTime : -1;
    if (at > st.lastAudio + 1e-3 && st.t < D.audio_s && Math.abs(at - (st.t + dt)) < 0.35) st.t = at;
    else st.t += dt;  // audio stalled, failed or finished: the timer carries on
    st.lastAudio = at;
    if (st.t >= T_END) { st.t = T_END; st.playing = false; st.desk = 1; if (audio) audio.pause(); }
    D.triggers.forEach((tr, i) => {
      if (tr.wall <= st.t && !st.fired.has(i)) { st.fired.add(i); flash(i); }
    });
    syncAudio(false);
    render();
    if (st.playing) raf = requestAnimationFrame(tick);
  }
  function flash(i) {
    const b = el.ev.querySelector(`[data-i="${i}"]`);
    if (b) { b.classList.remove('flash'); void b.offsetWidth; b.classList.add('flash'); }
  }
  function openDesk(i) {
    if (st.playing) pause();
    st.desk = i;
    if (st.t < D.triggers[i].wall) seek(D.triggers[i].wall);
    st.desk = i;
    render();
    el.desk.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  // ---------------------------------------------------------------- render
  let lastKey = '';
  function render() {
    const t = st.t;
    el.playIcon.setAttribute('d', st.playing ? ICON_PAUSE : ICON_PLAY);
    el.playTxt.textContent = st.playing ? 'Pause' : (t >= T_END - 0.05 ? 'Play from start' : 'Play');
    el.play.setAttribute('aria-label', st.playing ? 'Pause' : 'Play the broadcast');
    el.clock.textContent = `${t.toFixed(1).padStart(4, ' ')} s / ${T_END.toFixed(1)} s`;
    if (document.activeElement !== el.scrub) el.scrub.value = t.toFixed(1);
    el.ph.setAttribute('x1', el.x(t)); el.ph.setAttribute('x2', el.x(t));
    el.ph.style.display = t >= T_END - 0.05 && !st.playing ? 'none' : '';
    for (const r of el.env) r.classList.toggle('past', r._t <= t);
    D.sentences.forEach((s, i) => {
      el.spans[i].classList.toggle('spoken', s.audio_start <= t);
      el.spans[i].classList.toggle('done', s.emit_wall <= t);
    });
    D.triggers.forEach((tr, i) => el.tg[i].classList.toggle('fired', tr.wall <= t));

    // discrete state: rebuild panels only when something changed
    const nDone = D.sentences.filter((s) => s.emit_wall <= t).length;
    const nSpk = D.sentences.filter((s) => s.audio_start <= t && s.emit_wall > t && t < D.audio_s + 0.2).length;
    const dec = D.decisions.filter((d) => d.wall <= t);
    const nTrig = D.triggers.filter((tr) => tr.wall <= t).length;
    const key = [nDone, nSpk, dec.length, nTrig, st.desk, st.tab].join('|');
    if (key === lastKey) return;
    lastKey = key;
    renderTranscript(t, nDone);
    renderDecision(dec[dec.length - 1]);
    renderEvents(t);
    renderDesk();
  }

  function renderTranscript(t, nDone) {
    el.tx.textContent = '';
    const done = D.sentences.slice(0, nDone);
    if (!done.length && !D.sentences.some((s) => s.audio_start <= t)) {
      el.tx.append(h('div', { class: 'empty', text: 'Press Play. Sentences appear here when the streamed transcriber commits them, a few seconds after they are spoken.' }));
      return;
    }
    done.forEach((s, i) => el.tx.append(h('p', { class: i === done.length - 1 ? 'cur' : '' },
      [h('span', { class: 'rp-k', text: '+' + s.emit_wall.toFixed(1) }), h('span', { text: s.text })])));
    if (D.sentences.some((s) => s.audio_start <= t && s.emit_wall > t)) {
      el.tx.append(h('p', { class: 'pending' }, [h('span', { class: 'rp-k', text: '' }), h('span', { text: '… (spoken, not yet committed)' })]));
    }
    el.tx.scrollTop = el.tx.scrollHeight;
    requestAnimationFrame(() => { el.tx.scrollTop = el.tx.scrollHeight; });
  }

  function bar(label, p) {
    const fill = h('div', { class: 'rp-fill', style: `width:${(p * 100).toFixed(1)}%` });
    return h('div', { class: 'rp-row' }, [h('span', { text: label }),
      h('div', { class: 'rp-track', title: 'dashed line: 0.5 threshold' }, [fill, h('span', { class: 'th' })]),
      h('span', { class: 'v', text: fp(p) })]);
  }

  function renderDecision(d) {
    el.dec.textContent = '';
    if (!d) {
      el.dec.append(h('div', { class: 'rp-meta', text: 'Waiting for the first committed sentence.' }));
    } else {
      el.dec.append(h('div', { class: 'rp-meta', text: `sentence ${d.idx + 1} · returned +${d.wall.toFixed(1)} s · ${Math.round(d.ms)} ms` }));
    }
    // stance scale (drawn even before the first decision, without a dot)
    const W = 300, x = (s) => 18 + (W - 36) * (s - 1) / 4;
    const svg = S('svg', { viewBox: `0 0 ${W} 46`, class: 'rp-stance', role: 'img',
      'aria-label': d ? `Stance ${d.stance.toFixed(2)} on a 1 (very dovish) to 5 (very hawkish) scale` : 'Stance scale, no decision yet' });
    svg.append(S('line', { class: 'ln', x1: x(1), x2: x(5), y1: 16, y2: 16 }));
    for (let k = 1; k <= 5; k++) svg.append(S('line', { class: 'tk', x1: x(k), x2: x(k), y1: 11, y2: 21 }));
    svg.append(S('text', { class: 'lb', x: x(1), y: 40, 'text-anchor': 'start' }, 'dove'),
      S('text', { class: 'lb', x: x(5), y: 40, 'text-anchor': 'end' }, 'hawk'));
    if (d) {
      svg.append(S('text', { class: 'lb', x: x(3), y: 40, 'text-anchor': 'middle' }, `${d.stance.toFixed(2)} · ${STANCE[Math.round(d.stance)]}`));
      svg.append(S('circle', { class: 'dot', cx: x(d.stance), cy: 16, r: 6.5 }));
    }
    el.dec.append(svg);
    el.dec.append(h('div', { class: 'rp-row' }, [h('span', { text: 'rate action' }),
      d ? h('span', {}, [h('span', { class: 'rp-chip ' + d.action, text: d.action })]) : h('span', { text: '—' }),
      h('span', { class: 'v', text: d ? fp(d.action_p) : '' })]));
    el.dec.append(bar('≠ prior', d ? d.prior : 0), bar('re-evaluate', d ? d.reprice : 0));
  }

  function renderEvents(t) {
    el.ev.textContent = '';
    D.triggers.forEach((tr, i) => {
      if (tr.wall > t) return;
      const what = tr.kind === 'reprice'
        ? `trigger #${i}: re-evaluate p rose past 0.5 after sentence ${tr.after + 1} → desk run`
        : `trigger #${i}: end of broadcast → desk run`;
      const b = h('button', { type: 'button', 'data-i': String(i) }, [h('span', { class: 'rp-k', text: '+' + tr.wall.toFixed(1) + ' s' }), h('span', { text: what })]);
      b.addEventListener('click', () => openDesk(i));
      el.ev.append(b);
    });
  }

  function spark(m) {
    const W = 300, H = 112, padL = 6, padR = 6, top = 14, bot = 22;
    const t0 = m.t_min[0], t1 = m.t_min[m.t_min.length - 1];
    let lo = Math.min(0, ...m.ret_pct), hi = Math.max(0, ...m.ret_pct);
    const pad = (hi - lo) * 0.12 || 0.1; lo -= pad; hi += pad;
    const x = (v) => padL + (W - padL - padR) * (v - t0) / (t1 - t0);
    const y = (v) => top + (H - top - bot) * (hi - v) / (hi - lo);
    const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, class: 'rp-spark', role: 'img',
      'aria-label': `Intraday return from the 13:55 close: ${sgn(m.m30_pct)}% after 30 minutes, ${sgn(m.close_pct)}% at the close` });
    svg.append(S('line', { class: 'z', x1: padL, x2: W - padR, y1: y(0), y2: y(0) }));
    svg.append(S('line', { class: 'ev', x1: x(0), x2: x(0), y1: 4, y2: H - bot + 2 }));
    const pts = m.t_min.map((tt, i) => [tt, m.ret_pct[i]]);
    const path = (arr) => arr.map((p, i) => (i ? 'L' : 'M') + x(p[0]).toFixed(1) + ' ' + y(p[1]).toFixed(1)).join(' ');
    svg.append(S('path', { class: 'pre', d: path(pts.filter((p) => p[0] <= 0)) }));
    svg.append(S('path', { class: 'pth', d: path(pts.filter((p) => p[0] >= 0)) }));
    const p30 = pts.find((p) => p[0] === 30), pc = pts[pts.length - 1];
    for (const p of [p30, pc]) if (p) svg.append(S('circle', { class: 'pt', cx: x(p[0]), cy: y(p[1]), r: 2.6 }));
    svg.append(S('text', { class: 'lb', x: x(0), y: H - 6, 'text-anchor': 'middle' }, '14:00'),
      S('text', { class: 'lb', x: padL, y: H - 6, 'text-anchor': 'start' }, '12:30'),
      S('text', { class: 'lb', x: W - padR, y: H - 6, 'text-anchor': 'end' }, '16:00'),
      S('text', { class: 'lb', x: W - padR, y: top - 3, 'text-anchor': 'end' }, `${sgn(hi, 1)}%`),
      S('text', { class: 'lb', x: W - padR, y: H - bot - 3, 'text-anchor': 'end' }, `${sgn(lo, 1)}%`));
    return svg;
  }

  function renderDesk() {
    el.desk.textContent = '';
    el.desk.append(h('div', { class: 'rp-h' }, [h('span', { text: 'Trading desk' }),
      h('span', { text: st.desk == null ? '' : `trigger #${st.desk} · ${D.triggers[st.desk].kind === 'reprice' ? 'after sentence ' + (D.triggers[st.desk].after + 1) : 'end of broadcast'}` })]));
    if (st.desk == null) {
      el.desk.append(h('p', { class: 'note', text: 'The desk runs after a trigger fires. One instrument took 88–105 s (8 qwen3:14b calls), so its output arrives after the 52 s broadcast. It opens at the end, or click a trigger.' }));
      return;
    }
    const all = D.desk[String(st.desk)];
    const secs = TICKERS.map((k) => all[k].seconds);
    const tabs = h('div', { class: 'rp-tabs', role: 'tablist', 'aria-label': 'Instrument' });
    TICKERS.forEach((tk) => {
      const b = h('button', { type: 'button', role: 'tab', 'aria-selected': String(tk === st.tab), text: tk });
      b.addEventListener('click', () => { st.tab = tk; render(); });
      b.addEventListener('keydown', (e) => {
        const j = TICKERS.indexOf(st.tab) + (e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0);
        if (j !== TICKERS.indexOf(st.tab)) { st.tab = TICKERS[(j + 3) % 3]; render(); el.desk.querySelector('[aria-selected="true"]').focus(); }
      });
      tabs.append(b);
    });
    tabs.append(h('span', { class: 'sub', text: `ran after the broadcast · ${Math.min(...secs).toFixed(0)}–${Math.max(...secs).toFixed(0)} s per instrument` }));
    el.desk.append(tabs);

    const d = all[st.tab], m = D.market[st.tab];
    const maxS = Math.max(...secs);
    const pipe = h('div', { class: 'rp-pipe', role: 'img', 'aria-label': `Agent pipeline, ${d.seconds} s in total` });
    pipe.style.width = (100 * d.seconds / maxS).toFixed(1) + '%';
    d.nodes.forEach((nd) => pipe.append(h('span', { class: 'rp-n-' + NODE_CLASS[nd.node], style: `width:${(100 * nd.s / d.seconds).toFixed(2)}%`, title: `${nd.node} ${nd.s.toFixed(1)} s` })));
    const leg = h('div', { class: 'rp-pipe-l' });
    d.nodes.forEach((nd) => leg.append(h('span', {}, [h('i', { class: 'rp-n-' + NODE_CLASS[nd.node] }), `${nd.node} ${nd.s.toFixed(1)} s`])));
    leg.append(h('span', { text: `= ${d.seconds.toFixed(1)} s` }));
    const chip = (r) => h('span', { class: 'rp-chip ' + (RATING_CLASS[r] || ''), text: r });
    const ratings = h('div', { class: 'rp-ratings' }, [
      h('span', {}, ['Research mgr', chip(d.rm)]), h('span', {}, ['Trader', chip(d.trader)]),
      h('span', {}, [h('b', { text: 'Portfolio mgr', style: 'font-weight:500;color:var(--ink2)' }), chip(d.pm)])]);
    const ex = h('div', { class: 'rp-ex' });
    [['bull', 'Bull researcher'], ['bear', 'Bear researcher'], ['trader', 'Trader'], ['portfolio_manager', 'Portfolio manager']]
      .forEach(([k, lab]) => ex.append(h('details', {}, [h('summary', { text: lab }), h('p', { text: '“' + d.excerpts[k] + '”' })])));
    const mk = h('div', {}, [h('div', { class: 'rp-h' }, [h('span', { text: `${st.tab} on 16 Sept, % vs 13:55 close` })]), spark(m),
      h('div', { class: 'rp-moves' }, [h('span', {}, ['+30m ', h('b', { text: sgn(m.m30_pct) + '%' })]), h('span', {}, ['close ', h('b', { text: sgn(m.close_pct) + '%' })])])]);
    el.desk.append(pipe, leg, ratings, h('div', { class: 'rp-deskgrid' }, [h('div', {}, [h('div', { class: 'rp-h' }, [h('span', { text: 'Excerpts' })]), ex]), mk]));
  }

  fetch(BASE + 'replay.json').then((r) => r.json()).then((j) => { D = j; build(); render(); })
    .catch(() => { root.textContent = 'Replay data could not be loaded.'; });
})();
