// Interactive board view: pick a role on the photo, swap it from Air-like to XPS-like, read the 3-year health.
// Numbers come from data.json (v7 model, one-at-a-time swaps); boxes from the annotated iFixit figures.
(function () {
  const root = document.getElementById('anatomy');
  if (!root) return;
  const BASE = '/projects/battery-os-efficiency/anatomy/';
  const ROLES = {
    1: { name: 'SoC', what: 'active power', color: '#6A3D9A',
         air: '3.1 W web use', xps: '4.75 W web use',
         why: 'More watts for the same work means more of the pack is cycled each day (0.32 vs 0.49 full cycles).' },
    2: { name: 'PMICs', what: 'standby drain', color: '#1F77B4',
         air: '0.04 W asleep', xps: '0.40 W asleep',
         why: 'Sleep drain is small per hour, but a laptop sleeps most of the day, so it adds cycles too.' },
    3: { name: 'Charge path', what: 'charge hold', color: '#D97706',
         air: 'holds near 80\u00a0%', xps: 'holds at 100\u00a0%',
         why: 'Sitting full on the charger ages the cell faster; Optimized Battery Charging pauses at 80\u00a0%.' },
    4: { name: 'Pack', what: 'cell temperature', color: '#1B9E77',
         air: 'pack near the SoC', xps: 'slightly cooler',
         why: 'The only factor that goes the other way: the XPS pack runs a little cooler in the mapping.' },
  };
  const VIEWS = { front: 'Board · front', back: 'Board · back', chassis: 'Underside' };
  const st = { view: 'front', year: '3', swap: { 1: false, 2: false, 3: false, 4: false }, focus: null };
  let D = null;

  const h = (tag, attrs = {}, kids = []) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) k === 'text' ? (e.textContent = v) : e.setAttribute(k, v);
    for (const c of [].concat(kids)) if (c) e.append(c);
    return e;
  };
  const S = (tag, attrs = {}) => {
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  };
  const pp = (x) => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(2) + ' pp';

  function mix() {
    const d = D.decomp[st.year];
    const on = Object.keys(st.swap).filter((r) => st.swap[r]);
    let v = d.air + on.reduce((s, r) => s + d.eff[r], 0);
    if (on.length === 4) v = d.xps;               // all four on: the full XPS-like run, interactions included
    return v;
  }

  function stage() {
    const v = D.views[st.view];
    const wrap = h('div', { class: 'an-stage' });
    wrap.append(h('img', { src: BASE + v.img, alt: VIEWS[st.view] + ' of the M1 MacBook Air, iFixit photo', loading: 'lazy' }));
    const svg = S('svg', { viewBox: `0 0 ${v.w} ${v.h}`, class: 'an-svg', role: 'group' });
    for (const p of D.parts.filter((q) => q.view === st.view)) {
      const r = ROLES[p.role], [x, y, w, hh] = p.box, pad = 4;
      const dim = st.focus && st.focus !== p.role;
      const g = S('g', { class: 'an-part' + (dim ? ' dim' : '') + (st.focus === p.role ? ' on' : ''), tabindex: 0 });
      g.append(S('rect', { x: x - pad, y: y - pad, width: w + 2 * pad, height: hh + 2 * pad, rx: 3,
        fill: r.color, 'fill-opacity': st.swap[p.role] ? 0.38 : 0.16, stroke: r.color,
        'stroke-width': st.focus === p.role ? 3.2 : 2, 'vector-effect': 'non-scaling-stroke', 'stroke-dasharray': p.approx ? '8 5' : 'none' }));
      const t = S('title'); t.textContent = `${p.label}${p.approx ? ' (approx. position)' : ''} · ${r.name}`; g.append(t);
      g.addEventListener('click', () => { st.focus = st.focus === p.role ? null : p.role; render(); });
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter') g.dispatchEvent(new Event('click')); });
      svg.append(g);
    }
    wrap.append(svg);
    return wrap;
  }

  function roleRow(id) {
    const r = ROLES[id], eff = D.decomp[st.year].eff[id], on = st.swap[id];
    const btn = h('button', { class: 'an-swap' + (on ? ' on' : ''), 'aria-pressed': on, type: 'button' },
      [h('span', { class: 'an-sw-a', text: 'Air' }), h('span', { class: 'an-sw-x', text: 'XPS' })]);
    btn.addEventListener('click', (e) => { e.stopPropagation(); st.swap[id] = !on; st.focus = Number(id); render(); });
    const row = h('div', { class: 'an-row' + (st.focus === Number(id) ? ' focus' : ''), style: `--rc:${r.color}` }, [
      h('div', { class: 'an-name' }, [h('b', {}, [h('i'), r.name]), h('span', { text: r.what })]),
      h('div', { class: 'an-vals', text: (on ? r.xps : r.air) }),
      h('div', { class: 'an-eff' + (on ? ' live' : ''), text: pp(eff) }),
      btn,
    ]);
    row.addEventListener('click', () => {
      st.focus = Number(id);
      if (!D.parts.some((p) => p.view === st.view && p.role === st.focus))
        st.view = D.parts.find((p) => p.role === st.focus).view;
      render();
    });
    return row;
  }

  function chart() {
    const W = 520, H = 150, L = 34, R = 10, T = 10, B = 24;
    const t = D.traj, x = (yr) => L + (yr / 5) * (W - L - R), y = (v) => T + ((100 - v) / 30) * (H - T - B);
    const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, class: 'an-chart' });
    for (const g of [100, 90, 80, 70]) {
      svg.append(S('line', { x1: L, x2: W - R, y1: y(g), y2: y(g), class: 'an-grid' }));
      const tx = S('text', { x: L - 6, y: y(g) + 4, 'text-anchor': 'end', class: 'an-ax' }); tx.textContent = g + '%'; svg.append(tx);
    }
    for (let yr = 0; yr <= 5; yr++) {
      const tx = S('text', { x: x(yr), y: H - 6, 'text-anchor': 'middle', class: 'an-ax' }); tx.textContent = yr + ' y'; svg.append(tx);
    }
    const path = (arr) => arr.map((v, i) => (i ? 'L' : 'M') + x(t.years[i]).toFixed(1) + ' ' + y(v).toFixed(1)).join('');
    svg.append(S('path', { d: path(t.air), class: 'an-l an-air' }));
    svg.append(S('path', { d: path(t.xps), class: 'an-l an-xps' }));
    const yr = Number(st.year), m = mix();
    svg.append(S('line', { x1: x(yr), x2: x(yr), y1: T, y2: H - B, class: 'an-cursor' }));
    svg.append(S('circle', { cx: x(yr), cy: y(m), r: 5.5, class: 'an-dot' }));
    return svg;
  }

  function result() {
    const d = D.decomp[st.year], m = mix(), n = Object.values(st.swap).filter(Boolean).length;
    const line = n === 0 ? 'Everything set like the Air. Swap a part to the XPS value and watch the dot.'
      : n === 4 ? 'Every part set like the XPS: the full XPS-like run.'
      : `${n} part${n > 1 ? 's' : ''} swapped: ${pp(m - d.air)} against the Air-like run.`;
    return h('div', { class: 'an-result' }, [
      h('div', { class: 'an-nums' }, [
        h('div', {}, [h('small', { text: 'Air-like' }), h('b', { class: 'c-air', text: d.air.toFixed(1) + '%' })]),
        h('div', {}, [h('small', { text: 'your mix' }), h('b', { class: 'c-mix', text: m.toFixed(1) + '%' })]),
        h('div', {}, [h('small', { text: 'XPS-like' }), h('b', { class: 'c-xps', text: d.xps.toFixed(1) + '%' })]),
      ]),
      chart(),
      h('p', { class: 'an-line', text: line }),
    ]);
  }

  function seg(opts, cur, set) {
    const g = h('div', { class: 'an-seg', role: 'tablist' });
    for (const [k, label] of opts) {
      const b = h('button', { type: 'button', class: k === cur ? 'on' : '', 'aria-selected': k === cur, text: label });
      b.addEventListener('click', () => { set(k); render(); });
      g.append(b);
    }
    return g;
  }

  function render() {
    const f = st.focus ? ROLES[st.focus] : null;
    root.replaceChildren(
      h('div', { class: 'an-bar' }, [
        seg(Object.entries(VIEWS), st.view, (k) => (st.view = k)),
        seg([['2', '2 y'], ['3', '3 y'], ['4', '4 y']], st.year, (k) => (st.year = k)),
      ]),
      stage(),
      h('p', { class: 'an-why' + (f ? '' : ' idle'), style: f ? `--rc:${f.color}` : '' },
        f ? [h('b', { text: f.name + '. ' }), f.why] : 'Tap a box or a row to see what that part does to the battery.'),
      h('div', { class: 'an-grid2' }, [
        h('div', { class: 'an-rows' }, [
          h('div', { class: 'an-head' }, [h('span', { text: 'part' }), h('span', { text: 'value' }),
            h('span', { text: `at ${st.year} y` }), h('span', { text: 'set like' })]),
          ...Object.keys(ROLES).map(roleRow),
        ]),
        result(),
      ]),
      h('p', { class: 'an-foot', text: 'Your mix adds the single-swap effects you turn on to the Air-like run (interactions, under 0.1 pp, only enter when all four are swapped). ' +
        D.credit + '; boxes, roles and their link to ageing are my own analysis, not iFixit’s.' }),
    );
  }

  fetch(BASE + 'data.json').then((r) => r.json()).then((d) => { D = d; render(); })
    .catch(() => { root.textContent = 'The interactive view could not load.'; });
})();
