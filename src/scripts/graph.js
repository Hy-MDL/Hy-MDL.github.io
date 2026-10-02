// Knowledge graph panel: notes, series hubs and shared tags as a small force-directed graph.
(function () {
  const host = document.getElementById('kg'); if (!host) return;
  const data = JSON.parse(document.getElementById('kg-data').textContent);
  const cv = host.querySelector('canvas'), ctx = cv.getContext('2d');
  const card = document.getElementById('kg-card');
  const PAL = { light: ['#1F6F7A', '#8A5A2B', '#5B6BB5', '#2E7D4F', '#B0473A', '#7A4E9E', '#56707A'], dark: ['#63B6C3', '#D9A24A', '#8E9BE0', '#6CC08E', '#E08375', '#B48AD6', '#93AEB8'] };
  const isDark = () => document.documentElement.getAttribute('data-theme') === 'dark' || (!document.documentElement.getAttribute('data-theme') && matchMedia('(prefers-color-scheme:dark)').matches);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const seriesIdx = {}; data.series.forEach((s, i) => (seriesIdx[s.id] = i));
  const N = data.nodes.map((n, i) => ({ ...n, i, x: 0, y: 0, vx: 0, vy: 0, r: n.type === 'series' ? 7 : n.type === 'tag' ? 1.9 : 2.6 + Math.min(3, Math.sqrt(n.deg || 0) * 0.8) }));
  const byId = Object.fromEntries(N.map((n) => [n.id, n]));
  const L = data.links.map((l) => ({ s: byId[l.s], t: byId[l.t], kind: l.kind })).filter((l) => l.s && l.t);
  const nb = new Map(N.map((n) => [n.id, new Set()])); L.forEach((l) => { nb.get(l.s.id).add(l.t.id); nb.get(l.t.id).add(l.s.id); });
  let W = 0, H = 0, dpr = 1, alpha = 1, hover = null, sel = null, drag = null, moved = false, raf = 0;
  // seed positions: series hubs on a ring, members near their hub
  data.series.forEach((s, i) => { const a = (i / data.series.length) * Math.PI * 2 - Math.PI / 2, h = byId['series:' + s.id]; if (h) { h.x = Math.cos(a) * 130; h.y = Math.sin(a) * 130; } });
  N.forEach((n, i) => { if (n.type === 'series') return; const h = byId['series:' + n.series]; const a = i * 2.399; n.x = (h ? h.x : 0) + Math.cos(a) * 40; n.y = (h ? h.y : 0) + Math.sin(a) * 40; });
  function size() { const r = host.getBoundingClientRect(), was = W; dpr = Math.min(2, devicePixelRatio || 1); W = r.width; H = r.height; if (!was && W) kick(1); cv.width = W * dpr; cv.height = H * dpr; draw(); }
  function step() {
    if (!W) return;
    for (let i = 0; i < N.length; i++) { const a = N[i]; for (let j = i + 1; j < N.length; j++) { const b = N[j]; let dx = b.x - a.x, dy = b.y - a.y, d2 = dx * dx + dy * dy + 0.01; if (d2 > 40000) continue; const both = a.type === 'series' && b.type === 'series';
        const f = ((both ? 2600 : a.type === 'series' || b.type === 'series' ? 900 : 260) / d2) * alpha; const d = Math.sqrt(d2); dx /= d; dy /= d; a.vx -= dx * f; a.vy -= dy * f; b.vx += dx * f; b.vy += dy * f; } }
    L.forEach((l) => { const rest = l.kind === 'series' ? 58 : l.kind === 'tag' ? 46 : 70, k = (l.kind === 'series' ? 0.05 : l.kind === 'tag' ? 0.02 : 0.012) * alpha; let dx = l.t.x - l.s.x, dy = l.t.y - l.s.y, d = Math.sqrt(dx * dx + dy * dy) || 1, f = (d - rest) * k; dx /= d; dy /= d; l.s.vx += dx * f; l.s.vy += dy * f; l.t.vx -= dx * f; l.t.vy -= dy * f; });
    const lim = Math.min(W, H) / 2 - 14;
    N.forEach((n) => { n.vx -= n.x * 0.004 * alpha; n.vy -= n.y * 0.004 * alpha; if (n === drag) { n.vx = n.vy = 0; return; } n.vx *= 0.82; n.vy *= 0.82; n.x += n.vx; n.y += n.vy; const d = Math.hypot(n.x, n.y), m = n.type === 'series' ? lim - 22 : lim; if (d > m) { n.x *= m / d; n.y *= m / d; } if (n.type === 'series' && n.y > H / 2 - 34) n.y = H / 2 - 34; });
    alpha = Math.max(0.02, alpha * 0.978);
  }
  function color(n, pal) { return n.type === 'tag' ? css('--rule') : pal[seriesIdx[n.series] % pal.length]; }
  function draw() {
    const pal = isDark() ? PAL.dark : PAL.light, ink = css('--ink'), rule = css('--rule2'), bg = css('--bg2');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H); ctx.translate(W / 2, H / 2);
    const focus = sel || hover, near = focus ? nb.get(focus.id) : null;
    L.forEach((l) => { const on = focus && (l.s === focus || l.t === focus);
      ctx.globalAlpha = focus ? (on ? 0.92 : 0.04) : l.kind === 'cite' ? 0.30 : 0.13;
      if (on) ctx.strokeStyle = color(focus, pal);
      else { const g = ctx.createLinearGradient(l.s.x, l.s.y, l.t.x, l.t.y); g.addColorStop(0, color(l.s, pal)); g.addColorStop(1, color(l.t, pal)); ctx.strokeStyle = g; }
      ctx.lineWidth = on ? 1.2 : 0.7; ctx.beginPath(); ctx.moveTo(l.s.x, l.s.y); ctx.lineTo(l.t.x, l.t.y); ctx.stroke(); });
    const dark = isDark();
    N.forEach((n) => { const on = !focus || n === focus || near.has(n.id); const c = color(n, pal);
      const base = on ? 1 : 0.10, R = n === focus ? n.r + 1.2 : n.r;
      if (n.type !== 'tag') {                                     // bloom: a soft halo the dot sits in
        const bloom = (n === focus ? 3.6 : 2.5) * R;
        const g = ctx.createRadialGradient(n.x, n.y, R * 0.6, n.x, n.y, bloom);
        g.addColorStop(0, c); g.addColorStop(1, 'transparent');
        ctx.globalAlpha = base * (n === focus ? 0.34 : dark ? 0.20 : 0.12);
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(n.x, n.y, bloom, 0, 6.2832); ctx.fill();
      }
      ctx.globalAlpha = base;
      if (n.type === 'series') {                                  // hub: ring with a small core
        ctx.beginPath(); ctx.arc(n.x, n.y, R, 0, 6.2832); ctx.fillStyle = bg; ctx.fill();
        ctx.lineWidth = 1.6; ctx.strokeStyle = c; ctx.stroke();
        ctx.beginPath(); ctx.arc(n.x, n.y, R * 0.34, 0, 6.2832); ctx.fillStyle = c; ctx.fill();
      } else {
        ctx.beginPath(); ctx.arc(n.x, n.y, R, 0, 6.2832); ctx.fillStyle = c; ctx.fill();
      } });
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const placed = [];
    const fits = (x, y, w, h) => { for (const b of placed) if (x < b.x + b.w && x + w > b.x && y < b.y + b.h && y + h > b.y) return false; placed.push({ x, y, w, h }); return true; };
    const order = N.filter((n) => n.type === 'series' || n === focus || (focus && near.has(n.id) && n.type !== 'tag') || (sel && near.has(n.id)))
                   .sort((a, b) => (b === focus) - (a === focus) || (b.type === 'series') - (a.type === 'series'));
    order.forEach((n) => { const show = true; if (!show) return; ctx.globalAlpha = focus && n !== focus && !near.has(n.id) ? 0.2 : 1; ctx.font = (n.type === 'series' ? '500 11.5px ' : '10.5px ') + '"IBM Plex Sans",sans-serif'; const t = n.label.length > 26 ? n.label.slice(0, 25) + '…' : n.label; const w = ctx.measureText(t).width;
      const lx = Math.max(-W / 2 + w / 2 + 9, Math.min(W / 2 - w / 2 - 9, n.x));   // keep the chip inside the canvas
      const ly = Math.min(H / 2 - 20, n.y + n.r + 4);
      if (!fits(lx - w / 2 - 5, ly, w + 10, 15)) return; ctx.fillStyle = bg; ctx.globalAlpha *= 0.9; ctx.beginPath(); ctx.roundRect(lx - w / 2 - 5, ly, w + 10, 15, 7.5); ctx.fill(); ctx.globalAlpha *= 0.5; ctx.strokeStyle = rule; ctx.lineWidth = 0.7; ctx.stroke(); ctx.globalAlpha = focus && n !== focus && !near.has(n.id) ? 0.2 : 1; ctx.fillStyle = n.type === 'series' ? color(n, pal) : ink; ctx.fillText(t, lx, ly + 2); });
    ctx.globalAlpha = 1;
  }
  function loop() { step(); draw(); raf = alpha > 0.06 || drag ? requestAnimationFrame(loop) : 0; }
  function kick(a) { alpha = Math.max(alpha, a); if (!raf) raf = requestAnimationFrame(loop); }
  function pick(e) { const r = cv.getBoundingClientRect(), x = e.clientX - r.left - W / 2, y = e.clientY - r.top - H / 2; let best = null, bd = 1e9; N.forEach((n) => { const d = Math.hypot(n.x - x, n.y - y) - n.r; if (d < 9 && d < bd) { bd = d; best = n; } }); return { n: best, x, y }; }
  function showCard(n) {
    if (!n) { card.innerHTML = '<span class="kg-hint">Drag a node · click to see its neighbours · click a series ring to filter the list</span>'; return; }
    const links = [...nb.get(n.id)].map((id) => byId[id]); const notes = links.filter((m) => m.type === 'post'), tags = links.filter((m) => m.type === 'tag');
    if (n.type === 'post') card.innerHTML = '<b>' + n.title + '</b><span class="kg-meta">' + n.seriesLabel + ' · linked to ' + notes.length + ' notes' + (tags.length ? ' · ' + tags.map((t) => '#' + t.label).join(' ') : '') + '</span><a href="' + n.url + '" data-open="' + n.slug + '">Open note →</a>';
    else if (n.type === 'series') card.innerHTML = '<b>' + n.label + '</b><span class="kg-meta">' + notes.length + ' notes · ' + n.blurb + '</span>';
    else card.innerHTML = '<b>#' + n.label + '</b><span class="kg-meta">shared by ' + notes.length + ' notes: ' + notes.slice(0, 6).map((m) => m.label).join(', ') + (notes.length > 6 ? '…' : '') + '</span>';
  }
  cv.addEventListener('pointerdown', (e) => { const p = pick(e); moved = false; if (p.n) { drag = p.n; cv.setPointerCapture(e.pointerId); kick(0.5); } });
  cv.addEventListener('pointermove', (e) => { const p = pick(e); if (drag) { drag.x = p.x; drag.y = p.y; moved = true; kick(0.35); return; } if (p.n !== hover) { hover = p.n; cv.style.cursor = hover ? 'pointer' : 'default'; if (!sel) showCard(hover); if (!raf) draw(); } });
  cv.addEventListener('pointerup', (e) => { const p = pick(e), was = drag; drag = null; if (was && !moved) { sel = sel === was ? null : was; showCard(sel); if (was.type === 'series') { const b = document.querySelector('.filters button[data-f="' + (sel ? was.series : 'all') + '"]'); if (b) b.click(); } } else if (!was && !p.n) { sel = null; showCard(null); const b = document.querySelector('.filters button[data-f="all"]'); if (b) b.click(); } kick(0.12); });
  cv.addEventListener('pointerleave', () => { if (!drag) { hover = null; if (!sel) showCard(null); if (!raf) draw(); } });
  cv.addEventListener('dblclick', (e) => { const p = pick(e); if (p.n && p.n.type === 'post') (window.__openNote || ((slug, url) => (location.href = url)))(p.n.slug, p.n.url); });
  card.addEventListener('click', (e) => { const a = e.target.closest('a[data-open]'); if (a && window.__openNote) { e.preventDefault(); window.__openNote(a.dataset.open, a.getAttribute('href')); } });
  new ResizeObserver(size).observe(host); document.addEventListener('themechange', draw);
  new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  size(); showCard(null);
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) { for (let i = 0; i < 700; i++) step(); alpha = 0.02; draw(); } else { for (let i = 0; i < 600; i++) step(); alpha = 0.02; kick(0.10); }
})();
