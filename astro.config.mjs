import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// wrap every rendered table in a scrollable card so results stand apart from the prose
// A paragraph whose whole content is one italic line ("*Source: ...*") is a caption; tag it so the
// caption style applies only there, not to an ordinary sentence that happens to contain one <em>.
function rehypeEmCaption() {
  return (tree) => {
    const visit = (node) => {
      if (node.type === 'element' && node.tagName === 'p') {
        const kids = (node.children || []).filter((c) => !(c.type === 'text' && !c.value.trim()));
        if (kids.length === 1 && kids[0].type === 'element' && kids[0].tagName === 'em') {
          node.properties = node.properties || {};
          const cls = node.properties.className;
          node.properties.className = [...(Array.isArray(cls) ? cls : cls ? [cls] : []), 'em-caption'];
        }
      }
      (node.children || []).forEach(visit);
    };
    visit(tree);
  };
}

// On project pages, a paragraph holding only an image whose alt text is a numbered caption ("Figure 3 — ...")
// becomes a <figure> with that caption shown underneath, as on the hand-built paper pages.
function rehypeFigureCaption() {
  return (tree, file) => {
    const where = String((file && (file.path || (file.history || [])[0])) || '');
    if (!where.includes('/src/content/projects/')) return;
    const walk = (node) => {
      if (!node.children) return;
      node.children = node.children.map((c) => {
        walk(c);
        if (c.type !== 'element' || c.tagName !== 'p') return c;
        const kids = (c.children || []).filter((k) => !(k.type === 'text' && !k.value.trim()));
        if (kids.length !== 1 || kids[0].type !== 'element' || kids[0].tagName !== 'img') return c;
        const m = String((kids[0].properties || {}).alt || '').match(/^(Figure\s+\d+[a-z]?)\s*[—:.-]\s*([\s\S]+)$/);
        if (!m) return c;
        return { type: 'element', tagName: 'figure', properties: { className: ['mdfig'] }, children: [kids[0],
          { type: 'element', tagName: 'figcaption', properties: {}, children: [
            { type: 'element', tagName: 'b', properties: {}, children: [{ type: 'text', value: m[1] + '.' }] },
            { type: 'text', value: ' ' + m[2] }] }] };
      });
    };
    walk(tree);
  };
}

function rehypeTableCard() {
  return (tree) => {
    const walk = (node) => {
      if (!node.children) return;
      node.children = node.children.map((c) => {
        walk(c);
        if (c.type === 'element' && c.tagName === 'table') {
          return { type: 'element', tagName: 'div', properties: { className: ['tblwrap'] }, children: [c] };
        }
        return c;
      });
    };
    walk(tree);
  };
}


export default defineConfig({
  site: 'https://hy-mdl.github.io',
  redirects: { '/writing': '/blog', '/writing/[id]': '/blog/[id]', '/projects/finphasor': '/research/finphasor', '/projects/diffusion-scenarios': '/research/diffusion-scenarios', '/projects/tailflow': '/research/tailflow', '/projects/model-uncertainty-priors': '/research/model-uncertainty-priors', '/projects/rs-lab': '/research/rs-lab', '/projects/input-uncertainty-select': '/research/input-uncertainty-select', '/projects/commons': '/projects/lucubra' },
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex, rehypeTableCard, rehypeEmCaption, rehypeFigureCaption],
    syntaxHighlight: { type: 'shiki', excludeLangs: ['mermaid', 'math'] },
    shikiConfig: { themes: { light: 'github-light', dark: 'github-dark' } },
  },
});
