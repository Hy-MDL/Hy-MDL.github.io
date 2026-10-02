import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const writing = defineCollection({
  loader: glob({ pattern: '*/index.md', base: './src/content/writing', generateId: ({ entry }) => entry.split('/')[0] }),
  schema: z.object({
    title: z.string(),
    paper: z.object({
      title: z.string(),
      authors: z.string(),
      venue: z.string(),
      arxiv: z.coerce.string(),
      url: z.string().optional(),
      pdf: z.string().optional(),
      license: z.string(),
    }),
    series: z.enum(['score-to-flow', 'normalizing-flows', 'generative-finance', 'eswa-finance', 'stochastic-modeling', 'sequential-monte-carlo', 'submodular-optimization', 'surrogates-bo', 'vision', 'industrial-vision', 'ee-timeseries']),
    order: z.number(),
    tags: z.array(z.string()).default([]),
    date: z.coerce.date(),
    status: z.enum(['draft', 'published']).default('draft'),
    summary: z.string(),
  }),
});

const projects = defineCollection({
  loader: glob({ pattern: '*/index.md', base: './src/content/projects', generateId: ({ entry }) => entry.split('/')[0] }),
  schema: z.object({
    title: z.string(),
    slug: z.string(),
    category: z.enum(['Computer Vision', 'Stochastic Modeling', 'Simulation Optimization', 'Finance AI', 'Developer Tools', 'Green AI', 'Healthcare Operations', 'Platform', 'Web', 'Generative AI']),
    summary: z.string(),
    period: z.string(),
    status: z.string(),
    stack: z.array(z.string()).default([]),
    tags: z.array(z.string()).default([]),
    metrics: z.array(z.object({ label: z.string(), value: z.string(), note: z.string().optional() })).default([]),
    code: z.string().optional(),
    order: z.number().default(50),
    kind: z.enum(['research', 'project']).default('project'),
    thumb: z.string().optional(),
    scope: z.enum(['personal', 'lab', 'course']).default('personal'),
  }),
});

export const collections = { writing, projects };
