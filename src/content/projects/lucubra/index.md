---
title: "Lucubra — a research-agent platform where you talk to agents that verify"
slug: lucubra
category: "Platform"
summary: "Lucubra is a web app I built for my own research: bring a paper or a draft, and agents check the claims against the numbers, replicate what can be replicated, review the draft the way a referee for the target journal would, and help write the next version, while every decision stays with me. Built in four days at the end of September 2026, partly because I wanted the tool and partly to find out whether I could actually build an LLM agent system end to end: tool loops, structured outputs, personas, an MCP server, guards. Not public."
period: "2026.09.27 – (in daily use)"
status: "In progress"
stack: [Next.js 16, React, TypeScript, SQLite, "@anthropic-ai/sdk", MCP, OpenAlex, arXiv]
tags: [llm-agents, research-tools, agent-orchestration, peer-review, mcp]
metrics:
  - { label: "Build", value: "4 days to first use · 57 numbered rounds by 2026.10.02", note: "each round is a written brief an agent implements and a script verifies; the structure was rebuilt three times before it made sense" }
  - { label: "Agent tools · built-in personas · MCP tools", value: "11 · 25 · 11", note: "research APIs the conversation calls in a tool loop; referees, editors, peers and coaches assembled into committees; what Claude Code or Codex can drive from outside" }
  - { label: "Verification scripts", value: "141 Playwright flows · 30 HTTP smoke checks", note: "every page, every draft stage, the review ledger, the MCP server, security settings; rerun after each round" }
  - { label: "Cost without an API key", value: "$0", note: "runs on the local Claude Code CLI, or falls back to a scripted demo" }
code: "commons"
order: 5
kind: project
thumb: "/projects/lucubra/media/home.jpg"
---

## In short

**Where it started.** Two manuscripts under review taught me what I wanted and did not have: something that reads a draft the way a hostile referee would, checks the numbers instead of admiring the prose, knows what the target journal's recent papers actually look like, and leaves the decision to me. The agent reviewers that exist are built around arXiv machine-learning papers, and empirical finance has its own failure surface (a factor that "works" at t = 2, a backtest without costs, a signal that uses tomorrow's data). I also wanted to know whether I could build an LLM agent system end to end, not just call a model.

**What I learned building it.** Four days to the first usable version, and the structure was thrown away and rebuilt three times when I could not tell what the product was. What held was the process: each change is a numbered round with a written brief that Claude Code or Codex implements and scripts verify — 57 rounds, 141 Playwright flows and 30 HTTP smoke checks by 2026.10.02.

**What that made me curious about.** Whether I could keep strengthening a program through agents without losing control of it, and whether an agent review could converge instead of drifting from round to round.

**What worked, and what did not.** The review ledger works: later rounds may only close items or add what the revision itself introduced, so the number of open essentials can only fall. I use it on my own drafts every day. But a feature audit on 2026.10.02 marked several landing scenes as only partly built or not built yet, and the committee still gets things wrong: a point raised twice, a baseline axis that should not apply, a persona that reads a finance paper like an ML paper. Each of those becomes the next round's brief.

**Where it leads.** Calibrate the committee against real decisions so that "2 things to change" is a measured prediction rather than a persona's opinion; agent control on a budget; multi-user hosting with local execution kept off; and long-running workflows that reimplement a paper's method on new data end to end.

## The landing page, section by section

Lucubra is the name the platform carries now; its window title reads "Lucubra — research, verified in conversation", and the landing's palette is called lamplight. The public landing page is the outermost layer, the first thing anyone sees before signing in, so this page starts there: one capture per section of the landing page, top to bottom, taken on 2026.10.02 at 1440 × 900.

Two cautions about what the captures show. The landing's scenes run on one illustrative paper (a momentum-crash study with made-up numbers) and the landing says so; none of the numbers in them are results. And the landing is the *specification* of the product as much as a description of it: a feature audit of the app on 2026.10.02 marked several of the scenes as only partly built or not built yet, among them the Reader that links claims to tables with line references, fixes applied in place to the TeX source, and the Codex, Ollama and ORCID connections. Those scenes are being built next; the app screens further down show what runs today.

![Hero. "Research beyond the page.": bring a paper, follow the idea, from understanding to implementation to a sharper review. The card on the right is the promise in one example: a table that was read, reimplemented and run on your own data (0.97 reported, 0.95 yours, within tolerance), with what is still open and who in the community holds the missing data. Underneath, the five stages that organise the product: Read, Ideate, Build, Reimplement, Review.](./figs/landing-hero.jpg)

![Overview. "One workspace for the whole paper." The landing draws the product's own project screen, with numbered hotspots on the real controls: the sidebar (Chat, Projects, Library, Venues, Community, Reviews), a project with its target journal and deadline, the draft's stage rail (Premise → Write → Review → Submit), its data, the code built for it with a Run button, the results, and the conversation beside it where a methods referee re-runs the estimation window and updates the draft.](./figs/landing-overview.jpg)

![Tour. "From the first read to the next revision.": five stages, each scrubbed by scrolling and each ending in something you can open again. Shown here is stage 01, Read: a PDF goes in, and its structure comes out linked: three claims, each tied to the section and table it rests on, with the table cell highlighted. The other stages are Ideate (three agents return directions to choose from), Build (named figure and table styles, a TeX source and a live preview), Reimplement (method summary and a data plan matched against the project) and Review.](./figs/landing-tour.jpg)

![Planning and baselines. Pick a venue and the plan works back from its deadline, section by section; the checklist scores the draft against what recent papers at that venue report (length, exhibits, robustness checks, abstract), gives an outlook that is labelled "not a promise", and turns the gap into a planned item.](./figs/landing-planning.jpg)

![A committee of specialists. Skeptic, math, consistency and layout agents each run their own checks in parallel and post one finding to change: a claim with no table behind it, a sign flip between two equations, a number in the text that disagrees with Table 1 (opened here with the quoted line and a proposed fix), a figure placed pages after its first mention.](./figs/landing-committee.jpg)

![Checks. "Better questions. More careful checks.": the finance-specific failure surface the referees look for: factor spanning, multiple testing, backtest overfitting, the deflated Sharpe ratio, look-ahead bias, survivorship, transaction costs, and reference resolution.](./figs/landing-checks.jpg)

Between these, the landing also has a statement block (built for finance and machine-learning researchers, with agents in the loop from the first question to the referee's letter), a feature index, two more scrubbed scenes ("Write, run, and submit": a referee asks for the code, the run regenerates the exhibits; "Models, accounts and connections": Claude Code, Codex, Ollama and ORCID connected, one agent per role), an FAQ and a closing call to action.

## Why I built it

Two manuscripts under review taught me what I wanted and did not have: something that reads a draft the way a hostile referee would, checks the numbers instead of admiring the prose, knows what the target journal's recent papers actually look like, and leaves the decision to me. The agent reviewers that exist are built around arXiv machine-learning papers; empirical finance has its own failure surface (a factor that "works" at t = 2, a backtest without costs, a signal that uses tomorrow's data) and none of them speak it. So Lucubra is a research workspace where you bring a paper or a question and agents verify, replicate, review and draft, and people decide. It is built for my own work first, and I use it on my own drafts every day now.

The second reason is the one I would state in an interview: I wanted to know whether I could **build an LLM agent system**, not just call a model, and then whether I could keep strengthening a program through agents without losing control of it. Four days to the first usable version; the structure was thrown away and rebuilt three times when I could not tell what the product was. That was the point.

## Inside the app, screen by screen

Behind the landing page, after sign-in, the app itself, captured on 2026.10.02 after the rename and the lamplight restyle. The sidebar lists real working projects and a few test ones.

![Home: the conversation is the front door. One message box with the model picked per run (here Claude Haiku 4.5 through the local Claude Code), suggested questions underneath, and "Pick up where you left off": recent conversations and drafts with their venue and stage.](./figs/home.jpg)

**Chat** is the front door. A question, a paper, or an idea goes into one box; agents answer with cards that carry their reason and their source, and whatever they build (code, a data plan, draft LaTeX, notes) lands beside the thread as a versioned artifact. ⌘K opens a new conversation from anywhere; ⌘/ searches venues, drafts, papers and personas. "Open venues" or "go to review" typed in the conversation navigates the app.

**Projects** hold one research question each, with the journal it aims at, its drafts, data and everything built for it. A draft starts at **Ideate** (question, gap, pre-registered hypotheses), then moves along the four-stage rail **Premise** (the venue sets length, table style and figure spec for every cell) → **Write** → **Review** → **Submit**, and every agent result is kept with the draft.

![Write: the draft as LaTeX on the left of a live preview, sections listed by state, named figure and table styles, versions, and "Write all", which sends each section to its own agent team; bracketed placeholders stay highlighted in the preview until a result fills them. The stage rail above runs Premise → Write → Review → Submit.](./figs/draft-write.jpg)

![Review: the committee lanes (Skeptic, Math, Consistency, Layout and the venue's Referees) each post findings with an id, a severity and an anchor in the text (here two FIX items marked major and four SUG items marked minor); the source is shown read-only beside them, and the right column holds the outlook and a checklist scored against what recent papers at the target venue report. A banner asks for a re-run because the source changed since round 1.](./figs/draft-check.jpg)

**Review** is the part that started everything. The venue's committee of referee personas writes reports in parallel and an editor synthesises them into a verdict against the venue's *measured baseline*: what recent papers at that journal actually report (length, references, figures, sample length, cross-section size, robustness checks, costs, out-of-sample tests, multiple-testing corrections), each as a median and interquartile range from OpenAlex. A **review ledger** keeps the rounds honest: round one raises the blocking set (defects with quotes, gaps on below-baseline axes, capped in number); later rounds may only close items, add defects the revision itself introduced, or add gaps on axes that became measurable. A closed point is never raised again, so the number of open essentials can only fall and reviews converge instead of drifting.

![Venues: the whole OpenAlex source index (every journal and conference in 26 fields), searchable and ranked within its field, with impact-factor approximations, a quartile derived from the h-index rank, deadlines, LaTeX templates and the referee committee for each.](./figs/venues.jpg)

**Venues** answers the questions a graduate student otherwise looks up one by one: where a paper like this one goes (the 100 most relevant recent articles and where they appeared), how a journal ranks in its field, what it expects, when it is due. "Set as target" gives a draft a deadline and a schedule worked back from it.

![Analysis: Python runs in a Web Worker in the browser (Pyodide: numpy, pandas, matplotlib, scipy, statsmodels) with no access to the site's API; an agent writes the cell, the person runs it, the figures stay with the run, and a "Finance checks" button runs spanning, deflated-Sharpe, PBO, net-of-costs and look-ahead checks. Heavy jobs go to a server over SSH.](./figs/analysis.jpg)

![Community: a feed where every post links the paper, draft or venue it is about, with tabs for following and for data requests; people by interest; and a researcher search over OpenAlex by topic, institution and name. House rules on the right: agents post only when a person asks them to, and are labelled.](./figs/community.jpg)

**Library** is what is new on arXiv for your interests plus the classics with illustrative artifacts; **Reviews** is the workspace for papers you referee for others; **You** holds the profile, advisors, careers, archive, integrations and settings.

## How the agents are controlled

This is the part I care most about, because an agent system that cannot be steered is not a tool.

- **Roles, with a model each.** Mock referee, replication, implementer, writing coach and conversation are separate roles; Settings picks the model per role (Opus, Sonnet, Haiku, or Claude Code on the machine) and each run is priced and logged. Three ways to run: a hosted plan, your own API key, or the local Claude Code CLI, plus a scripted fallback so the whole app works with nothing. In practice it is the third: 64 of the first 66 live calls went through the local Claude Code CLI.
- **Skills: standing instructions.** Like skills in Claude Code, a skill is a short rule set added to the system prompt of every run in its area: *finance referee rigor* (look-ahead, survivorship, the significance hurdle, how many specifications were tried), *figure and table spec* (colour-blind-safe, one colour per method, booktabs), *academic English*, *replication package*, *citation guard*. Built-ins switch on and off; you can write your own or paste a SKILL.md.
- **Personas and committees.** Twenty-five built-in personas (fourteen referees, four editors, five peers such as a senior colleague, a quant or a skeptical theorist, and two coaches), journal- and conference-specific, assembled into a committee per venue; you can add your own advisor as one.
- **Tools in a loop.** The conversation calls research APIs as tools: arXiv search, OpenAlex works, authors and professors, venue fit, venue baseline, Crossref, a connected GitHub repository, FRED, the Ken French library, and any MCP server you connect. Plugins add tools with one click; the agent sees only the tools its plugins expose.
- **The other direction: Lucubra as an MCP server.** With a personal access token, Claude Code or Codex can drive Lucubra from outside: list submissions, read open issues, pull a venue's guidance and baseline, run the referee or the replication checks on a manuscript, read runs and balances. The platform I use to check my papers is itself a tool my coding agents can call.
- **Guards.** Manuscripts are sent to the model as untrusted documents under a system prompt that says so. Dataset links are fetched only from public addresses with every redirect re-checked. Local code execution is off by default, owner-only, runs in a fresh directory under a macOS sandbox that cannot read the repository or the database, and sees an allow-listed environment with no secrets. Rate limits on sign-in and sign-up, a Content-Security-Policy on every response, API keys encrypted at rest.

## How it is built, and how it keeps getting stronger

The build process is itself an agent process, and it is the reason this project exists on this site. Each change is a **numbered round with a written brief**: what the user said after the last build, what is on disk, what to change, what to verify and how. The brief is handed to Claude Code or Codex, which implements it, runs the type check and lint, and reports measured numbers back. Round 56's brief, dated 2026.10.01, is the landing page's product overview in the second capture above: the exact palette tokens, the five hotspots on real controls, the DOM measurements the agent must return (spotlight padding per phase, no overflow at five widths, strings that must be absent). Round 57, dated 2026.10.02, carries that overview's look into the app itself, screen by screen. The Playwright scripts have accumulated this way (141 of them on 2026.10.02): every page, every draft stage, the upload flow, the review ledger, the MCP server, the security settings, the design audits. They are rerun after each round, together with 30 HTTP smoke checks. Design review notes are written up the same way, with a before/after table and a list of what could *not* be verified.

The loop closes on the research side too: I run the referee committee on my own working papers, and what it gets wrong (a point raised twice, a baseline axis that should not apply, a persona that reads a finance paper like an ML paper) becomes the next round's brief. Strengthening the program and strengthening the research are the same loop.

## Where it stands, and what is next

Single user, not public, no payments connected; the parts most likely to be useful to someone else first are the venue analytics and the referee committee. Next, in order: calibrate the committee against real decisions (my own review history and public referee reports) so that "2 things to change" is a measured prediction rather than a persona's opinion; agent control on a budget (per-project credit caps, a stop that actually stops, a ledger visible to the person at every step); multi-user hosting with local execution kept off; and the long-running agent workflows the Build stage already sketches, which reimplement a paper's method on new data end to end with every intermediate artifact versioned.
