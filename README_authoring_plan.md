# README Authoring Plan — AI Agent Orchestration Platform
## Yuno AI Assessment · May 2026

---

## Overview

This plan tells you **exactly what to write, what to screenshot, and what diagrams to embed** in the README, ordered by section. Every section maps to an evaluation criterion from the spec. The assessment weights are: demo 40 %, architecture/code 30 %, UI/UX 20 %, documentation 10 % — so the README must pull weight across all four.

---

## Section Map (in order)

| # | Section | Assets needed | Eval weight it serves |
|---|---|---|---|
| 1 | Hero banner | Badges | Documentation |
| 2 | Demo GIF / video | **Animated GIF — critical** | Demo (40 %) |
| 3 | System architecture diagram | **SVG structural diagram** | Architecture (30 %) |
| 4 | Quick start | Code blocks | Documentation |
| 5 | Feature gallery | **6 UI screenshots** | UI/UX (20 %) |
| 6 | Template 1 flow | **Flowchart diagram** | Architecture |
| 7 | Template 2 flow | **Flowchart diagram** | Architecture |
| 8 | HITL sequence | **Swimlane sequence diagram** | Demo + Architecture |
| 9 | Backend module map | **UML-style class diagram** | Architecture (30 %) |
| 10 | MCP data-access chain | **Chain diagram** | Architecture |
| 11 | Langfuse observability | **2 screenshots** | UI/UX |
| 12 | Agent configuration reference | Table | Documentation |
| 13 | How to add a workflow template | Numbered code steps | Documentation |
| 14 | How to add a messaging channel | Numbered code steps | Documentation |
| 15 | Env vars / tests / seed data / project structure / tech stack rationale | Tables + code | Documentation |

---

## Section-by-Section Writing Guide

---

### 1. Hero Banner

**What to write:**

```markdown
# AI Agent Orchestration Platform

> Multi-agent LangGraph workflows with HITL via Telegram, real MCP tools,
> and a full React monitoring dashboard — running locally with one command.

![Build](https://img.shields.io/badge/build-passing-success)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker--compose-v2-blue)
![License](https://img.shields.io/badge/license-MIT-green)
```

**Tips:**
- Keep the tagline to one sentence; it appears in GitHub's preview card.
- List the four key differentiators in badges: real runtime, HITL, Telegram, one-command start.

---

### 2. Demo GIF / Video (HIGHEST PRIORITY — 40 % of score)

**What to record:**

Record two separate GIF clips (or one 60-second combined clip):

**Clip A — Template 1 (Food Ordering HITL):**
1. Open Telegram, type: `Order chicken biryani under Rs300, 4+ stars, Hyderabad`
2. Switch to browser: watch Monitor page — nodes pulse in sequence: `router → ordering → hitl`
3. Back to Telegram: bot sends the HITL summary with restaurant + price + risk score
4. Type `YES` in Telegram
5. Back to browser: watch `fraud → payment → notification` nodes pulse
6. Telegram: bot sends "Order confirmed! Meghana Foods, Rs280."
7. Zoom in on the Runs page showing token/cost tracking

**Clip B — Template 2 (Complaint Resolution):**
1. Click "Trigger workflow" in UI with complaint text
2. Watch canvas: `router → complaint → hitl` lights up
3. Telegram: bot sends resolution options (reorder / refund)
4. Type `YES` → notification fires
5. Langfuse page: show trace timeline with per-node token costs

**How to embed:**

```markdown
## Live demo

### Template 1 — food ordering HITL

![Food ordering HITL demo](docs/demo_food_ordering.gif)

### Template 2 — complaint resolution HITL

![Complaint resolution demo](docs/demo_complaint.gif)
```

**Tools:** Use QuickTime (Mac) or OBS → convert to GIF with `ffmpeg`:
```bash
ffmpeg -i demo.mov -vf "fps=12,scale=1200:-1:flags=lanczos" -loop 0 demo.gif
```
Target: < 10 MB per GIF. 12 fps, 1200 px wide.

---

### 3. System Architecture Diagram

Use the SVG diagram produced in this plan. Embed as inline SVG or export to PNG:

```markdown
## Architecture

The platform has five layers:

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend | React + React Flow | Visual workflow builder, real-time monitoring |
| Backend | FastAPI | REST API, WebSocket, Telegram bot background task |
| Runtime | LangGraph StateGraph | Graph execution, HITL interrupt/resume |
| Tool layer | FastMCP @ /mcp/sse | All business data access |
| Data | PostgreSQL 15 + Redis 7 | Persistence, WebSocket event streaming |
```

Then embed the architecture diagram image (export the SVG as `docs/architecture.svg`):

```markdown
![System architecture](docs/architecture.svg)
```

**What must be visible in the diagram:**
- React Frontend (port 3000)
- FastAPI backend (port 8000) with Telegram bot as a background task
- LangGraph StateGraph with all 6 node names matching React Flow node IDs
- MCP tool layer (6 tools) clearly separated from the agent layer
- PostgreSQL + Redis as the bottom layer
- Langfuse as a side-car observability service
- Redis Streams labeled with its two-and-only-two uses: `stream:logs:{run_id}` and `telegram:session:{chat_id}`

**Prohibited in this diagram:** Don't add a "cloud" or external API box — the platform is fully local.

---

### 4. Quick Start

```markdown
## Quick start

### Prerequisites
- Docker Desktop + Docker Compose v2
- OpenAI API key (`sk-…`)
- Telegram bot token — create one via [@BotFather](https://t.me/botfather)

### 1. Clone and configure
\`\`\`bash
git clone https://github.com/bharavi1905/yuno-ai-assessment
cd yuno-ai-assessment
cp .env.example .env
# Edit .env — minimum: OPENAI_API_KEY and TELEGRAM_BOT_TOKEN
\`\`\`

### 2. Start everything
\`\`\`bash
docker compose up --build
\`\`\`

First boot automatically seeds 50 restaurants, 500 menu items, 24 payment routes,
30 fraud rules, 500 orders, 5 default agents, and 2 workflow templates.
Subsequent starts skip seeding and boot in ~3 seconds.

### 3. Get your Telegram chat ID
Message your bot, then:
\`\`\`bash
curl "https://api.telegram.org/bot<TOKEN>/getUpdates"
# Find: "chat": {"id": <number>}
\`\`\`

### Service URLs
| Service | URL |
|---|---|
| Web UI | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Langfuse | http://localhost:3001 (admin@yuno.local / admin) |
| MCP SSE | http://localhost:8000/mcp/sse |
```

---

### 5. Feature Gallery (6 screenshots)

**Screenshots to take and their captions:**

| # | Page | What to show | File name |
|---|---|---|---|
| 1 | Dashboard | Stat cards (total runs, agents, active runs) + recent runs table + live activity feed | `docs/ss_dashboard.png` |
| 2 | Agents page | Agent cards with green "Config applied" badge on ordering/complaint and "Workflow usage" badges | `docs/ss_agents.png` |
| 3 | Workflow canvas | React Flow canvas with all 6 nodes rendered + "Trigger" button visible | `docs/ss_workflow_canvas.png` |
| 4 | Workflow canvas — live | Same canvas mid-execution with `ordering` node pulsing (highlighted state) | `docs/ss_workflow_live.png` |
| 5 | Run history + detail | Runs table on left, run detail panel on right with execution events and token breakdown | `docs/ss_run_detail.png` |
| 6 | Monitor page | Full-screen live event feed with colour-coded node events scrolling | `docs/ss_monitor.png` |

**Embed layout:**

```markdown
## Features

### Workflow canvas with live node highlighting
![Workflow canvas](docs/ss_workflow_canvas.png)
*React Flow canvas showing the food ordering graph. Nodes pulse with a live indicator
during execution. Click any node to configure its agent prompt, model, and tools.*

### Real-time monitoring dashboard
![Monitor](docs/ss_monitor.png)
*Colour-coded event feed streamed via WebSocket. Every node start/complete event,
HITL checkpoint, and state transition appears in real time.*

### Agent management
![Agents](docs/ss_agents.png)
*Agent CRUD with config-active badges (green = DB config loaded at runtime) and
workflow-usage badges showing which templates reference each agent role.*

### Run history with token tracking
![Run detail](docs/ss_run_detail.png)
*Drill-in run detail showing execution events, per-agent token usage, and USD cost
estimate calculated from OpenAI published pricing.*
```

---

### 6. Template 1 — Food Ordering Flowchart

Use the flowchart SVG from this plan. Export to `docs/flow_food_ordering.svg`.

```markdown
## Workflow templates

### Template 1 — smart food ordering concierge

**Trigger:** Telegram message or UI trigger button
**Example:** `Order chicken biryani under Rs300, 4+ stars, Hyderabad`

![Food ordering flow](docs/flow_food_ordering.svg)

**HITL checkpoint message sent to Telegram:**
\`\`\`
Restaurant: Meghana Foods ⭐ 4.6
Item: Chicken Dum Biryani
Price: Rs280 · Delivery: ~35 mins
Payment: Juspay UPI · Fee: Rs5.60 · Total: Rs285.60
Risk score: 14/100 ✓

Reply YES to confirm or NO to cancel
\`\`\`

**Alternate paths:**
- `NO` → Order cancelled cleanly
- `"show other options"` → re-runs ordering with relaxed price constraint (+Rs50)
- No reply in 10 min → session expired message sent to Telegram
- Fraud score > threshold → fraud block path skips payment, sends blocked notification
```

---

### 7. Template 2 — Complaint Resolution Flowchart

Export to `docs/flow_complaint.svg`.

```markdown
### Template 2 — wrong order resolution (complaint resolution)

**Trigger:** Complaint message via Telegram or UI trigger
**Example:** `I ordered chicken biryani but got veg biryani from Ohri's`

![Complaint resolution flow](docs/flow_complaint.svg)

**Resolution types:**
| Type | When | What happens |
|---|---|---|
| `reorder` | Wrong item received | Re-searches and places correct order through Ordering sub-flow |
| `compensate` | Quality issue / late delivery / explicit refund | Issues INR credit to user account |

**HITL checkpoint message:**
\`\`\`
Resolution for your complaint:
Re-order: Chicken Biryani from Paradise Restaurant
OR
Refund: Rs280 credit to your account

Reply YES to confirm or NO to cancel
\`\`\`
```

---

### 8. HITL Sequence Diagram

Use the swimlane diagram from this plan. Export to `docs/hitl_sequence.svg`.

```markdown
## HITL flow (human-in-the-loop)

![HITL sequence](docs/hitl_sequence.svg)

**Critical implementation constraint:**
`thread_id` MUST equal the Telegram `chat_id`. This is what `AsyncPostgresSaver`
uses to retrieve persisted `AgentState` when the user replies. A random thread_id
breaks HITL resume.

**Session expiry:**
Redis TTL of 600 seconds is set on every HITL session. After 10 minutes of no reply,
the bot sends: *"Your session has expired. Please start a new request."*

**What REQUIRES HITL:**
| Action | Requires HITL |
|---|---|
| Placing a food order | **Yes — always** |
| Payment processing | **Yes — always** |
| Complaint resolution | **Yes — always** |
| Fraud scoring | No — internal decision |
| Restaurant search | No — research phase |
| Telegram notifications | No — autonomous |
```

---

### 9. Backend Module / Class Diagram

Use the module diagram from this plan. Export to `docs/backend_modules.svg`.

```markdown
## Code architecture

### Backend module map

![Backend modules](docs/backend_modules.svg)

**Key design rules enforced in the codebase:**
1. Every agent file exports `*_TOOLS`, `*_PROMPT`, and `build_*_agent(config)` — no exceptions.
2. `build_agent(config)` in `agents/base.py` is the ONLY way workflow nodes instantiate agents.
3. Agent configs are loaded from PostgreSQL at workflow start — edit in UI, affects next run.
4. `AgentState` TypedDict is the single state contract — all nodes read from and write to it.
```

---

### 10. MCP Data Access Chain

Use the chain diagram from this plan. Export to `docs/mcp_chain.svg`.

```markdown
## MCP tool layer

![MCP data access chain](docs/mcp_chain.svg)

All six tools are mounted at `/mcp/sse` via FastMCP SSE transport alongside FastAPI.
They are mock-backed with real interfaces — no live third-party API calls required.

| Tool | Backing data | Purpose |
|---|---|---|
| `restaurant_search` | 50 seeded restaurants | Search by city, cuisine, price, rating |
| `menu_retrieval` | ~500 menu items | Fetch items for a restaurant |
| `order_lookup` | 500 historical orders | Retrieve user's recent order (complaint flow) |
| `payment_routing` | 24 gateway configurations | Select best gateway for a transaction |
| `fraud_scoring` | 30 rule-based fraud rules | Evaluate transaction risk |
| `telegram_notify` | Live Telegram bot | Send messages to user |
```

---

### 11. Langfuse Observability

```markdown
## Langfuse observability (self-hosted)

Langfuse runs fully inside `docker compose up` — no external account needed.

**Access:** http://localhost:3001  
**Credentials:** `admin@yuno.local` / `admin`

![Langfuse trace timeline](docs/ss_langfuse_trace.png)
*Every agent node execution is a Langfuse span. The session groups all spans from
one workflow run by `session_id = run_id`. Deterministic trace IDs (`uuid5(run_id + node_name)`)
mean HITL re-runs appear under the same trace.*

![Langfuse token costs](docs/ss_langfuse_tokens.png)
*Token usage and USD cost per model visible without any manual configuration.*

**What is traced automatically:**
- One trace per node execution
- Input/output tokens and latency via the LangChain `CallbackHandler`
- Model name per invocation
- Graceful degradation: if Langfuse is unreachable, workflows continue normally
```

**Screenshots to take for this section:**
1. `ss_langfuse_trace.png` — Langfuse Traces list view showing multiple spans for one workflow run
2. `ss_langfuse_tokens.png` — Token usage dashboard showing per-model cost breakdown

---

### 12. Agent Configuration Reference

```markdown
## Agent configuration

Agents are configured via the Agents page in the UI. Configuration is stored in
PostgreSQL and loaded at workflow start — UI changes take effect on the next run.

| Field | Type | Runtime-applied | Description |
|---|---|---|---|
| Name | string | — | Unique identifier |
| Role | string | Yes (ordering, complaint) | Determines which builder function to use |
| System prompt | text | Yes (ordering, complaint) | Full prompt loaded from DB at runtime |
| Model | select | Yes (ordering, complaint) | `gpt-4o` or `gpt-4o-mini` |
| Tools | multi-select | Yes | MCP tool names assigned to this agent |
| Channels | select | — | `Telegram` or `None` |
| Schedule | cron | — | UI metadata only |
| Memory window | int | — | UI metadata only |
| Skills | tags | — | UI display labels |
| Guardrails | JSON | — | UI metadata (max_tokens, forbidden topics) |

**Config-active badge:** Agents with role `ordering` or `complaint` show a green
"Config applied" badge — their DB config is actually loaded at runtime.
Agents with role `fraud`, `payment`, `notification` show "Fixed behavior" — they
use hardcoded deterministic chains regardless of UI edits.
```

---

### 13. How to Add a New Workflow Template

Copy exactly from the existing README section (it is already well-written). The key is to verify the 5 steps are clear with numbered headings and code blocks for each:

1. Implement node function in `graph/nodes.py`
2. Add graph builder in `graph/builder.py` and register in `init_graphs()`
3. Add routing in `graph/edges.py → route_from_router()`
4. Seed the template in `scripts/seed.py → _upsert_workflow_templates()`
5. Add canvas nodes/edges to `frontend/src/components/WorkflowCanvas.tsx`

Add a callout box at the top:
```markdown
> **Node naming constraint:** LangGraph node names MUST match React Flow node IDs
> exactly. This is what enables live node highlighting during execution.
> `current_step` in `AgentState` maps directly to the React Flow node ID that pulses.
```

---

### 14. How to Add a Messaging Channel

Copy from the existing README and add the critical constraint box:
```markdown
> **thread_id constraint:** `thread_id` MUST equal the channel user/conversation ID.
> This is the key `AsyncPostgresSaver` uses to retrieve persisted `AgentState` when
> the user replies to a HITL prompt. A mismatch breaks HITL resume silently.
```

---

### 15. Supporting Sections (Environment, Tests, Seed Data, Structure, Tech Stack Rationale)

These already exist in the README. Polish them by:

1. **Env vars table** — add a "Required" column and mark which vars have Docker defaults.
2. **Tests section** — add the exact commands from the existing README, no changes needed.
3. **Seed data table** — already good. Add a note: *"Seeding is idempotent. Re-running docker compose up never duplicates data."*
4. **Project structure** — the existing tree is accurate. Keep it.
5. **Tech stack table** — add a "Why chosen" column with one-line justifications.
6. **Runtime rationale** — the LangGraph section is already strong. Add a comparison table:

```markdown
| Alternative | Why not chosen |
|---|---|
| CrewAI | No native interrupt/resume primitive for HITL across turns |
| AutoGen | Conversation-centric, not graph-centric; harder to map to React Flow |
| Custom runtime | HITL state management + async checkpointing = re-implementing LangGraph |
| LangGraph | First-class interrupt(), AsyncPostgresSaver, explicit topology — exactly what this demo needs |
```

---

## Screenshot Checklist

Take all screenshots with the browser at 1440 px wide, dark mode or light mode (pick one and be consistent). Crop to the content area only — no browser chrome.

| File | Page | State required | Notes |
|---|---|---|---|
| `docs/ss_dashboard.png` | Dashboard | At least 2 completed runs showing | Show stat cards + run table + live feed |
| `docs/ss_agents.png` | Agents | Default agents seeded | Show "Config applied" badge on ordering |
| `docs/ss_workflow_canvas.png` | Workflows | Food ordering template selected | All 6 nodes visible, no active run |
| `docs/ss_workflow_live.png` | Workflows | During an active run | One node mid-pulse (ordering or hitl) |
| `docs/ss_run_detail.png` | Runs | A completed run selected | Event log + token breakdown visible |
| `docs/ss_monitor.png` | Monitor | During or just after a run | Color-coded event stream visible |
| `docs/ss_langfuse_trace.png` | Langfuse | After at least one full run | Trace list with spans visible |
| `docs/ss_langfuse_tokens.png` | Langfuse | Same run | Token/cost tab visible |
| `docs/demo_food_ordering.gif` | Telegram + browser | Live HITL flow | 30–60 s, < 10 MB |
| `docs/demo_complaint.gif` | Telegram + browser | Complaint HITL | 30–60 s, < 10 MB |

---

## Diagram Export Instructions

All SVG diagrams from this plan can be saved from the browser (right-click → Save as SVG) or re-generated as code. Place them in `docs/`:

| File | Diagram | Section |
|---|---|---|
| `docs/architecture.svg` | System architecture (5-layer structural) | Section 3 |
| `docs/flow_food_ordering.svg` | Template 1 flowchart | Section 6 |
| `docs/flow_complaint.svg` | Template 2 flowchart | Section 7 |
| `docs/hitl_sequence.svg` | HITL swimlane | Section 8 |
| `docs/backend_modules.svg` | Backend class/module map | Section 9 |
| `docs/mcp_chain.svg` | MCP data access chain | Section 10 |

---

## Final README Quality Checklist

Before submission, verify:

- [ ] GIF embeds render correctly in GitHub preview (< 10 MB each)
- [ ] All 8 screenshots are in `docs/` and referenced correctly
- [ ] All 6 SVG diagrams are in `docs/` and display in GitHub (GitHub renders inline SVG)
- [ ] Table of Contents has working anchor links
- [ ] Quick start works from a clean `git clone` — tested on a second machine or Docker clean
- [ ] `thread_id = chat_id` constraint documented in HITL section
- [ ] LangGraph rationale comparison table present
- [ ] "Config applied vs Fixed behavior" agent distinction documented
- [ ] No broken image links
- [ ] All code blocks have a language tag for syntax highlighting (` ```python `, ` ```bash `, etc.)
