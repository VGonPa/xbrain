# XBrain Dashboard Style Guide

## Product Direction

XBrain is a private intelligence terminal for a saved knowledge corpus. The UI should feel calm, precise, and premium: more command center than marketing page. It prioritizes scanning, triage, evidence, and confident navigation across the corpus.

## Theme

The dashboard uses a forest editorial system with paired light and dark themes. Every visible color must route through CSS custom properties in `dashboard.template.html`; chart colors are read from the computed theme at runtime.

Core roles:

- `--color-bg`: page background.
- `--color-panel`, `--color-panel-raised`, `--color-panel-soft`: surfaces.
- `--color-text`, `--color-text-muted`, `--color-text-faint`: text hierarchy.
- `--color-border`, `--color-border-strong`: borders and dividers.
- `--color-accent`: primary action and selected state.
- `--color-forest`, `--color-moss`, `--color-fern`, `--color-clay`, `--color-sky`, `--color-amber`, `--color-stone`, `--color-bark`: chart and status accents.
- `--color-danger`: failures and risk states.
- `--shadow-panel`, `--shadow-drawer`, `--shadow-focus`: elevation and focus.

## Type

- Display: `Fraunces` for brand and major numbers only.
- UI: `Hanken Grotesk` for labels, prose, and controls.
- Mono: `Spline Sans Mono` for timestamps, commands, metadata, and compact technical labels.

Scale:

- Hero/brand: 44-58px responsive via media queries, never viewport-scaled.
- Panel title: 18-24px.
- Body: 14-15px.
- Metadata/chips: 9-12px.

## Components

- Cards are used only for repeated metrics, panels, modals, and the drawer. No card-in-card nesting.
- Segmented controls express view or data range choices.
- Workspace navigation separates Library, Atlas, Analytics, and Operations; Ask remains available globally as a contextual drawer.
- Library is the default route. Search, filters, quick views, sorting, and document detail belong there.
- The shared filter bar is the single analytical state for Library, Atlas, and Analytics. It persists in both URL parameters and local storage.
- Signal rail cards are compact command surfaces: they summarize one live metric, show proportional state, and navigate to the relevant detail.
- Icon buttons are square, token-sized, and include accessible labels/tooltips.
- Chips use the mono font, uppercase labels, and semantic colors.
- Charts must use the active CSS theme, not hardcoded JS colors.
- Drawers are used for document/topic drill-down and contextual Ask so primary workspaces remain dense.
- Every chart has a semantic label and a table or ranked-list alternative. Atlas Documents is the primary non-canvas graph alternative.
- The served note viewer (`/notes`) uses the same forest tokens, theme persistence, compact controls, and editorial reading rhythm as the dashboard.

## Interaction Principles

- First screen: searchable Library, not a landing page or undifferentiated KPI wall.
- Preserve all existing dashboard actions: refresh, retry failures, command copy, drill-down charts, note links, and library chat.
- Theme toggle persists in local storage and must not require a server.
- Controls need clear hover, focus, active, disabled, and loading states.
- Mobile layout keeps the primary search visible, folds secondary filters behind one 44px control, and uses ranked/table fallbacks where canvas is too dense.
- All interactive controls expose visible focus, keyboard access, and at least a 44px touch target where they act as primary controls.
- Production serving must stay local or behind authenticated reverse proxy access because the dashboard includes Ops and Ask actions.
- Analytics KPIs stay as a compact four-column system and always describe the current filtered selection.
- Operations is the only home for maintenance and storage: actions, pending counters, failure evidence, command copy, and footprint.

## Copy Tone

Copy is concise, operational, and specific. Avoid generic dashboard filler. Prefer verbs and precise nouns: "Refresh pipeline", "Evidence stream", "Review queue", "Captured articles".
