# Remi WebUI frontend (`webui/`)

Status: wave 1 (foundation), 2026-10-15. Binding contracts:
[architecture.md](architecture.md) (layers, HTTP/WS API) and
[schema-format.md](schema-format.md) (the `remi.ui` JSON the renderer draws).

Stack: React 19 (React Compiler), Vite 8 (rolldown), TypeScript 6 strict,
Tailwind 4, shadcn/ui `base-nova` on Base UI, Zustand 5, TanStack Query 5,
i18next, CodeMirror 6 (prompt editor), Vitest + Testing Library (jsdom).

## Scripts

Run in `webui/`:

| command | what it does |
|---|---|
| `pnpm dev` | Vite dev server; proxies `/api` (incl. WS), `/sdapi`, `/internal` (the server's API prefixes) to `REMI_BACKEND` (default `http://127.0.0.1:7860`) |
| `pnpm dev:mock` | No backend: `vite --mode mock` (`.env.mock` sets `VITE_MOCK=1`; `VITE_MOCK=1` in the environment does the same in any mode), API answered in the browser from fixtures, jobs simulated with live previews |
| `pnpm build` | `tsc -b` then the production build into `webui/dist` (served by the Python server) |
| `pnpm preview` | serves the built `webui/dist` (Vite preview, port 4173) with the same proxies as `pnpm dev`: the production bundle against a running server |
| `pnpm typecheck` | type-check only |
| `pnpm gen:api` | regenerates `src/api/gen/schema.d.ts` from `src/api/openapi.json` (`openapi-typescript`); run it after `$PYTHON tools/export_openapi.py` (from the repository root) whenever a server route or model changed; `gen/gen.test.ts` and the `contract:openapi` step of `tools/check.sh` fail while either is stale ([API layer](#api-layer-srcapi)) |
| `pnpm test` / `pnpm test:watch` | Vitest; needs Node.js 22.22.2+, 24.15+ or 26+ (what jsdom, Vitest and Babel all run on) |
| `pnpm test:e2e` | Playwright against mock mode (`tests/e2e`; starts `vite` on port 5199, bypassing proxies for loopback) |
| `pnpm lint` | oxlint (must be silent) |
| `python3 tools/shadcn_fetch.py <names>` | add shadcn registry components with curl (Node cannot use the proxy) |
| `$PYTHON tools/dump_schemas.py` (requirements.txt, no Gradio) | re-dump the real tab schemas into `src/api/mock/contract/*.json` (contract tests) after a `remi/ui` / `remi/core` change; `--check` (the `contract:schemas` step of `tools/check.sh`) fails when they are stale |

## Layout of `src/`

```
main.tsx            boot: load locale, mount providers, <App/>
index.css           design tokens (Tailwind @theme), base styles, renderer row classes
styles/shadcn.css   vendored keyframes and variants from shadcn/tailwind.css
app/                shell: App, router, routes (nav + lazy pages), NavRail/BottomBar/MoreSheet,
                    Header (+ header slot, ModelQuickBar, AppearanceMenu), command palette,
                    ws-bridge (WS events → query cache / progress / toasts)
api/                http transport, typed endpoints, models, query keys, WS store + frames,
                    progress store, mock/ (fixtures, in-browser server, mock socket)
ui/                 the schema renderer (see below)
widgets/<name>/     custom widgets, one directory each (auto-registered)
features/           one directory per feature, shared by the pages: prompt-editor, prompt-tools,
                    gallery (session results, history grid), library (boards, lineage), lightbox,
                    compare, progress (run machine + controller, restore, wake lock / title /
                    notifications), queue, pipelines, recipes, networks, model-bar, quicksettings,
                    send-to, infotext, pnginfo, extras, auth, errors, server-control
canvas/             the image editor (state, engine, ui; see "Image editor")
pages/<page>/       one lazily loaded chunk per page; pages/ext/ = extension pages (#/ext/<id>),
                    pages/custom/<name>/ = dedicated views for extension pages with `widget=`
components/         app-level components (SearchSelect, Segmented, ErrorBoundary, …)
components/ui/      vendored shadcn components (edit sparingly; lint-exempt)
i18n/               i18next setup, locale store, catalog loading
locales/            chrome translations, one JSON per locale (zh-CN.json; en.json: English plural forms only)
lib/                appearance store (theme.ts), toast, media query, cn(), persisted-store guard
                    (persist.ts), the tab's identity (tab-identity.ts), stale-chunk reload,
                    retryable-lazy / lazy-module (loads that retry, modules fetched on first
                    intent), IME keys, safe area, platform (isMac), layout hooks (use-label-fits,
                    use-fitting-placeholder, use-pinned-bar, use-scroll-fade), motion features
styles/             fonts.css, shadcn.css (keyframes and variants the vendored components use)
test/               setup.ts (jsdom shims, jest-dom, i18n init), fake-socket.ts (WebSocket double),
                    api-fixtures.ts (complete API models), fetch-recorder.ts (records and answers
                    the API clients' requests)
```

`src/canvas/` is the image editor (see "Image editor"); it is the only module
that may import Konva, and only `canvas/engine/renderer.ts` does, inside the
editor's lazy chunk. `react-konva` stays unused (the editor drives Konva
imperatively).

## App shell

- **Routing**: `app/router.ts` is a typed hash router (`#/txt2img`, `#/img2img`,
  `#/extras`, `#/png-info`, `#/gallery`, `#/networks`, `#/queue`,
  `#/settings`, `#/extensions`, `#/ext/<id>` (extension pages; `#/page/<id>`
  still opens them), optional `?query`). Hash routing
  keeps the server contract to "serve `index.html` at `/`". Use
  `navigate(route)`, `routeHref(route)` for links, `useRoute()` to read.
  The filters of the queue history, the recipes, the installed and the
  available extensions and the diagnostics live in the address (`app/route-params.ts`:
  `useRouteParams`, and `useRouteSearch` for a search field, written after a
  pause), so a reload, a link and Back restore them; an Extensions tab starts
  with no filters. Settings keep theirs in a module store mirrored into the
  address (`settings-filters.ts`); the gallery's filters and the network
  browser's persist (`remi-gallery-view`, `remi-networks`); the gallery's
  search text is per visit.
- **Pages** are code-split: `app/routes.tsx` maps each route name to a
  `retryableLazy(() => import("@/pages/…"))`; nav items preload their chunk on
  hover/focus. A page component receives `{ route }` (its own route variant).
- **Navigation**: `BUILTIN_NAV` plus extension pages from `GET /app`
  (`pages`), ordered/filtered by `Bootstrap.tabs`; Queue, Settings and
  Extensions are always present. Desktop: slim left rail with tooltips
  (in a short window, a phone held sideways, its pages scroll: the links
  keep their 40 px and the cut edge fades);
  below `md`: bottom bar (primary items) + "More" sheet. Both sit on the
  neutral `--rail` surface (the sidebar tone, a shade off the page, never a
  colour of its own); only the current page is marked, by a `--primary`
  tint (`--rail-active`) and its icon in `--primary` (`--rail-current`).
- **Header**: page title, a page slot (`<HeaderSlot>` from
  `app/header-slot.tsx` portals page controls into it), the model quick-bar,
  command palette button, WS connection dot, appearance menu, and a progress
  line while a job runs. On a phone (below `sm`) a page whose own tabs are
  in the slot (Generate's txt2img / img2img, marked `data-page-tabs`) takes
  the title's room: the title is visually hidden there (`sr-only`, still the
  page's `h1`), as the tabs and the bottom bar name the page, rather than
  cut to a stub ("G…"); a page without tabs there keeps its title,
  truncated with its full text as a `title` once its controls take the row.
  A header button showing its icon alone is a 28 px square like search and
  the appearance menu: the networks button wherever its label is hidden
  (below `lg`), the quick settings button on a phone, its count of pinned
  settings waiting for `sm`, and on a phone a page's own button whose
  label has no room there (the queue's Pause / Resume, the gallery's
  Rescan, PNG Info's Clear), its label read out (`sr-only`, never
  `hidden`, which would leave the button without a name), and its icon
  16 px like theirs, not the 14 px of a small labelled button
  (`ICON_SQUARE_BELOW_SM` / `ICON_SQUARE_BELOW_LG` in `app/header-slot.tsx`).
  Every header button is ghost like them (no fill or edge until hovered or
  pressed), a page's own included: an outlined one among them reads as pressed. On a phone the header's controls sit 4 px
  apart, and the hairline that sets the networks button apart from
  Generate's tabs is left out (its icon does that there).
- **Command palette** (Ctrl/Cmd+K, `cmdk`, its own chunk loaded on first
  open). Register commands with
  `useRegisterCommands(source, commands)`; labels and groups are English and
  translated at render.
- **Keyboard shortcuts** (`app/ShortcutsDialog.tsx`, lazy): every app-wide
  shortcut in one dialog, opened with `?` outside text fields and modal
  dialogs (`app/shortcuts-help.ts`; a handler that uses `?` itself — the image
  editor's list — prevents its default), the palette's "Keyboard shortcuts"
  and the appearance menu. Add a shortcut there when you add one.
- **Page-wide keys and modals**: a key handled for the whole page (the
  generate keys, the settings' Ctrl/Cmd+S, `?`) asks `modalOpen()`
  (`lib/modal-open.ts`) whether a modal dialog has the keyboard, never a
  selector of its own. A dialog, an alert dialog, a sheet and the sign-in
  screen are modal; a popover or a search select's popup (a `role="dialog"`
  in Base UI's positioner, `role="presentation"` with `data-side`), a menu,
  a panel marked `aria-modal="false"` (the Networks drawer) and a dialog
  fading out (`data-closed`) are not. Escape is another matter: an open
  popover or menu takes it (`escapeIsFree` in `pages/generate/shortcuts.ts`).
- **Toasts**: `sonner`; use `showToast(level, message)` from `lib/toast.ts`
  for server-provided levels, or `toast` directly. They sit bottom-right,
  lifted above any control pinned to the bottom edge under them
  (`app/toast-offset.ts`): mark such a control `data-toast-obstacle` (the
  phone's bottom navigation and sticky Generate bar, the canvas's bottom
  toolbars, the apply bars and the gallery's selection bar carry it). A
  description keeps its line breaks (`whitespace-pre-line` on the toaster's
  description slot): put each note of a different kind, and each item of a
  list ("Some values were skipped"), on a line of its own with `\n`.
- **Pinned bars and the focus**: a bar pinned over an edge of a scroller (the
  generate and Extras run bars, the settings and extensions apply bars, the
  JSON Builder's send bar) takes its ref from `usePinnedBar(edge)`
  (`lib/use-pinned-bar.ts`). It gives the scroller scroll padding as deep as
  the bar reaches in, so Tab and Shift+Tab stop a control clear of the bar
  instead of under it (WCAG 2.4.11), and the same depth as `--pinned-top` /
  `--pinned-bottom` for what the page lays out at that edge: below lg the
  JSON Builder page is a size container and its stacked canvas is at most
  `100cqh` less the send bar's depth (its 16 rem floor gives way to that
  down to 10 rem), so on a phone held sideways its zoom bar is not under the
  send bar. A bar laid over a scroller it is not
  inside passes a function that finds it (the settings list, the gallery's
  selection bar). A virtualized list scrolls a row into view itself and reads
  no scroll padding: the gallery grid takes the scroller's bottom scroll
  padding as the virtualizer's `scrollPaddingEnd`, leaves that much room after
  the list (so the last row and the load-more status scroll clear), and
  lifts the focused tile when the bar comes up over it (only the tile button
  with the focus, only a tile on screen: a clicked check mark moves
  nothing). The bar takes the
  grid's key, so a rebuilt grid gets the padding too. The e2e check is "focus
  is not hidden under a pinned bar" in `a11y.spec.ts`.
- **Page column**: Extensions and every extension page lay their content out
  in `PAGE_CONTAINER` (`components/page-container.ts`): one max width,
  padding and vertical rhythm.
- **Appearance** (`lib/theme.ts`, persisted as `remi-appearance`): theme
  (light/dark/system), density (comfortable/compact), accent
  (`iris|blue|teal|green|amber|coral|rose`). `index.html` applies the stored
  values before first paint; `useApplyAppearance` keeps `<html>` in sync.
  The theme comes from, first to last: `?__theme=light|dark|system` in the
  URL (A1111's theme link: it picks that theme, as the menu would, and is
  remembered), the user's pick (`themeChosen`), the server's `--theme`
  (`window.__REMI_THEME__`, written into the shell by `remi/server/spa.py`
  before the pre-paint script), then dark. A browser that never picked a
  theme follows `--theme` when it changes.
- **user.css** (`app/user-stylesheet.ts`): when the bootstrap's
  `user_assets` lists `user.css` (the data folder's, as the Gradio UI
  applied it), the shell links `/api/v2/user-assets/user.css` at the end of
  `<body>`, after every stylesheet the app loads (Vite appends lazily loaded
  page CSS to `<head>`), so a user rule beats an app rule of the same
  specificity. It is linked again on the next page load after the file
  changes (served `no-cache`).
- **Persisted stores** (Zustand `persist`): the stored state is read back
  inside `create()`, while the module is still being evaluated. Everything
  `merge`/`migrate` reaches, directly or through the checks it calls, is
  declared above the store or as a `function` (hoisted): a `const` below
  the store is still in its temporal dead zone, the `ReferenceError` is
  swallowed and the store silently starts empty (the next write then
  overwrites what was stored). Test a store's restore by seeding the
  storage and importing the module afresh (`vi.resetModules()`), not by
  calling its `merge` directly. What the storage holds was written by an
  older build, by hand or by another app on the origin, so `merge` keeps
  only what is of the store's own types: `mergeTyped` (`lib/persist.ts`),
  with `rules` naming the values a field takes (an enum, a colour) and
  `keepEntries` checking the items of a list or map of objects; a store
  that restores its state its own way (the form drafts, the locale, the
  JSON Builder) checks each field it reads as strictly.
  `lib/persist-stores.test.ts` loads every store from damaged snapshots:
  add a new one to its table (it fails on a module that imports zustand's
  `persist` and is neither there nor on its list of stores tested
  elsewhere).

## Design system

- Tokens live in `index.css`. Surfaces follow an animation studio's light table
  in two calibrated studio palettes: **Washi Paper & Technical Ink** (light) and
  **Darkroom Graphite & Warm Ivory** (dark: `--canvas` → `--sidebar` →
  `--background` → `--card` → `--popover` calibrated neutral luminance ladder),
  keeping generated images free of colour cast while maximising inspector
  readability. One restrained brand hue drives `--primary` and `--ring`:
  `--brand-h/-c/-l/-fg`, defaulting to a technical cobalt/slate pencil (`blue`,
  hue 248°, restrained chroma) set per accent with `[data-accent=…]` (every
  pairing audited to ≥ 4.5:1 WCAG AA across all 7 accents in both themes).
  shadcn's `--accent` stays a neutral hover surface; `--glow` renders a machined
  warm-ivory (dark) or technical-ink (light) keycap for the primary creation CTA
  (`Generate`, `Send to txt2img`).
  Colours come from these tokens, never from Tailwind's palette (`red-300`,
  `neutral-900`: a shade ignores the theme and the accent;
  `lib/design-guards.test.ts`, "colours"). White and black stay for controls
  over a picture or the viewer's black backdrop; a control there that needs
  a token colour (the viewer's Remove, red on hover) carries the `dark`
  class, so the tokens it reads are the dark theme's in either theme.
  Extra semantic colours:
  - `success`, `warning`, `destructive`, `star` (warm gold star marks, ≥ 3.5:1
    on paper), `canvas` (borderless image backdrops), utility `bg-checker`;
  - **Editorial callout strips & advice**: `--advice-warn`, `--advice-warn-bg`,
    `--advice-warn-border`, `--advice-info`, `--advice-info-bg`,
    `--advice-info-border` (rendered as 2 px left-accent hairline strips, never
    pill-inside-box);
  - **Model & capability metadata**: `--capability-badge`,
    `--capability-badge-bg`, `--capability-badge-border` (inline borderless
    mono micro-tags in the header);
  - **Structured prompt/parameter diffs**: `--diff-add`, `--diff-add-bg`,
    `--diff-del`, `--diff-del-bg`.
  The shell (nav rail, bottom bar) is a neutral surface too: `--rail` is
  `--sidebar`, separated by a crisp hairline (`border-border`), with the active
  destination elevated calmly in neutral surface tone (`--rail-active`) plus a
  2 px left registration peg and active icon in `--rail-current`.
- **Studio Instrument Craft (Anti-AI-Slop Rules)**:
  - **Unified Model Bar Instrument Strip**: `ModelBar` groups Preset,
    Checkpoint, VAE/TE, and Low-Bits Dtype inside one continuous 28 px recessed
    strip (`bg-field`) separated by vertical hairlines (`h-3.5 w-px bg-border`),
    eliminating disconnected header box soup while preserving unclipped focus
    rings (`focus-visible:z-10`). Model capability badges (`Diffusers`, `GGUF`,
    `INT8-ConvRot`, `NVFP4`, `SVDQuant`, `FP8`, `Turbo`, `RGBA`) render as
    whisper-quiet borderless inline mono micro-tags.
  - **Single Status-Dot Rule**: `ConnectionDot` is the sole green LED in the
    header; a lost connection adds the word "Offline" beside it (a touch
    screen cannot hover its tooltip), in the dot's own 28 px room first; on a
    phone the page title then yields to it (truncated, its full text as a
    title, or not shown beside the page's own tabs), never the controls.
    Where even then a phone's header has no room for it (beside Generate's
    tabs with a quick setting pinned, in English below 333 px), the word
    gives way to the dot: its box shrinks toward 28 px and the word wraps
    to a second line the box clips, the state still read out (`role=status`)
    and in the tooltip. `WorkerHealthPill` (`GET /api/v2/workers`) stays a calm borderless
    ghost monochrome readout (`CPU MPS` / `CUDA` / `1/2 GPU`) when healthy and
    only shows an amber/red indicator dot when a worker is degraded or failed.
  - **Live Darkroom Aspect-Ratio Viewfinder**: when the Generate stage has no
    image yet, `Stage.tsx` renders a live proportional framing box synced to
    `core.width` / `core.height` (`viewfinder.ts`) with four L-shaped hairline
    crop marks, optical center crosshair, and monospaced resolution telemetry
    (`1024 × 1024 · 1:1 · 1.05 MP`) — never cliché sparkle/wand placeholders.
  - **Editorial Strips, Never Pill-Inside-Box**: `WorkflowAdviceBar`, PUP
    warnings, and JSON Builder VGL's `CC BY-NC 4.0` notice use crisp 2 px
    left-accent strips (`border-l-2`) with plain monospaced prefixes instead of
    nested bordered pills inside bordered boxes; pane/extension counts use clean
    tabular mono numerals.
  - `ScriptsPanel` and `Slot` consume `ScriptSchema.applicable` /
    `requires_capabilities` / `inapplicable_reason`, automatically collapsing
    non-applicable model extensions into a quiet "Not applicable for active
    model" drawer instead of cluttering the workspace.
  - `PromptField` exposes a lazy-loaded Multimodal PUP Upsampler dialog
    (`PupTrigger` → `PupUpsamplerDialog`, `POST /api/v2/prompt-tools/upsample`)
    with model-family detection, reference image & aspect ratio attachment,
    unified split editorial token diff sheet, and one-click Apply / Revert.
  - `widgets/image-list` displays model-aware clickable slot badges
    (`<image_1>`, `ref_image_0`, `Picture 0`, `Image 1`) that insert tokens at
    the active prompt cursor, drag-and-drop reordering with drop-target ring,
    and `RGBA` badges when alpha channels are present.
  - `pages/custom/json-builder` supports MOSS-Image, FLUX.3 Layout, and Bria
    Fibo 2.0 VGL 2.0 JSON schema mode (with left-accent CC BY-NC 4.0 studio
    notice and unified `--glow` primary send CTA).
- **Density** scales Tailwind's `--spacing` (`[data-density=compact]` →
  0.215rem), so every `p-*`, `gap-*`, `h-*`, `size-*` shrinks together. Use
  spacing utilities, not fixed pixel sizes, for anything that should follow.
- Toolbars and option grids respond to their own width (Tailwind container
  queries, `@container/<name>`), not the window's: a pane of the three-pane
  generate view is narrow on a wide screen. A toolbar keeps to one row by
  moving its less used actions into a "More actions" menu (`ResultActions`
  `collapsible`, on `@container/actions`; its groups, keep / reuse / send /
  remove, are set apart by hairlines), or, narrow, by giving search a row of
  its own with a "View options" menu (sort, layout, size) beside it and the
  filters on a second row that scrolls sideways (`GalleryToolbar`,
  `@container/gallerybar`; the network browser folds card size and refresh
  into one menu; the img2img stage gives the result actions a row of their
  own under its canvas controls, `Img2imgStage`, and on one row shows its
  staging buttons' labels only where the result actions keep their folded
  set beside them). It never wraps into ragged rows. A size container
  (`@container/…`) has no width of its own to hold in a flex row, so one that
  shares a row gets a `min-w` for what it cannot fold: too narrow a row then
  wraps instead of sliding it under its neighbour. A grid never leaves an
  empty cell (the img2img mode picker and the phone's More sheet centre a
  short last row). A strip of fixed tabs that may not fit a phone (three or
  more, or labels that grow once translated) is a `FitTabsList`, which turns
  into a select rather than scroll, wrap or cut a label (Extensions, the
  network kinds, PNG Info, the queue's modes); a strip whose number of items varies scrolls
  sideways instead (the boards in a `FadeStrip`, the tag groups), its edges
  fading where there is more (`useScrollFade`, which also watches items that
  arrive after it mounts). A grid that can
  hold a line that does not wrap (a file name, a module name, a tab strip)
  and whose columns start at a breakpoint is `grid-cols-1` below it: its
  implicit column is as wide as that line and pushes a phone's page or
  dialog sideways (the backups, the model metadata editor).
- A running job's progress shows once where the user looks: over the stage
  (`ProgressOverlay`, which registers itself in
  `features/progress/stage-overlay.ts`; in a short window, a phone held
  sideways, one line along the stage's bottom edge, clear of the picture's
  middle), and in the run bar's status line
  only while the stage does not show it (img2img's Source view, the progress
  bar turned off), so the form does not move for the length of a job. The
  header's hairline is the global indicator on every page.
- Typography: self-hosted OFL faces (`@fontsource-variable`, no CDN), each
  split by unicode range so only the slices a page renders download, declared
  in `styles/fonts.css`: **Remi Sans** is Geist (the UI's Latin letters and
  digits, also the prompt editor's), **Remi Mono** is Geist Mono (setting
  keys, hashes, seeds, key caps, code, JSON) and Han is **Noto Sans SC**
  (`Noto Sans SC Variable`, next in `--font-sans`), so Chinese has the same
  letters and the same weights on every system instead of Microsoft YaHei's
  two; the system CJK faces stay behind it for the moment before a slice
  loads. `.tabular` / numeric inputs use tabular figures.
- Typesetting (`index.css` base layer): `text-autospace` puts the CJK gap
  between Han and Latin or digits in the text the UI writes (`2026 年 7 月
  1 日`), never in inputs, text areas or the prompt editor; paragraphs and
  labels use `text-wrap: pretty`, headings and text in a `.text-center` block
  `balance` (no single character left on an empty state's last line).
  A flex container does not ellipsize its text: put the text in its own
  `truncate` box. A button label that may not fit beside its icon (the
  running Generate button's "Queue another") is measured, not guessed with a
  breakpoint: `lib/use-label-fits.ts` shows it whenever it fits and otherwise
  leaves the icon alone, the label read out (`sr-only`) and kept as `title`.
- Motion: `tw-animate-css` + base-nova animations; a global
  `prefers-reduced-motion` rule neutralises animations and transitions.
  A scroll the app starts itself (`scrollIntoView`, `scrollTo`) takes its
  `behavior` from `scrollBehavior()` (`lib/motion.ts`): a `behavior:
  "smooth"` written out ignores that rule (`lib/motion.test.ts` fails on
  one). What may move, how long, and the two exceptions (a collapsible's
  height, colour and focus ring feedback) are in
  [Visual rules](#visual-rules). Nothing that stays on screen while a job
  runs animates endlessly (no `animate-pulse` / `animate-spin` on running
  indicators): an infinite animation makes the compositor redraw at
  display rate for the whole job,
  which measurably slows sampling where the browser shares the GPU
  (docs/research/ui-performance.md). Spinners are for short waits only.
  A generation step is one React commit and one drawn frame: while previews
  come every step, `api/progress-gate.ts` holds a `job.progress` until the
  next preview frame, and `api/paint-batch.ts` applies both in one animation
  frame once the image has decoded. No `backdrop-filter` on surfaces that
  change every step (the stage's progress overlay, the header with its
  progress line), and no fade or transition between live preview frames
  (`pages/generate/LivePreview.tsx` stacks the new frame over the previous
  one without animating it: a 150 ms fade drew about 12 frames per step).
- Scrollbars: thin, theme-aware (`--scrollbar`). Focus: visible rings from
  `--focus-ring` everywhere (`outline-focus-ring`, `focus-visible:ring-3
  focus-visible:ring-focus-ring`): `--ring` mixed with transparency, 80 %
  of it in the light theme (half of it read at about 2.2:1 on the paper)
  and 60 % in the dark theme (half stayed under 3:1 on the page for some
  accents), so that it reaches 3:1 on the page and on `--canvas` for
  every accent in both (`styles/focus-ring.test.ts`). It is re-derived
  with `--ring` wherever an accent is set. A focus ring, a keyboard
  cursor or highlight included, takes the token, never a share of `--ring`
  of its own (`ring-ring/60`; the test fails on one); a hover ring, which
  marks no focus, takes `--primary` (`hover:ring-primary/40`).
- See [Visual rules](#visual-rules) for the design rules every new screen
  follows.
- shadcn components import `cn` from the `cn` package (what the base-nova
  registry ships); app code imports it from `@/lib/utils`. Local fixes to
  vendored components: their focus rings take `ring-focus-ring` and
  `outline-focus-ring` instead of the registry's `ring-ring/50`; scalar
  sliders render one thumb and forward `getAriaLabel`; a select's value
  ellipsizes (line-clamp does nothing on the flex value); a dialog or
  alert dialog is no taller and no wider than
  the window less 2rem and the safe-area insets, and scrolls inside (up and
  down only; the full-screen editor opts out with `max-h-none`), and the command dialog no taller than the
  room below its top, its list shrinking; a popover, like the menus and
  selects, takes the room beside its trigger (`--available-height`) and
  scrolls inside (up and down only), and one that caps its own height keeps
  that room in the cap (`max-h-[min(60dvh,var(--available-height))]`); a menu or popover gives the focus
  back to its trigger only while nothing else has taken it (a dialog one of
  its items opened keeps it, `final-focus.ts`); a field focused as its popup
  opens takes the focus through the popup's `initialFocus`, never `autoFocus`
  (the popup would give the focus back to that field on close; a field that
  appears later in an open popup, an inline rename, may use `autoFocus`);
  a panel collapsed to nothing is `inert` (its controls leave the Tab order
  and the accessibility tree; a focus inside it moves to the separator
  beside it); `number-field.tsx` is ours.
- **Safe area**: the app shell pads its sides for a notch
  (`env(safe-area-inset-left/right)` in `App.tsx`) and the phone's bottom
  bar pads for the home indicator, but overlays are portalled outside the
  shell. An overlay that reaches a screen edge pads for the inset of that
  edge itself (the image viewer on every side, its black backdrop still to
  the edge; the full-screen editor at its sides and bottom; the networks
  sheet and the settings sections at the bottom and their open sides; the
  More sheet, shown on a phone only, at the bottom), and a centred dialog subtracts the insets from any size it
  takes from the viewport (the base dialog and alert dialog, `96vw`,
  `90dvh`; the command dialog's room below its top) and the base dialog and
  alert dialog centre on the safe area, not the window (moved by half the
  difference of the opposite insets: a notch on one side only). The nav rail pads its
  bottom for the home indicator, and a bar pinned to the bottom of a page
  (the gallery's selection bar) stands clear of it from `md` (below, the
  phone's bottom bar already keeps that room under the page). What positions itself
  in script reads the insets as numbers (`lib/safe-area.ts`): every Base UI
  positioner (menus, selects, popovers, comboboxes, tooltips, hover cards)
  takes the padding of `useCollisionPadding()` (called at the top of the
  component: `collisionPadding={collisionPadding}`), its own 5px from the
  edges of the safe area, and the toaster's offsets start at it
  (`toast-offset.ts`). A new positioned popup takes the same padding
  (`lib/safe-area.test.ts` checks every `Positioner` tag).

### Visual rules

The interface is an animation studio's light table: the desk recedes, the
work (the stage, the gallery, the previews) carries the colour. The rules
below keep new screens in that world and away from the generic look of
generated interfaces (sources: Linear's
2024 and 2026 refreshes, Figma UI3, Refactoring UI):

- **The work first.** Chrome never competes with images: neutral surfaces,
  no gradients or tinted washes on panels or on the shell (the nav rail is
  a neutral strip, not a coloured bar), no colour without meaning. The
  blue pencil (`--primary`) marks selection and focus, the red pencil
  destroys, the lamp (`--glow`) is the Generate button. The only decorative
  colour is a missing preview's monogram.
- **Spend boldness once per screen.** One element leads (the Generate button,
  the stage, the image grid); everything around it stays quiet. Before adding
  an accent, a badge or an icon, remove one. Section labels in one menu are
  all text or all iconed, never a mix; a touch screen, which cannot hover,
  shows a card's menu button only, not its row of hover shortcuts. A screen
  has one filled (`default`) button: in a generate form that is Generate, so
  an extension's `primary` button renders `secondary` there
  (`ui/layout/ActionButton.tsx`); status badges are neutral. A two-step
  tool (write a prompt, then use it) hands the filled look to the second
  step once there is a result.
- **Structure is felt, not seen.** Group with space and surface tone before
  borders; one border tone (`--border`); no card inside a card. Radii follow
  hierarchy (controls `sm`/`md`, panels `lg`), not one radius everywhere;
  shadows only on what floats (menus, dialogs, dragged items, and the
  slider's cap, the handle a drag holds); a segmented control's selection is
  a tone, not a raised pill. One depth cue per element, never an edge and
  a shadow together, and no one-sided edge: a field (input, select, number,
  the prompt editor, `FIELD_BUTTON_CLASS`) is a flat fill (`--field`) with
  no underline or bottom shadow, and focus rings it evenly (`border-ring`
  plus a 3 px `ring-focus-ring`). Image tiles (network cards) keep a hairline
  ring and no shadow. A focus ring over an image is drawn where the image
  cannot cover it: an inset ring is a box shadow and the tile's own outline
  is painted before its content, so an image that fills the tile hides
  both once it is on a layer of its own (a hover zoom, a selection layer
  over it). It is an outline on the tile's `after:` layer
  (`focus-visible:after:outline-3 focus-visible:after:-outline-offset-3`;
  the gallery's tiles), or a ring outside the tile where its scroller keeps
  4 px of room (the Extras strips), which leaves a selection mark inside
  visible with it. A star mark uses `--star` (≥ 3:1 on the paper), not a
  fixed amber. Touch screens get no number steppers (the numeric keyboard
  types the value), and controls hidden until hover take no taps while
  hidden. A scroller (the JSON Builder column, `FadeStrip`) keeps 4 px of
  room around its controls so their focus rings are not clipped, and so does
  a control at an edge of `main`, which clips (the queue's mode tabs: their
  tab list's padding is as wide as a tab's focus ring). Toolbar
  controls (header, settings, extensions, gallery) are one 28 px height.
  A pressed toggle wears the selection tint (`--primary` text on a 15 %
  wash, 20 % in the dark theme, 25 % there when hovered), set once in
  `components/ui/toggle.tsx` (`PRESSED_TINT_CLASS`, which a button acting as
  a toggle wears as well); a near-black tone alone does not show a lone
  toggle's state, and neither does a variant swapped once pressed (a filled
  button; `lib/design-guards.test.ts` checks for hand-written tints and
  such swaps). A toggle (`aria-pressed`) keeps one name, and its pressed
  state says on or off: a star is "Star" pressed, a pin "Pin … to quick
  settings" pressed, a layer's eye "Show …" pressed, the prompt's tag view
  "Show as tags" pressed; its glyph stays put unless the glyph itself draws
  the state (the layer eye). A button named for what its next click does
  (the queue's Pause / Resume, the settings' Review / Show all, the panes'
  Hide / Show, a secret's Show / Hide) is an action without `aria-pressed`
  (`design-guards.test.ts`, "toggle names"). A toggle that draws its
  pressed state its own way is listed in that check with the reason: a tag
  chip is a solid `--primary` fill once pressed, as unpressed chips already
  wear a primary wash (up to 32 %) for how often the tag occurs and a 15 %
  tint would read as a frequent tag; the chosen swatch of a swatch picker
  (the RGBA backdrops) is ringed, as a wash would tint the swatch; the
  viewer's toggles, white on its always-black backdrop where the accent tint
  does not read with a dark accent, are a 25 % white wash with a full-white
  glyph, well above their 12 % hover. A light theme's hover keeps the 15 % wash: a deeper one
  takes some accents' labels under 4.5:1.
  Truncated names and URLs carry their full text in a `title`. A form column (the generate parameters, an
  extension page, the JSON Builder's side column, the Extensions list) is an inspector, not a
  stack of cards: each block is a panel with a hairline above it and no box,
  fill or radius (`ui/layout/panel.ts`), its title on the column's edge and
  a section's chevron at the row's end; a panel inside a panel (an
  extension in the Extensions list) is a level down, with a fainter hairline
  and a 14 px title. Lists (the queue history) are rows with a hairline
  between them. Section and pane titles are 15 px semibold, below the 20 px
  page title; the settings list's category headings take the 20 px size,
  because each category reads as a page of its own in one long scroller. The slider is a fader: a recessed groove filled with
  `--primary` up to the value and a squared cap with a centre notch (the
  switch's shape family; `--slider-*` tokens). It fills its row; a width
  the caller gives (`w-18`) replaces that.
  Radii are concentric: a child's radius is its parent's minus the padding
  between them, never below `sm`. Floating lists (menus, selects, combobox,
  popover, command) are `rounded-xl` with `p-1`, so their items stay
  `rounded-md`; dialogs are `rounded-xl` and their flush footer takes the
  same radius. A well inside a dialog or popover (a list box, a layer row)
  is a flat `bg-background` fill with no edge: it sits below the popover in
  both themes, where `--muted` does not in dark. A segmented control's or a
  tab strip's `--muted` track takes that tone there in the dark theme
  (`dark:on-popover:bg-background`; `on-popover` is any dialog, alert
  dialog, popover or sheet). A native `select` (the
  table widgets) draws the Select component's chevron from
  `--select-chevron`, whose hex must equal the theme's `--muted-foreground`
  (`lib/theme.test.ts` checks it), and takes `pr-7` for it; a chip or badge inside a row is `rounded-xs`/`rounded-md`,
  never a pill. Overlays fade in 150 ms.
  Floating chrome that sits over a still picture (the image editor's panels
  and the candidate bar) uses `material-float`: a translucent popover
  surface with a backdrop blur, a 1 px top highlight, and an edge plus a
  shadow, tuned separately for light and dark (`--material-float*`). Floating
  layers are the one exception to "never an edge and a shadow together":
  this material, the floating lists (menus, selects, combobox, popover, hover
  card) keep a hairline ring with their shadow, because they
  open over anything (a busy picture, a surface of their own tone) and the
  shadow alone does not hold the outline there. A floating bar (the settings
  apply bar, the gallery's bulk bar) and a drawer take the shadow alone. The
  material stays opaque enough that its text keeps its contrast over pure
  black or white.
  Nothing else is glass: not the header, menus, popovers, the command
  palette, the mobile bottom bar or its More sheet, the progress overlay or
  the live preview (a blur over a surface that redraws every step costs
  sampling time).
  Motion moves `transform` and `opacity` only (a meter fills with
  `scaleX`, not `width`), 120 to 250 ms. Two exceptions keep to the same
  range and are neutralised by the global reduced-motion rule like the
  rest: a collapsible (a parameter section and group,
  `ui/layout/Section.tsx` and `Group.tsx`, the vendored accordion)
  animates its `height` over 200 ms, because its content's height is
  not known ahead and a transform would not move what follows it; and
  feedback that only repaints (hover, press, focus, selection) fades its
  colour, its focus ring's box shadow or a filter (`transition-colors`,
  `transition-[color,box-shadow]` and the like, at Tailwind's 150 ms
  default). A lone small control (a badge's
  ×, the advice bar's buttons) keeps its look and gets
  an invisible `after:` hit box of at least 40 px; the box never reaches a
  neighbouring control, so packed toolbars are not given one (it would take
  the taps of the next tool). A small control between neighbours (a field's
  reset dot, between its label and the row's controls) gets margin that makes
  the room for a 32 px box up to them. A box that reaches past a scroller's padding
  makes the scroller scroll: give the scroller the box's reach as room, or
  clip the axis it does not scroll along (`overflow-x-hidden` on the parameters
  under a result). Chart and band colours come from `--chart-2`
  to `--chart-5`, never fixed Tailwind hues (`--chart-1` follows the accent
  and can match another band).
  Controls in one row share one look: a button among selects (the header's
  search and VAE / TE, the gallery's date and star) wears the filled field
  style (`FIELD_BUTTON_CLASS`), and everything that opens a list shows the
  same chevron. Empty areas follow two patterns: a drop zone (dashed box,
  round tinted icon, a 14 px title, a muted hint: img2img, PNG Info) and a
  placeholder (a dimmed glyph over one muted line: the stage, Extras, the
  history). Error views follow three: a list or panel that could not load is
  `LoadFailed` (what failed in a destructive line, then Retry; never an empty
  list; another way out follows the Retry as its children, a license text's
  page on the web), a page that could not load is `PagePlaceholder` with a Retry
  (`RetryButton`), and both stay while their retry runs, the Retry busy but
  focused (`useLoadFailure` in `api/load-failure.ts`: a refetch of a query with
  no data goes back to "pending" and drops its error; a load that is no query,
  the image editor's picture or the recipe capture's read of a form, keeps its
  error in its own state and marks it retrying instead of clearing it). When
  the retry brings the data, the focus the failure view held goes to what
  took its place, its first tabbable element or else the nearest focusable
  container (a dialog, `#main`), never to `<body>` (a line that nothing
  replaces, a later page of the network list over the cards that stayed,
  hands it to the content after it); a focus the user moved
  elsewhere stays (`useFocusAfterRetry` in `components/focus-after-retry.ts`,
  built into `LoadFailed` and `RetryButton`; a Retry of its own puts its ref
  on its view). The third, an
  action's error, is a plain destructive line by the action (`role="alert"`,
  `text-sm text-destructive`, no tinted box; the settings import's skipped
  values too, read with the rest of its check in that check's polite live
  region, and a failed Extras run in its result area, not announced again:
  its toast says it). A control that failed to render (`ErrorBoundary`
  inline) says so in the same plain line at its field message's `text-xs`.
  A destructive fill is left to one tone among others (a text output's
  status, workflow advice) and to round tinted icons. The network browser's
  pane, a next page that failed (`LoadMoreStatus`), the image editor's stage
  and the model picker keep their own Retry (the pane's, the stage's and the
  picker's stay up and busy the same way); `components/error-views.test.ts`
  checks every class and every Retry against this, announced or not.
  Image surfaces (the stage, Extras, the image editor) are one borderless
  `bg-canvas` (the JSON Builder's layout canvas too). In the light theme the
  page and the fields are separate steps of lightness, and control tracks
  (slider, switch) are dark enough to read as controls.
- **Type carries hierarchy.** One scale: 0.625 rem (badges, tags on
  thumbnails), 0.6875 rem (the bottom bar, dense metadata), `text-xs`,
  0.8 rem (compact controls), `text-sm`, 0.9375 rem (section and pane
  titles), `text-xl` (the page title); no other arbitrary sizes. (Written as
  `text-[…rem]`, not theme names: `cn` would take an unknown `text-*` name
  for a colour and drop it beside one.) The one relative size is inline code
  in prose (`ui/rich-text.tsx`, `0.85em`): mono drawn a step smaller than the
  text around it, whatever that text's size (`lib/design-guards.test.ts`
  holds every other size to the scale). Size and weight, then colour; muted text keeps
  the surface's hue. Sentence case everywhere: no all-caps labels, no eyebrow
  labels above headings, no numbered markers unless the content is a
  sequence. Mono is for strings a person copies (keys, hashes, code), not for
  decoration.
- **Words do one job.** Buttons say what happens ("Add to queue", not
  "Submit") and the toast repeats the verb; errors say what failed and how to
  fix it; an empty state says what to do next. No `→` welded to buttons.
  The middle dot separates facts on one compact metadata line (a queue row,
  a model's architecture and size, the window title); a header, a dialog or
  a details table lays its facts out instead.
- **Motion answers the user.** Opening, expanding, confirming, reordering
  animate briefly (`--ease-out-soft`); nothing animates by itself, and nothing
  that stays on screen during a job loops (see Motion above).
- **Every state is designed.** Hover, focus, active, disabled, loading, empty
  and error, in both themes and both densities, from a 390 px phone up; text
  and controls keep 4.5:1 (`tests/e2e/a11y.spec.ts` scans every accent).
- **Check with screenshots.** `pnpm dev:mock`, both themes (`?__theme=light`,
  `?__theme=dark`), the real states (a running job, a result, open menus);
  messy content (long names, CJK, 4 000 networks) before calling it done.

## API layer (`src/api`)

- `gen/schema.d.ts`: types generated from the server's OpenAPI schema
  (`openapi-typescript`, pinned): `$PYTHON
  tools/export_openapi.py` writes `src/api/openapi.json`, then `pnpm gen:api`
  regenerates the types; `gen/gen.test.ts` fails when they drift.
- `http.ts`: the path-typed client (`openapi-fetch`, pinned) on the mock
  switch's transport, `unwrap()` (data, or `ApiError` for any non-2xx: status,
  decoded body in `detail`, a readable message from Remi's
  `{"errors": [{owner, param, message}]}` or FastAPI's `detail`),
  `validationErrors(error)` for the per-field list, `ROOT` / `API_BASE` /
  `apiUrl()` from `window.__REMI_ROOT__` (`--subpath`), and a 401 outside
  `/auth` marks the session as ended (`session.ts`).
- `session.ts` + `features/auth`: signing in (`--auth`). The server renders
  no login page; `main.tsx` asks `api.auth.status()` before the first
  render. Signed out, `AuthGate` renders only the sign-in form (the app
  requests nothing) and reloads the page once signed in (the app, or the
  API docs the browser asked for). A 401 while the app is open shows the
  form in a dialog over it (the app stays mounted, `inert`); signing in
  closes it, refetches the queries and reconnects the socket. "Sign out
  (user)" is in the Appearance menu and the command palette. Strings come
  from the chrome catalog like every other text. The mock backend has a
  login too (`api/mock/auth.ts`, `sessionStorage["remi-mock-login"]`,
  demo / demo; `tests/e2e/auth.spec.ts`).
- `endpoints.ts`: **the only place that builds URLs** (`api.app()`,
  `api.schema(tab)`, `api.action(owner, action, body)`, `api.jobs.*`,
  `api.upload(file)`, `api.fileUrl(id)`, `api.thumbUrl(id, w)`, …). Paths,
  params and bodies are checked against the generated `paths`; request
  fields the server defaults are filled in here.
- `types.ts`: aliases of the generated models (`Job`, `JobOutput`,
  `Bootstrap`, …); only free-form JSON the server types as `dict` (schemas,
  param values, action patches, `JobRequest.core/scripts`) is refined by hand.
- `queries.ts`: the `QueryClient` (no retries on 4xx), the `keys` factory and
  `queryOptions` per resource. Schemas and bootstrap are `staleTime: Infinity`
  and are invalidated by WS events.
- `ws.ts`: `useWs` (Zustand). One connection to `/api/v2/ws`; on open sends
  `{"type":"hello","features":[…],"visible":…}` (and `{"type":"visibility",
  "visible":…}` when the page is hidden or shown: a hidden page gets no
  previews) and stores the server's `hello.features`
  (`useServerFeature`). JSON events are dispatched to `onWsEvent(type, fn)`
  subscribers (`"*"` for all); both `{type, …fields}` and
  `{type, data: {…}}` envelopes are accepted, and `events.ts` validates the
  required fields per type (unknown or malformed events are dropped). Binary frames
  (`u8 type, u8 fmt, u32 job_seq, u32 step`, little-endian, then bytes;
  `frames.ts`) become `preview = {jobSeq, step, url}`; the previous object URL
  is revoked, late frames of the same job are dropped. Reconnect: exponential
  backoff 0.5 s → 15 s with ±25 % jitter, immediate retry when the tab becomes
  visible or the network returns. Every connect attempt carries a token, so
  overlapping attempts (StrictMode's connect → disconnect → connect, a
  wake-up during a pending open) never leave two live sockets.
- `progress.ts`: the running job (global: one worker), the jobs this client
  queued (`track(jobId, tab)` after `POST /jobs` or a retry) and the newest
  result **per form** (`lastResult(s, tab)`; other clients' jobs never show;
  a job that finishes before its POST answer is attached when tracked), and
  model loading. Fed by `app/ws-bridge.ts`, which also clears the live
  preview on `job.started` / completion (the panel only shows frames whose
  `job_seq` is the running job's), invalidates queries on `queue.changed`,
  job events, `options.changed` (incl. `["schema"]`, `["page"]`, models and
  styles: core forms and the model lists follow the options),
  `networks.changed` (networks, models) and everything after a reconnect or a
  `resync` (`pong` is accepted and ignored). Model loading
  ends on `model.loaded`, a reconnect, or after 180 s without news (the
  server's only `model.loading` stage is "loading"; the bridge also ends it on
  a stage `done|loaded|error|failed`, for an older server).

Build: chunks and their preloads resolve relative to the importing module
(`experimental.renderBuiltUrl`), so the app works under `--subpath`; the
server prefixes `index.html`'s absolute URLs. A tab opened before an update
that asks for a chunk the new build lacks loads the new build
(`lib/chunk-reload.ts`): when a page or control could not load its chunk
(`retryableLazy`, a popup asked to open: `loadForUse` in
`lib/lazy-module.ts`, or the prompt tools' completion engine at the first
completion), the app fetches the shell and reloads only if it starts
other module scripts than this page's, at most once a minute. A failed
prefetch (a nav item or a popup trigger on hover) never reloads, nor does a
chunk that failed while the server is down: the page's error view, with "Try
again", stays. A module fetched on intent is a `lazyModule`, which forgets a
failed fetch so the next intent fetches again (a promise kept with
`x ??= import(...)` would keep the control broken for the session;
`lib/lazy-module.test.ts` fails on one).
`index.html` carries `darkreader-lock` (the app has its own dark theme).

## Schema renderer (`src/ui`)

The generation pages and Extras load a tab with `useFormLoader`
(`pages/generate/use-form-loader.ts`: loads the schema before the first paint;
on a reload it keeps uploaded images, moves untouched params to their new
defaults and re-derives patch locks;
`ensureSchemaLoaded` does the same for code that reads a form away from its
page). The generation pages render it with `CoreForm` and `ScriptsPanel`,
Extras with `RenderLayout` directly; both build on `RenderNode` /
`RenderLayout`. `ui/SchemaForm.tsx` (`TabForm`, `PageForm`) is a harness for
the renderer's tests only. `jobRequest(store, schema)` (`ui/tab.ts`) builds the
`POST /jobs` body.

| file | role |
| --- | --- |
| `schema.ts` | TypeScript mirror of schema-format.md |
| `Renderer.tsx` | `RenderNode`: dispatch by `type`, `visible_if`/`enabled_if`, patch visibility, row cells |
| `inputs/*` | one component per input kind: Toggle, Slider (+ number box), Number, Text (`prompt` → lazy `PromptEditor`), Choice (select / segmented or radio list / searchable; `choice_hints` as option tooltips), MultiChoice (chips or checkboxes), Seed (dice / recycle), Color, Image, ImageList, File (`DropZone`: drop, paste, click → `POST /uploads`), Json, Custom |
| `paste-targets.ts` | a pasted image goes to the image field under the pointer or holding the focus (`DropZone`, the image editor); with none, to the first empty image field on screen, else the last one on screen (A1111's `dragdrop.js` rule). Image inputs and the editor register while mounted |
| `layout/*` | Section (enable toggle in header, open state, section reset), Group, Tabs (`bind`), ActionButton, Gallery/Output, Slot, row sizing |
| `Field.tsx` | label row, modified dot (click = reset to default, tooltip shows the default), info text, server validation message (`FieldError`); `HintedLabel`: the param's `hint` as the label's tooltip (dotted underline) and as the control's accessible description (`hintRef`) — the former `hints.js` map and `tooltip=` |
| `a11y.ts` | ids of label / info / error texts and `describedBy(id, info, error)` for controls |
| `conditions.ts` | condition evaluator (`param` / `eq` / `ne` / `in` / `gt…lte` / `all` / `any` / `not`, `"owner.param"`) |
| `form-store.ts` | per-form Zustand store (`getFormStore(formId)`); `setValues`/`applyPatches` skip params locked by a toggle's `locks` (`locks.ts`). Both return the params a lock kept from changing and, unless written with `{ quiet: true }`, name them themselves in an info toast ("Overridden by a lock: …", `reportLocked`: one toast per form for the writes of a task, a later write replaces it), so a new helper reports without trying; `{ whole: true }` writes values that only make sense together all or nothing (the background swap keeps the image and its painted layers when a lock keeps either; applying styles keeps the prompts and the style selection, which still applies when the job runs, when a lock keeps a prompt; a card's prompt and negative text go in together). The form's own helpers rely on that: size presets and "W x H" paste, the resolution-presets widget, ControlNet's "use the image size", the size taken from an uploaded img2img image (`img2img_autosize`, not taken when a lock keeps the image), Reuse seed, applying styles, inserting a card into the prompt, inserting a reference's slot token (`image-list`; with a lock on the prompt it is not typed into the editor either), clearing the prompts, region boxes (a move, a layout, a region's on switch), the daemon tab a `sigma-curve` thumbnail selects, the background swap, the image editor's layers and mask uploads, `send-button`'s prompt send, ControlNet's merge of a pasted unit (it writes only units the paste wrote, so a lock never stops it), settings discard and import, an action patch without a toast of its own, an advice fix. A widget writing a param other than its own uses `setValues`, never `setValue` (a user edit, which a lock does not stop), and feedback of its own does not claim a write a lock kept: a card kept out is not announced as added, `send-button` does not say "Added to the prompt", Send to reference switches ImageStitch on only for references that arrived. A path with a toast of its own writes quietly, names them there with `lockNote` and leaves them out of its count: paste, a recipe, Send to and "Use these parameters", an opened or reused job, a lineage source job, Compare's "Apply A/B to form", the JSON Builder's send, `pp-send`, Send to reference, a send to an extension page (`sendToPage`, `send-button`'s image send), a preset or another action with a toast (`deliver` and `sendToPage` resolve with what a lock kept). Values nobody asked for stay silent by design (quiet): defaults carried over on load, the quick settings' sync and their revert after a failed save, uploads restored on a page reload, and the `__conflicts__` pass (it runs on load, after each paste and on conflict param edits); a recipe also names its values the conflict locks change. `persist: false` inputs are never drafted |
| `messages.ts` | translation of server texts (toasts, text outputs, `{text, params}` messages); a merged action toast's `parts` are translated one by one and joined by " · " (`toastText`) |
| `actions.ts` | `runAction` + patch application, `runChangeEffects` (on_change + conflict locks after an edit), `resetParams` |
| `uploads.ts` | upload + local previews (object URLs revoked when unreferenced, at most 24 kept) |
| `RichText.tsx`, `rich-text.tsx` | Markdown → React elements (no raw HTML); HTML sanitised via DOMParser → React elements (allow-list, no `innerHTML`) |
| `widgets.ts` | custom widget registry |

Rules the renderer implements:

- **Owners**: `core` for the generation form, the script id for scripts,
  `page:<id>` for pages. Hooks: `useParam(node)` binds an input
  (value, default, modified, override, enabled, `setValue`, `reset`);
  `useCondition`, `useFormState`, `useRunAction` (`ui/hooks.ts`).
- **Form store** (one per form id: `txt2img`, `img2img`, `extras`,
  `page:<id>`): values and defaults per owner, action overrides
  (`choices/visible/enabled/label/info/minimum/maximum/step`), outputs, open
  sections, selected tabs. Drafts persist under `remi-form:<formId>`
  (writes coalesced to 300 ms and flushed on page hide), twice
  (`draftStorage`): in `sessionStorage`, this tab's own draft, and in
  `localStorage`, the latest draft of any tab. A draft is read from this
  tab's own first, so with two browser tabs open each one's reload brings
  back its own prompt, and a new tab (no draft of its own) starts from the
  most recent one; a duplicated tab copies its original's. A write that
  changes nothing this tab last read or wrote (cleared errors, a schema
  reload) leaves the latest draft to the tab that changed it. Decision:
  drafts are not kept in step across open tabs (no `storage` listener): a
  prompt must not change under the user because another tab was edited.
  On load, a draft value is kept only if it still fits the node (type,
  choice membership) and is coerced like the server does (range clamp,
  integer rounding, `number.precision`); upload ids (`image`, `image_list`,
  `file`) and hidden `state` are never persisted. A schema reload keeps
  current values. The store also holds the schema's `conflict_params` and
  the last rejected submit's `errors` (not persisted).
- **Visibility**: `visible_if` hides a node and its subtree; a patch
  `visible: false` hides an input (AND). `enabled_if` disables the subtree;
  patch `enabled: false` disables an input. `inactive_if` dims an input
  (`data-inactive`, full opacity again on hover or focus, an off-screen
  "Not used with the current values" note) without disabling it.
  `hidden: true` and `state` inputs
  are not rendered but are submitted. A `tab` with `visible_if` false has
  neither trigger nor content (an unbound strip then shows its first visible
  tab); its `enabled_if` disables its content. Patch `label` / `info` replace
  the static text; `null` clears it.
- **Conditions** follow Python (`remi/ui/conditions.py`): `True == 1`,
  `in` uses that equality, `gt…lte` order numbers (booleans as 0/1) and
  strings (by code point); any other pair, `null` or a missing param is false.
- **Actions**: body `{tab, values, core?, scripts?, …extra}`: `values` is the
  calling owner's, `core` the generation form's (for script owners) and
  `scripts` every other non-core owner's values by id, so cross-owner actions
  see the live form. Patches apply to the calling owner or `"owner.param"`;
  `__toast__` → toast, `__outputs__` → the owner's outputs, `__settings__`
  → merged into the cached option values (the server already stored them;
  the client never writes them back). A newer call of the same action wins
  (stale responses are dropped).
- **Change effects** (`runChangeEffects`): after a user edit (or reset) of a
  param, its `on_change` action runs — immediately for discrete kinds,
  debounced 300 ms for slider / number / text / color / json — and, when
  `"owner.param"` is in the tab schema's `conflict_params`, so does
  `core.__conflicts__` with `selected` = that reference. This covers section
  enable toggles, choices, toggling off, and resets. A section reset or
  "Reset all parameters" (`resetOwners`, over every owner at once) runs each
  distinct `on_change` once and one conflicts call (`selected` only when a
  single conflict param changed). A param's `on_input` action runs only
  after the user edits that control (`useParam`): never after pastes,
  presets, bulk resets or action patches. "Reset all parameters" asks first when the
  prompts or a source image would be lost.
  Values set by a patch do not trigger effects (no loops). After a schema
  load, if a draft restored a non-default conflict param, the locks are
  re-derived with one `__conflicts__` call (patch overrides are not persisted).
- **Validation errors**: a rejected `POST /jobs` (`{"errors": [...]}`) sets
  `errors[owner][param]` in the form store; the field shows the message
  (`aria-invalid`, `aria-describedby`), the sections / titled groups /
  unbound tabs around it are opened, and a toast lists all messages. Editing
  or resetting a field clears its error; a successful submit clears all.
- **Slots**: scripts render in the core layout's `slot` whose `section`
  matches, sorted by `order`; unknown/absent sections go to `scripts`, which
  is appended after the core layout if the schema has no such slot. A script
  whose layout is not a section or titled group (e.g. an untitled group,
  the `sampler` row) gets a titled frame; an untitled group's own box is
  replaced by it. The extensions panel lists every untitled script as a
  closed section titled by the script.
- **Uploads**: images are previewed from object URLs; removing / replacing
  an image revokes its URL once no form store references the id, older
  unreferenced previews are swept before each upload, and at most 24 are
  kept (then `/files/{id}` is used).
- **Keyboard** (generate view): Ctrl/Cmd+Enter generates (ignores key repeat
  and IME composition); while this form's job runs it interrupts and, unless
  `ctrl_enter_interrupt` is on, generates again once the job has ended.
  Ctrl/Cmd+Shift+Enter adds the form to the queue ("Add to queue").
  Alt+Enter skips. Escape interrupts only when nothing else owns it: not in a
  text field, not inside a dialog, menu or popup, and no popup open. In a
  prompt editor it interrupts as in A1111 (the placeholder advertises it)
  unless the editor used it first (closing the completion list, dropping a
  selection: CodeMirror calls `preventDefault`). The interrupt is decided
  after the event has run its course (`setTimeout`), so the image editor's
  Escape (discard a staged candidate) wins whichever of the two listened
  first. Generate keys are ignored while a modal dialog is open, not while
  a popover is (`lib/modal-open.ts`).
- **Rows**: `width` 1–12 = share of 12 columns, `"fill"` = the rest,
  absent/`"auto"` = equal share, buttons hug content. Rows are CSS containers;
  under 30 rem spans and fills go full width (phones stack).
- **Prompt editor**: `text` nodes with `prompt: true` get the full editor of
  `features/prompt-editor` (`ui/prompt/PromptEditor.tsx` re-exports it; the
  pure editing functions stay in `ui/prompt/`). See "Generate page".

### Adding a widget

Create `src/widgets/<name>/index.tsx` whose default export is a component
taking `WidgetProps<Value, Props>` from `@/ui/widgets`:

```tsx
import type { WidgetProps } from "@/ui/widgets"

export default function StageTimeline({ value, onChange, props, disabled, id, label }: WidgetProps<number[], { stages: string[] }>) {
  // Put `id` on the primary focusable element; the field label points at it.
}
```

That is all: `import.meta.glob` discovers it at build time, it is loaded
lazily the first time a `custom` node with `widget: "<name>"` renders, and
the field chrome (label, modified dot, reset, info) is provided around it.
An unknown widget name falls back to the JSON editor so values are never
lost. Example: `widgets/block-weights`. Shared widgets with a server-side
node: `widgets/range-slider` (`ui.RangeSlider`).

A widget may also edit an input of another kind through its `widget` hint
(today `image_list`: `ui.ImageList(name, widget="image-list")`). The value
keeps that kind's rules (an image list's upload ids are never restored from
drafts, and count as source images), and the widget receives the node as
`WidgetNode`. An async widget edit (an upload) must commit against the
store's current value (`store.getState().values[owner][node.name]`), not the
value of the render that started it (see `widgets/image-list`).

A widget may sum up its values in the header of the section holding it
(shown while the section is collapsed, like the old ControlNet "2x Unit"
badge): a `src/widgets/<name>/summary.ts` exporting
`sectionSummary(values, nodes)` (the owner's values, the section's inputs of
that widget) returning translated text or `null`. It is imported eagerly
(`ui/section-summary.ts`), so keep it free of heavy imports. Example:
`widgets/controlnet-unit/summary.ts`.

### Adding a page

- Built-in page: create `src/pages/<page>/index.tsx` (default export
  `({ route })`), add the route name to `router.ts`, its loader in
  `routes.tsx` and a `BUILTIN_NAV` entry.
- Extension page with a dedicated view (`Page(..., widget="x")`): create
  `src/pages/custom/x/index.tsx` (default export `({ page })`; example:
  `custom/json-builder`, the JSON Builder, which keeps its draft in a
  persisted store and sends with `deliver()` from `features/send-to`). Without
  `widget`, the page is schema-rendered from `GET /api/v2/pages/{id}` by
  `pages/ext` (`PageView`): its form `page:<id>` is loaded once per schema
  object and kept across visits (values, upload ids and outputs survive
  leaving the page and a schema reload), and its `on_load` action runs each
  time the page is shown. Other views send values to a page with
  `sendToPage(id, values)` (`pages/ext/deliver.ts`: applied at once to a
  loaded page, else right after its first load; each value's `on_change`
  runs). Pages declaring `send_to` image targets get "Send to *label*"
  buttons wherever `PageSendButtons` (`pages/ext/send-targets.tsx`) is
  placed: PNG Info's "Send image" group.
- Page widgets: `task-runner` (start a page task and follow it: progress,
  status text, Cancel; the poll outlives the view, so a result lands even
  after leaving the page), `send-button` (a param of the page to the
  txt2img / img2img prompt — append with the old ", " joining, or replace,
  with or without switching — to another page's image input, optionally
  after an action such as a download, or an http(s) link in a new tab) and
  `secret-text` (a masked, write-only text such as an API key; declare it
  `persist=False, submit=False`).
- Extension page icons are names from the curated map in `app/icons.ts`
  (unknown → puzzle piece).
- The **Checkpoint Merger** (A1111's `modelmerger` tab) is such a
  schema-rendered page, from core: `remi/core/merger.py`, registered by
  `scripts/checkpoint_merger.py` so it follows "Reload UI" like extension
  pages (and "UI Tab Order" / "Hide UI Tabs" by its title). Its merge and
  the two "Save Current Model" buttons are `task-runner` tasks behind
  generation (progress per stage and per tensor, Cancel between steps); the
  new file reaches the checkpoint lists through `networks.changed`. None of
  them replaces a file already in the folder (a merge's generated name
  aside), checked again once the task holds the generation lock.

## Generate page (`pages/generate`, `features/`)

txt2img and img2img share one view; the mode switch (two links) sits in the
header slot, and each mode has its own form store and schema (the other
mode's schema is prefetched).

- **Layout**: desktop ≥ 72 rem: three resizable, collapsible panes
  (parameters | stage | history; `components/ui/resizable.tsx`, the subset of
  react-resizable-panels' API the app uses, sizes persisted under its key);
  48–72 rem: parameters | stage with a history strip; phones: stage and
  history strip first, then the parameters with the run controls stuck to
  the bottom (and the stacked model bar). The run controls are sticky at the
  top of the parameters pane; with the pane collapsed they move to the stage
  header. A short window (a phone held sideways) scrolls the stage column:
  the img2img Source view keeps the editor's 16rem floor (`SOURCE_FLOOR`
  holds the stage open) and the history strip follows it, never under it;
  the result picture has no floor and scales down to the room there is.
- **Form loading** (`use-form-loader.ts`): the page loads its schema itself
  (not `TabForm`), once per schema object. On a reload (options changed, UI
  preset) image ids survive and params still at their old default follow
  the new default (`carryOver`).
- **Curated form** (`curate.ts`): prompt, negative prompt, styles, the
  `prompt_before` slot (its scripts, e.g. Prompt Anchor, render at the top
  of `PromptsPanel`, above the main prompt) and the `script` choice are
  taken out of the core layout; the remaining top-level
  blocks are classified by content (source image, sampler slot, size,
  sections, guidance, seed, other) only to frame them (cards, size tools);
  they keep the schema's order, so the user's category order
  (`ui_reorder_list`) decides the layout, and render with the schema
  renderer, so server-side layout changes still show. Size gets aspect
  presets (keeping the pixel budget, 64-px grid) and "W x H" typing / paste
  (paste anywhere in the size card). Scripts without a slot of their own are
  a searchable, collapsible list with an "only modified" filter, per-script
  reset and "reset all"; the selectable-script choice and the chosen script
  follow. img2img's source blocks render in `SourceSlot` (`CoreForm.tsx`)
  as the canvas module's `Img2imgSourcePanel`; the image is edited on the
  stage (see "Image editor").
- **Prompt editor** (`features/prompt-editor`, CodeMirror 6, lazy chunk):
  colouring from `tokenize.ts` (attention weights, `[a:b:0.5]` / `[a|b]`
  schedules, `<lora:…>` with numeric arguments, `BREAK` / `AND`,
  `__wildcards__`, `{a|b}`, escapes, `#` comment lines, bracket depth),
  bracket-balance warnings (lint underline plus a summary in a polite live
  region, so screen readers announce it; replaces prompt-bracket-checker),
  Ctrl/Cmd+↑/↓ attention with the `keyedit_*` precisions, Alt+←/→ tag
  moves, undo/redo (600 ms grouping; always on, so the Gradio-era
  `undo_redo` setting has nothing left to switch, and `prompt_debounce` is
  moot: typing never makes a request — only the token count does, debounced),
  image drops (PNG `tEXt`/`iTXt`/`zTXt`, JPEG/WebP EXIF UserComment read in
  the browser, else upload + `/infotext/image`). `PromptField` adds the label,
  reset, token counter (`POST /prompt/tokens`, 700 ms after the text and the
  steps stop changing, superseded requests aborted, last count kept; counts
  taken with no model loaded are redone after `model.loaded` or a finished job) and page actions: paste
  parameters (from the prompt text, or the last generation when empty) and
  clear (confirmed). Styles: the `styles` multiselect, "apply to prompts"
  (`POST /styles/apply`, clears the selection) and a style editor dialog
  (`/styles` CRUD; refreshes the schemas).
  **Completion sources**: `registerCompletionSource(id, source, {priority})`
  from `@/features/prompt-editor` adds a CodeMirror `CompletionSource` to every
  open and future editor (no autocompletion extension while none is
  registered); sources read `promptKind` (`facets.ts`) to tell positive from
  negative prompts. Tab accepts like Enter; `completionAcceptKey()` tells
  them apart inside a completion's `apply`; options may carry `className`
  and `decorate()` (`DecoratedCompletion`: a class on the row, a node after
  the label). **Prompt decorators**: `registerPromptDecorator(id,
  Component)` wraps every prompt editor (`PromptField` and the renderer's
  `prompt: true` text fields, via `PromptDecorators.tsx`); the component
  gets the editor as `children` (keep it mounted), the value and setter,
  the kind, the label and the editor handle.
- **Run controller** (`features/progress`): `run-machine.ts` is a pure
  reducer (Generate / queue another / Ctrl+Enter restart / Interrupt (also
  cancels this form's queued jobs and a submit in flight) / Skip / generate
  forever / repeat N / restore); `run-controller.ts` performs its effects
  (`POST /jobs`, `/interrupt`, `/skip`, `DELETE`) and feeds it WS job events
  (`job.completed` with status `interrupted` / `cancelled` counts as
  interrupted). Forever and repeat submit the next run when the previous one
  has finished, from the form's values at that moment, so edits made
  meanwhile apply to the next run as in A1111 (a failure or Interrupt ends
  them). They are driven by WebSocket job events, not timers or animation
  frames, so they keep going with the tab in the background. Submits in flight are a
  counter (an Interrupt cancels every one of them when it answers; a second
  Ctrl/Cmd+Enter while one is pending queues nothing). Every submit carries a
  fresh `Idempotency-Key` (api.md); a second Generate press within 400 ms
  (`DOUBLE_PRESS_MS`) of one that has no answer yet is a double click and is
  sent with the first press's key, so the server queues one job and answers
  both (the repeated answer adds nothing); later presses queue another job. Jobs in flight are
  kept in `sessionStorage` (`remi-run:<tab>`: `{jobs, forever, repeatLeft}`,
  written from the run state), which a reload keeps and other tabs do not
  share, so a tab opened meanwhile never takes a running loop over. A
  duplicated tab starts with a copy of that storage. Decision: a restored
  run belongs to one tab. Each tab keeps an id in its session storage
  (`remi-tab-id`, `lib/tab-identity.ts`), and at boot (`main.tsx`) a tab
  that has one asks the browser's other tabs on the `remi-tabs`
  BroadcastChannel whether one of them is open with it (`sessionIsOwn`).
  A tab that is answers, and the asking tab is its copy: it takes a new id
  and drops the runs it copied without touching them on the server (their
  progress and results stay the original's; finished results it copied stay
  in its session gallery). It drops them at boot whatever page it shows
  (`features/progress/copied-runs.ts`, which loads the run modules only in
  a copy): left in its session storage, its next reload, under its new id,
  would take them over. No answer within 250 ms is a reload (or the
  original was closed), which restores as before; a tab without an id is
  new, or stored by a build before the ids, and restores at once. Two tabs
  asking at once (a tab duplicated while its original reloads) leave the id
  to the one whose page load sorts first. Not Web Locks: a server reached
  with `--listen` over plain HTTP is not a secure context. The same holds
  for the Extras page's jobs in flight (`remi-extras-runs`). A loop the
  restoring tab copied is still checked with the other tabs before it
  resumes (`loop-owner.ts`, the `remi-run-loops` channel: one of them may
  run it). The jobs are re-attached after a reload (`restore.ts`), once the
  tab knows they are its own (the run is read first, so a job queued while
  it asks is kept): attached first, then each looked up with `GET /jobs/{id}`, so events
  arriving meanwhile apply; finished ones go to the history. After the
  lookups, Generate forever and the runs left of Generate several times
  resume (`resume` event: the menu item is checked again and the next run
  follows) if one of the restored jobs is still queued or running; when they
  all ended while the page was closed, the loop is over and nothing new is
  queued. Events lost while the socket was down are recovered the
  same way: every `hello` and every `queue.changed` whose `running` is not
  the job a form believes runs re-check that form's jobs (`wireRunSync`), and
  `queue.changed` drops a stale live progress. While a job of
  this client runs: screen wake lock (`prevent_screen_sleep_during_generation`),
  `[42%] ETA 12s · …` in the title (`show_progress_in_title`), and on
  completion with the page in the background the user's `notification.mp3`
  when the server lists it in `user_assets` (`notification_audio`,
  `notification_volume`; silent otherwise, as the setting says) and a browser
  notification (permission asked from the Generate menu; a batch's grid is
  not counted when `return_grid` is on). The status line shows the queue
  position (from `GET /queue/items`, i.e. run order with priorities and
  reordering), step, batch and ETA.
- **Results** (`features/gallery`): this session's results (sessionStorage,
  400 max) in a virtualized thumbnail grid / strip (one listbox tab stop:
  arrows select, Enter opens, Delete removes). The stage shows the selected
  result, or the live preview while a job runs unless an older result was
  picked ("Show live preview" returns). The preview is double-buffered
  (`pages/generate/LivePreview.tsx`): each frame, decoded before it is
  shown (`api/ws.ts`), is stacked over the previous one, so the stage never blinks between
  steps. The legacy UI's crossfade is deliberately not reproduced: a 150 ms
  fade drew about 12 frames per step and tripled the browser's CPU per step,
  against the one-frame-per-step rule (Motion above;
  docs/research/ui-performance.md). The lightbox (lazy chunk): wheel zoom
  at the cursor, two-finger pinch zoom around the fingers' midpoint, drag to
  pan, double-click fit ↔ actual size, ←/→ (or a sideways wheel — tilt
  wheel, horizontal two-finger swipe — at most once per
  `js_modal_lightbox_gamepad_repeat`; or a gamepad's left stick / d-pad
  with `js_modal_lightbox_gamepad`, repeating per
  `js_modal_lightbox_gamepad_repeat`), Home/End,
  S downloads, +/−/0/1, T tile preview, B cycles the RGBA backdrop
  (`features/lightbox/keys.ts`; the shortcuts dialog lists each key there),
  middle click opens in a new tab,
  Esc closes; initial zoom per `js_modal_lightbox_initially_zoomed`; follows
  the live preview with `js_live_preview_in_modal_lightbox`. It opens with
  the focus on the image, whose keys it handles (the popup's
  `initialFocus`), not on its first toolbar button. A job's video
  (output `kind: "video"`, e.g. Wan) belongs to its first frame: the stage
  plays it (controls; `video_player_auto` starts it, `video_player_loop`
  loops it), the history and the lightbox keep the frames.
- **Result actions** (`features/send-to`): download (`/files/{id}?download=1`,
  the server names the file), save to the Save folder or the whole
  generation as a zip (`POST /outputs/save`), open in new tab, copy
  parameters, reuse seed (the param whose `infotext` is `Seed`), use these
  parameters (all of them), send to img2img / inpaint / extras / txt2img.
  On txt2img's stage and lightbox, "Upscale with Hires. fix" (✨, A1111's
  button under the gallery) queues the txt2img form's values with
  `upscale_from` = the result (api.md): not for grids, extra images (output
  `role`) or videos. The upscaled result replaces its source in the history,
  or follows it with `hires_button_gallery_insert` (then
  `hires_insert_index` selects it, otherwise the source stays selected).
  "Send to" parses the infotext for the target form, drops seed / size per
  `send_seed` / `send_size`, puts the output file id into `init_img` (`mode`
  img2img / Inpaint) or Extras `image`, and switches page; values reach a
  form that is not loaded yet after its next schema load (`deliver`). A new
  source image clears the layers drawn over the old one (image params whose
  `source` names it: `sketch`, `mask`).
  Pasted values run each `on_change` once and one `__conflicts__` pass.
  The paste toast names the pasted params the form does not have
  (`PastePlan.skipped`, "Not in this form: …", core ones bare) under the
  values a lock kept.
- **Model bar** (`features/model-bar`, the header's `ModelQuickBar`): UI
  preset (`__forge_preset__` on the form on screen, then all schemas
  refresh), checkpoint (searchable, architecture and size), VAE / text
  encoders (applied when the picker closes, so one change loads the model
  once; the picker says which checkpoints need them and links to where to
  download them, the Gradio UI's "?" next to the dropdown), "Diffusion in
  Low Bits"; option writes are optimistic `PATCH
  /options`; the `model.loading` stage shows next to it. Narrow headers fold
  preset, modules and dtype into a popover.

## Image editor (`src/canvas`)

Replaces ForgeCanvas (core-frontend-features.md §3.33, §10.G). Public
surface (`@/canvas`): `<ImageEditor>` (lazy; Konva and the editor UI are one
chunk, `preloadImageEditor()` fetches it early), the img2img integration
(`Img2imgSourcePanel`, `Img2imgStage`) and the schema binding helpers.

| directory | role |
| --- | --- |
| `core/` | pure logic, unit-tested: geometry and view math, undo `History` (≥ 32 states, byte budget), pixel patches (empty / RLE / raw, masks as alpha), dab placement along strokes, flood fill, tiles, mask export / luminance fallback, resize-frame and "Only masked" area maths (mirrors `modules.images` / `modules.masking`), shortcuts, schema → mode layers (`modes.ts`) |
| `engine/` | `EditorDocument` (background, one raster per layer, history, per-layer upload sync), `CanvasRaster` / `MemoryRaster` (tests), `LayerSync` (debounced, hash-cached uploads), `Interaction` (pointer / wheel / pinch), `Renderer` (the only Konva file), `registry` (documents by key) |
| `state/` | tool settings persisted per browser (`remi-canvas-tools`); `activity.ts` (a painting gesture is in progress, readable from the main bundle) |
| `ui/` | the editor chrome: tool rail, options bar, colour popover, layers popover, shortcuts help, empty state, full-screen dialog |
| `img2img/` | the img2img form binding, the Source panel and the stage's Source / Result switch |
| `form/` | schema-declared layers in any form (`SchemaImageEditor`, used by the renderer's image input) and the undoable "replace the background" write |

- **Documents** live in `engine/registry.ts` by `docKey` (`"img2img"`, a
  ControlNet unit…), so layers and history survive unmounting (switching
  pages, the full-screen dialog, the Result view). A layer is one
  full-resolution offscreen canvas; a stroke is drawn into a scratch raster
  and composited once at the brush opacity (no per-stroke nodes). Undo keeps
  the pixels under the touched 256-px tiles before the edit (compressed) and
  the edit itself for redo. Undo and redo act on the edited layer only
  (never a hidden layer of another mode; edits of different layers commute,
  so the newest entry of that layer is taken out of the shared stack). The
  budget is 1/32 of `navigator.deviceMemory` within 96–384 MiB, 160 MiB when
  the browser does not report it, at most 96 MiB on coarse-pointer devices.
  The painted bounds are tracked by the edits ("Only masked" preview); after
  an erase or undo only the previous bounds are scanned.
- **Values**: the editor follows its `image` / layer values (set from outside
  by send-to, paste, reset: a new background clears its painted layers and
  the history). After an edit, once the user pauses (600 ms, then idle), the
  layer is exported as a PNG in wire format (sketch RGBA; mask white + alpha),
  hashed, uploaded unless the same content was uploaded in the last hour
  (undo back to it; older ids may have been pruned by the server), and
  committed with `onLayerChange`. `pendingKey` registers the
  document with `ui/pending-writes`; the run controller flushes it before
  `POST /jobs`, so the job carries what is on screen. Every upload that sets
  a form value (image, file and image-list fields, img2img's source rows, a
  background dropped on the editor) is a `pendingWrite` of its form until
  the value is set, and every submit (Generate, Process, saving a queued
  job, an action button or widget) waits for them; replacing a background
  inside such a write flushes the layers only (`values: false`), or it
  would wait for itself. Painting waits until
  the document shows the value's image; an image that fails to load clears
  the document and shows an error with Retry. Replacing or removing the
  background flushes the layers first and, when painted layers are dropped
  with it, offers Undo in a toast (`form/replace-background.ts`).
- **Schema layers**: any form's `image` input that `sketch` / `mask` inputs
  of the same owner name as `source` renders as one editor
  (`docKey` `formId:owner:param`, `pendingKey` the form); the layer inputs
  render nothing; the visible ones (own `visible_if`, `hidden`, patches) are
  offered, with a picker when there are several. A `mask` without `source`
  stays an upload field.
- **Modes** come from the schema, not a table (`core/modes.ts`): the image
  input whose name other image inputs' `source` refers to is the background;
  the layer edited is the visible `sketch` / `mask` input; a visible
  `mask_blur` makes a sketch's alpha the inpaint mask. img2img: no layer;
  Sketch / Inpaint sketch: `sketch`; Inpaint: `mask`; Inpaint upload:
  `mask_upload` (uploaded in the Source panel, shown and editable on the
  canvas); Batch: no canvas (the stage shows results).
- **Tools**: brush (size on a log slider, opacity, softness, colour with
  picker / hex / swatches / recent / eyedropper / screen picker), eraser
  (Shift while painting), fill (tolerance; by the layer or the image as
  shown), rectangle and lasso (Shift subtracts), invert mask, clear; masks
  always paint at full strength (the server thresholds at 50 %). Pointer
  events are coalesced and applied once per animation frame; pen pressure
  scales the dab.
- **View**: wheel zooms at the cursor (or pans, a preference; Ctrl+wheel /
  pinch always zoom), Space / right / middle drag / Hand / two fingers pan,
  R or 0 fit, 1 actual size, +/− steps, F full screen. With nothing to
  paint, one finger scrolls the page on touch screens (`touch-action:
  pan-y`); two fingers pan and zoom. The brush cursor shows
  the real diameter; brush sizes are image px, or screen px with
  `forge_canvas_consistent_brush`. Canvases follow `devicePixelRatio` (and
  moves between screens). `forge_canvas_plain(_color)` and
  `forge_canvas_toolbar_always` are honoured. The bottom row holds the
  history and zoom toolbar (its zoom steps fold away below `@md`) and the
  image toolbar; where both do not fit (an editor under about 385 px with
  a keyboard, whose shortcuts button the image toolbar adds), the image
  toolbar scrolls sideways with a fading edge, as the tool options do, and
  the history and zoom one keeps its size.
- **Candidates** (the staging bar): staged results wait for Accept or
  Discard, so their bar stays in view while the other overlays fade
  (`forge_canvas_toolbar_always` off). It keeps to one row: as the canvas
  narrows (`@container/staging`) the candidate dots go first (the counter
  and arrows stay), then the opacity fader, then Accept's label (the icon
  stays, named). Staged with "Stage latest result", whose button leaves with
  the Result view, the bar itself takes the focus (`requestStagingFocus`),
  never Accept or Discard, which a held Enter would press; results staged by
  a finished run leave the focus where it is. While candidates wait, ← →
  pick one, Enter accepts it and Esc discards it, except that Enter on a
  focused control in the editor (Discard, Remove image…) presses that
  control (`takesEnter`), and Enter does nothing while the editor is busy
  (loading, uploading: Accept is disabled then too).
- **Keys** (pointer over the editor - entered, moved or pressed inside, or
  `:hover`, so an editor that appears under a resting pointer takes them -
  or focus inside it, never while typing):
  B E G M L I H tools, `[` `]` and W / A / S + wheel size / opacity /
  softness, Ctrl/⌘+Z / Y / Shift+Z, Ctrl/⌘+X or Delete clear, Ctrl/⌘+I
  invert, `?` the shortcuts popover.
- **Overlays**: while width / height / resize mode change, the target frame
  of the resize mode (crop / fill shading) shows for a moment ("Resize by":
  the image × scale, snapped like `snap_resolution`; the Source panel also
  fills the `scale_by_resolution` output, with the source size from the
  upload's metadata, a sent result's output metadata (`describeFile`), the
  editor's loaded background, or the image read once - shared by every
  reader, `img2img/image-size.ts`); mask blur
  (CSS blur at the view scale) and the "Only masked" area (padding and
  processing ratio, as `modules.masking` computes it) show while their
  settings change, or always from the layers popover.
- **img2img**: the parameters pane's Source card has the mode picker
  (balanced tiles, no wrapping strip), one line per image (thumbnail, size,
  "use image size", replace, remove) and the mode's options from the schema,
  flattened into the card (untitled groups lose their box). The stage has a
  Source / Result switch: it turns to Result when the first job of a run of
  this form starts (not for the next job of Generate forever, a repeat or a
  queue, and never during a stroke; a stroke in progress when the editor
  unmounts is committed) and back to Source when a new source arrives or a
  submit is rejected for a source layer. Upload, paste, drop, replace, remove and a blank canvas of
  the target size are on the editor itself.
- **Widgets** (ControlNet units): `noUpload`, `noScribble`, `fixedColor`,
  `fixedSize`, `fixedOpacity`, `fixedSoftness`,
  `layers={{ mask: { kind: "mask", value } }}` + `activeLayer="mask"` for a
  mask-only editor, `height` (default: the "Canvas Height" setting,
  `forge_canvas_height`) or `fill` (fill a sized container), `hint`, `error`.

## Networks (`features/networks`, `pages/networks`)

One browser (`NetworkBrowser`) serves the full-screen Networks page and the
drawer in Generate (header button "Networks", Alt+N, also in the command
palette; its chunk loads on first open). Each surface keeps its own kind,
search, sort, view, card size and per-kind filters (`store.ts`, persisted as
`remi-networks`, with the drawer's open state and width and the folder
tree's width). Until a sort is chosen, `extra_networks_card_order_field` /
`extra_networks_card_order` decide it.

- **Data**: `GET /networks` (kinds with their card action), fixed windows of
  100 items per `{kind, q, folder, archs, compatible, sort, order}`
  (`useNetworkWindow`: the grid reports the indices it shows, debounced
  120 ms, and only the windows covering them — plus half a window around —
  are fetched with `offset`, so a jump to the end of 5,000 items is one
  request and an invalidation refetches only the windows on screen). The
  first window (with the total) of a previous search is kept while a new
  search of the same kind loads, never across kinds; search is debounced
  180 ms. `/facets` feeds the folder tree and the architecture chips.
  `networks.changed` invalidates everything under `["networks"]`, and so does
  `options.changed` for the options the answers depend on (`forge_preset`,
  `extra_networks_default_multiplier`, `lora_preferred_name`, the hidden
  folder options: `NETWORK_OPTION_KEYS`).
- **Grid** (`NetworkGrid`, TanStack Virtual): rows of cards (3:4 preview,
  name, folder, architecture / LyCORIS badges, a check when on) or list rows
  (thumbnail, path, badges, size, date). The overscan is computed so that at
  most 60 cards are mounted (`grid-layout.ts`); items not loaded yet are
  skeletons. A new kind, search, filter or sort mounts a new grid (keyed by
  them): it starts at the top with no cursor. Previews are `/thumbs` at the
  card's device-pixel width snapped to the server's widths (stable URLs while
  resizing), lazy, over the blurred `placeholder`; video previews are muted
  loops (still with `prefers-reduced-motion`); without one a tinted
  monogram. The card / row is the `option`; its hover buttons are a sibling
  overlay outside the accessibility tree, and the item count's live region
  sits outside the listbox. One listbox tab stop (`aria-multiselectable`, selected =
  on): arrows, Page Up/Down, Home/End, Enter/Space apply, I metadata, E
  editor, Shift+F10 / Menu / "." the item menu; "/" focuses the search.
- **Clicking a card** (`actions.ts`): prompt kinds toggle their text in the
  form's `core.prompt` / `core.negative_prompt` (`prompt-insert.ts`: a
  `<type:name:…>` tag matches whatever its weight and takes the activation
  text after it along; other text matches as a whole word; the separator is
  `extra_networks_add_text_separator`). A card with negative text fills both
  prompts; otherwise the target is the prompt editor focused last (read from
  its `promptKind` facet on `focusin`) or the Prompt / Negative switch.
  Writing through the form store lets the editors undo it; on the Networks
  page the target form is chosen in the header and keeps the text until its
  page loads. Checkpoint cards write `sd_model_checkpoint`, VAE / text
  encoder cards toggle `forge_additional_modules`.
- **Item actions** (hover buttons, one shared Base UI menu: at the pointer
  for a right click, under the "…" button, at the card for Shift+F10):
  metadata dialog (file facts, training facts, dataset tags
  as frequency-weighted chips — click copies, "Copy top 40" / "Copy all" —
  and the raw header, filterable), user-metadata editor (the kind's fields;
  dataset-tag chips toggle words in the activation text; random prompt from
  the tags; the preferred weight is bounded like the server, ±100), set
  preview (this session's results, the paged gallery or an upload; a radio
  group with arrow keys; every opening starts with nothing picked), copy
  path, open folder (only with the bootstrap feature `folders.open`, i.e.
  not with `--hide-ui-dir-config` and only in a browser on the WebUI
  machine). Edit, set preview and open folder are
  hidden for items without a model file (`editable: false`). The
  checkpoint editor no longer offers upstream's unused "Preferred VAE / Text
  encoder(s)" (`vae_te`) field; the key is kept in sidecars that have it.
- **Filters**: architecture chips with counts, "For *preset*" (default on
  for LoRA when `lora_preset_filter` is set; it filters by the *effective*
  architecture, the user's `sd version` or else the one read from the file,
  where upstream only used `sd version`), the folder (a resizable tree pane
  on wide pages, a popover elsewhere; a WAI-ARIA tree with counts whose
  "All" row counts every folder), "Clear filters". Failed requests show an
  error with Retry. On phones the Networks page puts the form and prompt
  target switches in a row above the browser.
- Mock mode serves 5,000 LoRA in nested folders, 454 of them in a hidden
  `styles/.wip` folder (`api/mock/networks.ts`).

## Settings, quick settings, extensions, server control

- **Settings** (`pages/settings`, `#/settings?view=&q=&filter=&section=`):
  `GET /options` is loaded into one form store (`settings-store.ts`, owner
  `settings`, not persisted) whose defaults are the schema's built-in
  `defaults`, so each control's modified dot resets to the default; the
  stored values are kept beside it and their difference is the unsaved
  edits (marked per row and per nav section, reverted per row, "Unsaved"
  filter, sticky Apply bar, Ctrl/Cmd+S). Apply is one `PATCH /options`,
  optimistic; on any refusal every edit stays an edit (rejected ones show the
  server message under the control) until the refetched values settle it.
  An options reload keeps unsaved edits. Edits survive leaving the page; the
  shell asks before the tab closes, and leaving shows a "Review" toast. Every
  setting renders with the schema renderer inside a virtualized list
  (category / section headings, `settings-model.ts` flattens it); search
  covers keys, labels, notes, links, help HTML text and choice labels, in
  English and the active language, all terms required, with matches
  highlighted through the CSS Custom Highlight API (`highlight.ts`; no DOM
  changes). Filters: All / Changed (from the default) / Unsaved / Pinned.
  The search text and filter live in a module store (`settings-filters.ts`)
  so they survive leaving the page, and are mirrored into `q` / `filter`
  (replace navigation, typing after a pause); a route change made elsewhere
  (a "Review" link, back/forward) wins, the page's own writes are not read
  back. Rows carry notes, links, sanitized help HTML, "Requires restart /
  Reload UI" badges, a refresh button (`refresh:<key>` action) and a pin
  toggle. Settings the server refuses (`locked`, schema-format.md §6) are
  disabled with their reason (not repeated per row under `--freeze-settings`,
  where a banner says it); import skips them. Applied settings that need a reload or restart raise a notice
  with the matching command. Export (changed or all values, JSON) and import
  (ours or an A1111 `config.json`; differences validated with
  `POST /options/validate`, then staged as edits). System views: server and
  models (the old footer's version and API documentation link `/docs`, the
  loaded checkpoint's hash with its CivitAI search link — `model_hash` of
  `GET /server/status`, a link the user follows, nothing is fetched —; the
  footer's startup profile is System info → "Startup"; Reload
  UI, restart, unload / load / load again from disk, the
  weights in memory, hash every checkpoint with progress, notification
  permission, settings file, localization template download), system info
  (summary, collapsible sections, download, copy, file check), licenses
  (structured data; license texts fetched on demand; and a link to the
  build's `THIRD-PARTY-NOTICES.txt`, every package the web interface ships
  with its license text, written by `webui/tools/notices.mjs`; the dev
  server answers it with the direct dependencies' notices).
  `tools/dump_settings.py` writes the real schema to
  `api/mock/contract/settings.json` (tests and mock mode).
- **Former Settings tabs and actions** (decisions):
  - *Defaults* (the `ui-config.json` editor) is not carried over: Remi forms
    never read `ui-config.json`; their starting values are the schema
    defaults, the UI preset's defaults and the saved form drafts.
  - *Reload custom script bodies* is covered by "Reload UI": the schemas are
    built from the scripts, so reloading script bodies without rebuilding
    them would leave stale forms (and there is no injected JS/CSS to
    refresh).
  - *List loaded models* → Server and models → "Show the weights in memory"
    (`GET /models/loaded`).
  - *Calculate hash for all checkpoints* → "Hash all checkpoints"
    (`/models/hashes`, background, polled progress).
  - *Download localization template* → "Download localization template":
    the English keys of the SPA catalog, the server catalog and the settings
    schema, with the active language's translations filled in.
  - The `quicksettings_list` "info" link (`settingsHintsShowQuicksettings`,
    a popup of all setting names) is replaced by the pin buttons on every row
    and a "Show pinned settings" link under that setting (the Pinned
    filter); the settings search covers keys and labels.
- **Gradio-era settings that were removed, and what replaces each.** Every
  setting left has a reader (`tests/test_remi_option_consumers.py`); these
  only configured the Gradio page and are covered by controls of the SPA
  that the browser remembers:

  | removed setting(s) | replacement |
  | --- | --- |
  | `prompt_box_style` (Default / Compact / Scrollable / Accordion) | the prompt editors grow with their text up to 24 rem and then scroll (the former Scrollable); the form column of Generate is a resizable pane; the extra networks open in the Networks drawer beside the prompts (Alt+N) rather than the Compact layout's tabs |
  | `txt2img_settings_accordion`, `img2img_settings_accordion` | every section of the form collapses on its own and keeps its open state (`sectionKey`) |
  | `gallery_height` | the result stage fills the page height beside the form (the split between them is a drag-resizable pane) |
  | `do_not_show_images` | not carried over: the Generate page is built around the result stage (turn saving off with `samples_save` to keep results out of the output folder) |
  | `quicksettings_accordion`, `quicksettings_accordion_starts_closed` | quick settings live in the header's ⚡ popover, closed until opened |
  | `extra_networks_card_width`, `_card_height`, `_card_text_scale` | the S / M / L card size and grid / list switches of the network browser, remembered per surface (`remi-networks`) |
  | `extra_networks_card_show_desc`, `extra_networks_card_description_is_html` | descriptions show in the item's metadata dialog as plain text (raw HTML from model files is never rendered) |
  | `extra_networks_tree_view_default_enabled`, `_tree_view_default_width`, `_tree_view_style` (Tree / Dirs), `extra_networks_dir_button_function` | the folder tree pane (toggle and drag-resizable width, remembered) on wide layouts and a folder popover elsewhere; picking a folder filters to it, as the Dirs buttons did |
  | `img2img_sketch_default_brush_color`, `img2img_inpaint_sketch_default_brush_color` | the image editor's colour control (the last colour and eight recent ones are remembered) |
  | `img2img_inpaint_mask_brush_color`, `img2img_inpaint_mask_scribble_alpha`, `img2img_inpaint_mask_high_contrast` | the mask overlay colour (six presets, white and black for contrast) and opacity in the editor's Layers popover, remembered |
  | `keep_alive` | nothing to keep alive: the SPA does not depend on animation frames to go on. Generate forever and Generate several times queue the next run on WebSocket job events (which a background tab still receives), so they keep going with the tab in the background; jobs already queued run on the server even with the tab closed, and a reload re-attaches them and resumes the loop (see "Run controller"). Closing the tab ends Generate forever after its current run, as in A1111 (to keep generating without a tab, queue the jobs: "Add to queue", or the API's `forever` flag, which the Queue page can stop) |
  | `live_preview_refresh_period` | previews are pushed over the WebSocket at `live_preview_fps` |
  | `no_spellcheck`, `undo_redo`, `prompt_debounce` | the prompt editor never spellchecks or autocorrects, always has undo / redo (Ctrl/Cmd+Z, Shift+Ctrl/Cmd+Z), and is never delayed (extensions no longer read the DOM) |
  | `remove_image_on_hover` | dropping an image on a filled image field always replaces it |
  | `save_selected_only` | "Save" saves the image on screen; "Save this generation as zip" saves all of it |
  | `send_image_info_not_ui` | "Send to" always sends the parameters recorded in the image (e.g. the expanded wildcards) |
  | `open_dir_button_choice` | "Show in folder" opens the folder of the image itself (local browsers only) |
  | `cross_attention_optimization` (a read-only "Automatic") | the attention backend is chosen by launch flags |

- **Quick settings** (`features/quicksettings`): the header's ⚡ button lists
  `quicksettings_list` minus the model bar's keys (checkpoint, VAE / TE,
  dtype, preset), rendered by the schema renderer in a lazy popover and
  written as they change (continuous kinds debounced; a refusal reverts with
  the message). Pins are set from the settings rows (optimistic
  `PATCH quicksettings_list`).
- **Extensions** (`pages/extensions`, `#/extensions?tab=installed|available|url|backup|diagnostics`,
  plus the filters of Installed, `q`, of Available, `q sort recent tags
  tagmode match`, and of Diagnostics, `q filter`):
  installed list (user and built-in apart; switches and update marks are
  staged, then "Save" or "Apply and reload UI"; per-extension or bulk update
  checks with progress, one request each, four at a time, stoppable;
  uninstall with confirmation, never for built-in ones; "Disable all
  extensions"), the extension index (server-fetched; the URL loaded last
  stays loaded for the browser tab, `sessionStorage["remi-extension-index"]`;
  virtualized, search,
  tag hide / only-show with any / all matching, eight sorts, install), install
  from URL (validated URL, branch, folder), and config-state backups (list,
  save, detail against the current commits, restore the extensions, delete;
  both confirmed). The WebUI checkout is never restored from here (a backup
  made before the rewrite points at an upstream commit; `git reset --hard`
  would drop the working tree). Without extension access every write is
  disabled with an explanation. Server errors are shown translated: their
  `template` (English, `{{name}}` placeholders) is a catalog key
  (`adminErrorMessage`). The Available list is virtualized inside the page's
  own scroll (`data-scroll-root`, `scrollMargin`), so there is one
  scrollbar; date sorts put undated entries last.
- **Server control** (`features/server-control`, mounted by the shell):
  "Reload UI" / "Restart server" from the settings page, the palette or
  after applying extension changes; a confirm dialog (warns about unsaved
  edits and a running job), then an overlay that follows the server through
  `/internal/ping` and reloads the page once a ping answers with another
  `instance` than before the command (a server without instance ids: after
  a missed ping); the old server answering again after a missed ping puts
  the overlay back to "stopping". Timeout, "Keep waiting" and "Reload page
  now". `unsaved.ts` is the shared unsaved-edits registry behind the unload
  guard; it is bypassed only right before the page reloads (a dismissed or
  failed command keeps the close-tab prompt).
- Menu and popover triggers with tooltips use `usePopupTooltip`
  (`components/popup-tooltip.ts`): focus returning to the trigger when the
  popup closes no longer reopens the tooltip (the Appearance menu bug).

## Extras, PNG Info, Gallery, Queue

- **Extras** (`pages/extras`, `features/extras`, `features/compare`): the
  `extras` schema rendered as is (source tabs Single Image / Batch Process /
  Batch from Directory / Single Video bound to `mode`, then every
  postprocessing operation's section in its slot: upscale with two upscalers
  and the second's visibility, face restoration, …). Process (Ctrl/Cmd+Enter,
  Ctrl/Cmd+Shift+Enter adds to the queue, Esc interrupts) submits an `extras` job tracked in the progress store
  (`run.ts`; jobs in flight survive a reload, this session's results are kept
  in sessionStorage with the source image of each run). A single-image result
  shows in `CompareSlider` against its source (drag anywhere, or the handle
  with ←/→, Shift for 10 %, Home / End; both images fitted to the frame so an
  upscale compares feature for feature) or alone; batches get a thumbnail
  strip. Result actions (download, save, send to img2img / inpaint / Extras,
  copy parameters), the viewer, "Add to queue". While the schema loads, the
  opening of a form stands in (the source image's label over its empty drop
  zone, then two fields), made of the same sizes; no fixed height fits every
  form, so on a phone the result area under the form comes in with the form.
- **PNG Info** (`pages/png-info`, `features/pnginfo`,
  `#/png-info?file=<id>`): drop, paste (anywhere on the page) or choose an
  image; it is uploaded once and read by `POST /pnginfo`. The infotext panel,
  "send parameters" to txt2img / img2img (page switch), copy, "send image" to
  img2img / inpaint / Extras, and three tabs (a select when they do not
  fit; another image opens on the first): parsed parameters, every
  metadata entry grouped (image, text chunks, EXIF IFDs; binary values as
  sizes) and the PNG chunk list. Values are rendered as text only.
- **Gallery** (`pages/gallery`, `features/library`): `GET /gallery` pages
  (120 items) per filter set; filters and the view persist
  (`remi-gallery-view`, the search text is per visit): search (debounced, all
  words in the infotext, name or model), form / folder (txt2img, img2img,
  Extras, grids, saved, other, with counts), model (searchable, counts from
  `/gallery/facets`), date range (local days, presets), starred, sort
  (newest / oldest / name / largest), layout (justified rows or masonry
  columns at each image's exact aspect ratio, `layout.ts`, pure) and tile
  size. TanStack Virtual renders only the rows / tiles near the viewport
  (rows for the justified layout, lanes for masonry) and fetches the next
  page near the end; thumbnails are `/thumbs` at the tile's device-pixel
  width. Keyboard: the grid is one tab stop (roving focus), arrows move
  (↑ / ↓ to the image above / below), Home / End, Enter opens, Space selects
  (Shift extends), F stars; Ctrl/Cmd+A selects everything loaded, Delete
  removes the selection, Esc clears it. Selection (`selection.ts`, pure:
  toggle, Shift ranges from the anchor, pruned when items leave the list)
  shows a floating bar: download as zip (`/gallery/zip`), star / unstar
  (optimistic), compare (exactly two: `CompareDialog`, the slider, and the
  parameter table, whose values wrap after a name's separators `.` `_` `/`
  and inside a part only when it alone is wider than the column), remove
  (confirm; "also delete the files" is opt-in). The viewer is the shared
  lightbox with an `aside`: file facts, star, PNG Info, show in folder
  (`/gallery/{id}/reveal`, refused for remote browsers), delete, and the
  infotext panel; its toolbar has the result actions ("use parameters",
  send to, download). `outputs.changed` and finished jobs refresh it;
  "Rescan" indexes the output folders.
- **Queue** (`pages/queue`, `features/queue`, `#/queue?view=pipelines|recipes`,
  plus the history's `status form starred q` and the recipes' `recipes
  family`): two columns on wide screens
  (queue | history), tabs below 64 rem (`section=history` keeps the History
  tab). The running card (live preview frames
  of the running job, step / batch / ETA, skip, interrupt), "Up next" in run
  order from `GET /queue/items` (`queue.changed` keeps it current) and the
  history. Queued jobs: drag the grip (pointer) or move it from the keyboard
  (Space picks up, ↑ / ↓, Space drops, Esc cancels; announced), move to top /
  bottom, priority (high / normal / low), duplicate, cancel, edit. A move
  across priorities takes the neighbours' priority (`reorder.ts`, pure: the
  list stays in run order) and is shown at once. **Edit** (`edit-session.ts`)
  loads the job's named values into its form (`deliver`, the page switches)
  and, for a queued job, opens an edit session: a banner over the run
  controls offers "Update job" (`PUT /jobs/{id}/params`, validation errors
  land on the fields) or stop editing; a job that started meanwhile leaves
  the values in the form. The session is kept in `sessionStorage`
  (`remi-queue-edit`), as the form's draft is: a reload keeps the banner, so
  Generate does not queue the edited job a second time. History: paged (50) and virtualized, search
  (debounced), status (all / completed / failed / stopped), form and starred
  filters; each row shows status, duration, up to four output thumbnails
  (open the viewer), star, queue again (`retry`), open in its form, remove;
  imported jobs show their notes, and jobs that are not replayable say so
  and cannot be queued again as they are. Header: pause / resume and the
  settings menu: import from Agent Scheduler / Prompt Queue (dialog: the
  sources found, one click each, then the report with counts and every task
  that was not imported exactly, with its notes), stop every generate
  forever, stop and cancel everything, clear the history (keeping starred
  jobs by default). **"Add to queue"** (`FormQueueControls.tsx`) sits next
  to Generate / Process in every form: it posts the form as a job without
  making it the form's run (a toast links to the Queue; Ctrl/Cmd+Shift+Enter
  on the Generate and Extras pages, the Agent Scheduler's "Enqueue"
  shortcut with a fixed key). A press within 400 ms (`DOUBLE_PRESS_MS`) of
  one still waiting for its answer is a double click, as for Generate: it
  joins that press (one request, one `Idempotency-Key`, one job, one toast);
  Extras' Process sends nothing while its submit is pending. Generate forever and
  repeat N stay the run controller's (client-driven, see "Generate page").
  The Agent Scheduler's queue settings live in Settings → System → Queue
  under their old keys (history retention, queueing failed jobs again, the
  action when the queue is done; server-side, `remi/jobs/queue.py`); its
  layout settings are not carried over (see
  `extensions-builtin/agent-scheduler/README.md`).

## Prompt tools (`features/prompt-tools`)

Tag autocompletion and the former Prompt All-in-One, on `/api/v2/prompt-tools`
(server: `remi/prompt_tools`, `remi/server/routes/prompt_tools.py`; data and
settings come from the built-in `sd-webui-tagcomplete-neo` and
`sd-webui-prompt-all-in-one-neo` extensions, disabling one turns its part off).
`register.ts` (imported by `main.tsx`, small) registers the completion source
and the prompt decorator and refreshes on `options.changed`, `model.loaded`,
`networks.changed` and `job.queued`; everything else is lazy.
`GET /prompt-tools/config` says what is on (`tags.*` mirrors the `tac_*`
settings, `tools.*` the `prompt_tools_*` ones); without it (mock mode, an
older server) nothing is shown.

- **Completion** (`completion/`, lazy chunk on the first keystroke):
  `context.ts` finds the target at the cursor (a tag back to the last
  unescaped delimiter; `<lora:` / `<l:` → LoRA, `<e:` → embedding, `<c:` /
  `<chant:` → chant, bare `<` → all three; `__name` → wildcard names,
  `__name__word` → that wildcard's values; nothing in weights, comments —
  every `#` / `//` to the end of the line and `/* … */`, as Forge strips
  them — or `BREAK`/`AND`; a typed `@artist` prefix is not searched but
  kept for an artist tag). Where it runs: `tags.enabled` (also off for the
  checkpoints of `tac_activeIn_modelList`), `tags.negative`, and per editor
  the `promptScope` facet (`features/prompt-editor/facets.ts`: the form, and
  whether it is an extension's prompt field — `Text` with `prompt: true` — or
  a main prompt — `PromptField`; the style editor has no form) against
  `tags.txt2img` / `tags.img2img` / `tags.third_party`. The server ranks
  (match tier, then posts, then the user's usage; tags found by an alias
  only with `tac_frequencyIncludeAlias`); the client keeps that order
  (`filter: false`), caches queries (LRU, 60 s; cleared on settings / network
  changes) and waits `tags.delay` (`tac_delayTime`, 40 ms) before an
  uncached request. `insert.ts` applies the insertion settings
  (underscores and their exclusions, escaped brackets, a comma and a space after a tag but
  not before `,` `)` `]` `}` `>` `|` `:`, `<lora:alias:weight>`, activation
  text at the start / end / before / after, Anima's `@` artist prefix, Tab
  completing nested wildcards one folder or to the first difference). An
  accepted item is counted (`POST /usage`, per positive / negative prompt).
  Options show the category colour, alias (`← alias`; with
  `tags.only_alias` the alias alone is the label), translation, kind and
  post count; the info panel has the LoRA preview, description,
  activation text and wiki link.
- **Decorator** (`PromptToolsDecorator.tsx`, in the shell): the editor stays
  mounted and is hidden in the tag view; the tools bar (`PromptToolbar`,
  lazy) offers text ↔ tags, quick tags, history, favorites, translation
  (only with a configured provider; a network provider's API key is entered
  in its menu, `PUT /translation-key`, stored apart from the settings and
  never sent back) and, when the provider is the OpenAI-compatible one
  (`tools.generate`), "Write a prompt" (`panels/GeneratePanel.tsx`: a
  description in any language, optionally changing the current prompt, →
  `POST /generate` with `prompt_tools_generate_instructions` or the built-in
  instructions; the answer is editable, then replaces or extends the
  prompt; closing the panel unmounts it and aborts a request still running,
  as closing the upsampler does, so a late answer lands nowhere: the
  server's call to the provider is not cancelled, it ends at its own
  timeouts). `prompt_tools_toolbar` limits the bar to the main
  prompts (`PromptDecoratorProps.main`, set by `PromptField`). The view per
  prompt is persisted (`remi-prompt-tools`). The history's times ("3
  minutes ago") move on while the panel is open: a relative time is counted
  from `useNow()` (`lib/use-now.ts`, one shared 30 s timer while something
  reads it), passed to `timeAgo`, never from a `Date.now()` the render does
  not read (the React Compiler would keep the first text).
- **Tag chips** (`chips/`): `parse.ts` turns a prompt into chips and
  separators so `serialize(parse(text)) === text`; weights are effective
  (`((x))` 1.21, `[x]` 0.91, `(x:1.3)`, a network's first number); an edit
  rewrites only its chip (`(x:1.3)` canonical form). A disabled chip is a
  prompt comment `/*x, */` that takes one separator along (offered only with
  `enable_prompt_comments`). `ChipsView`: roving focus (one tab stop),
  ←/→ focus, Alt+←/→ move, Delete removes, Ctrl/Cmd+Z undoes in the hidden
  editor, pointer drag to reorder (touch: horizontal), a popover per chip
  (text, weight slider / number / ±0.1 / reset, enable / disable, favorite,
  delete; a weight is written once, on release / blur), an input with tag
  suggestions (`ui/combobox`; a picked tag is written and counted like a
  completion); moves and removals are announced in a live region.
- **Panels** (lazy, in a popover): quick tag groups (categories, search;
  a tag toggles in the prompt, written like a completion: underscores,
  escaped brackets), history (search, use, append, name, star, delete,
  clear; positive or negative prompts, one item per distinct text: naming
  and deleting cover the whole group), favorites (the prompt's own kind and
  tag snippets, paged by the server; collections, search, save the prompt,
  use / append, rename, move, reorder, delete, JSON export / import of
  everything). Translation: "write the prompt in English" (the non-English
  tags, all replaced on one parse, a translated list becomes several tags,
  weights kept, undoable) and "show the tags in …" (tag view, network
  providers); long lists are sent in chunks.
- **What did not come over.** Tag Autocomplete settings without an
  equivalent are listed with their reasons in the "Not carried over" table of
  `extensions-builtin/sd-webui-tagcomplete-neo/README.md` (a test keeps it in
  step with the old 58 `tac_*` keys); Prompt All-in-One features are mapped
  one by one (kept, merged into a core feature, or dropped and why) in
  `extensions-builtin/sd-webui-prompt-all-in-one-neo/README.md`. Their old
  HTTP APIs (`/tacapi/v1/*`, `/physton_prompt/*`) answer `410 Gone`
  (architecture.md).

## i18n

- Keys are the **English text** (`keySeparator`/`nsSeparator` off).
  Namespaces per locale: `chrome` (bundled `src/locales/<locale>.json`, one
  lazy chunk each; `en.json` holds only English plural forms and ships with
  the i18n module) and `server` (`GET /api/v2/i18n/{locale}`, e.g. `zh_CN`,
  for schema labels, choices, info texts and extension strings). `t()` looks
  in `chrome`, then `server`, then returns the key.
- Every UI chrome string goes through `t()` with a **literal English key**:
  `t("Reset to default")`, `t("Step {{step}} of {{steps}}", { step, steps })`.
  Keep constants that are translated later as `label:` / `group:` /
  `title=` / `description=` literals (routes, commands, placeholders). No
  string concatenation: a label and its value are joined by
  `t("{{label}}: ", { label: t("Errors") })` (Chinese `{{label}}：`), never
  `{t("Errors")}: ` with a colon of the source's own (`locales.test.ts`).
  A sentence that names keys takes them from
  `formatShortcut` through a placeholder (`Undo with {{keys}} in the
  editor.`): `locales.test.ts` fails on a chrome string that spells a
  modifier (`Ctrl+`, `Cmd+`, `Alt+`, `⌘`, `⌥`). Prefer sentences that work
  without plural forms (`Saved: {{count}}`). A count followed by a noun
  (`{{count}} steps`) gets its English singular in `en.json` as
  `"{{count}} steps_one": "{{count}} step"`;
  the key itself is the plural (en.json holds only `_one` forms); `zh-CN.json`
  needs no plural forms. `locales.test.ts` fails on a `{{count}} <plural noun>`
  key without its `_one` form and on any other form in en.json.
- Schema strings are passed to `t()` as-is (`t(node.label)`); `markdown` and
  `html` node text and a setting's HTML comment are looked up whole (key = the
  exact source text; the catalogs translate them keeping the Markdown or
  markup), then rendered like the original (Markdown as React elements, HTML
  sanitized). `tests/remi_core_strings.py` collects them, so an untranslated
  settings help block fails `tests/test_remi_core_zh_cn.py`.
- **Failure text**: a failure is shown in the UI language. The server's
  error details are English (for logs and API clients), so the server's
  `detail` (an `ApiError`'s message) is never the whole message:
  `features/errors/error-text.ts` `errorText` says the sentence a refused
  request's body carries (`template` and `params`: a queue, pipeline,
  recipe, admin or other refused operation; it says more than any reading of the status)
  or else a sentence chosen by status (callers pass their own for a status
  that means something specific to them, such as a name already taken; a
  422 lists the refused values' messages, which are catalog keys). A
  caller's own 404 sentence comes before the body's: the server names what
  is gone by its id ("no recipe 'r1'"), the caller in the user's words
  ("The recipe was deleted meanwhile."), so a call that can meet a 404 of
  its own passes one;
  `failureText` also keeps the app's own errors (already translated
  sentences), `serviceFailureText` is for the prompt tools (a 502 says its
  body's sentence, the service's own words in it, after a translated one: it
  is the only clue to what to fix; a 409 or 413 says its body's sentence, else
  its own), and
  `renderFailureText` is for an error boundary (a request by its status, a
  render bug by its own message). A request that got no answer (`fetch`
  failed, a body that is not JSON, a lazy chunk whose import failed: it could
  not be fetched, and a chunk that throws as it runs reads the same) is
  marked where it happens as a `RequestError` (`api/http.ts` `request` /
  `unwrap`, `lib/retryable-lazy.ts`), so a boundary and `failureText` can
  tell it from a bug (any other `TypeError` keeps its own message); the
  caller's own abort is left as it is. Admin errors (extensions,
  Settings > Server) are said by their `template` and `params`, anything
  else by status (`adminErrorMessage`). `features/errors/failure-text.test.ts`
  fails on any read of an error's message, detail or text outside
  `error-text.ts` and `api/` that is not listed there with its reason. A text
  the server stored as it is (how a job or a model load ended: `job.error`,
  the `job.failed` and `model.loaded` toasts) goes through `serverText`: a
  fixed sentence (`remi/jobs/texts.py`, a key of the server catalogs) is
  translated by its exact key; a job's own exception stays as it is. The
  values a job refused when it ran (`job.errors`, `job.failed`'s `errors`:
  each names its `owner` and `param` like a refused submission's) are said
  by `jobErrorText`, each named by its field's label in the job's form
  (`schemaIndex` of the form's schema; the Queue page fetches it only for
  such a job) as `toastInvalid` names a refused submission's values. The
  model bar names the server's stage (`model.loading`, only ever "loading")
  with its own key.
- **Server sentences**: what the server says comes in one of three forms.
  A fixed sentence is a catalog key, looked up by its exact text
  (`serverText`, `messageText` for a string). A sentence with values in it
  carries its template: a message object `{text, params}` (toasts, text
  outputs), a note `{message, values}` (a job's notes), an admin error's
  `template`, or a `remi.ui.Sentence` (`template` and `params` beside the
  English: model advice, why a script does not apply, a rejected value's
  message), shown with `sentenceText` (`ui/messages.ts`): the template
  translated, a list param's catalog names each translated and joined the
  language's way (`Intl.ListFormat`), other values as the server wrote
  them. Text the server only passes on stays as it is in every language: a
  job's own exception, extension diagnostics (but a manifest error that comes
  with its template, such as an extension that needs a newer Remi, is
  translated: `manifest_error_details`), a worker's state, what an
  extension's own `validate` or paste function raised, git output. A field's
  validation messages are stored as the server sent them
  (`FormState.errors`, a list per param, never empty) and translated where
  they are shown (`fieldErrorText`, several joined by " · "), so a language
  switch re-reads them. A refusal the browser itself makes follows the same
  form: the JSON Builder's `DraftError` keeps its template and values
  beside the English (a path such as `items[2].bbox` is a value), and the
  page translates it where it is shown (`draft-text.ts`).
- **Text made outside `t` takes the language**: the React Compiler memoizes
  a render on the values the render itself reads, and a helper that reads the
  global `i18n` inside does not count. Text a render makes with such a helper
  (`messageText`, `sentenceText`, `validationMessage`, `fieldErrorText`) is
  given the render's language, `useTranslation().i18n.language`, or it stays
  in the old language after a switch. A store selector returns what the
  server sent (a param's error entries) and the render translates it, never
  the selector (a selector re-runs only when the store changes). A text the
  server stored (`serverText`), why a job failed (`jobErrorText`) and the
  texts for a failed request (`errorText`, `failureText`,
  `serviceFailureText`, `renderFailureText`) take the language too; the
  error boundary's wrapper passes it to the class that renders the fallback.
  A component keeps the failure itself in its state, not its text (a
  dialog's error line holds `{ error }`, the pose editor its problem), and
  makes the text at render, so a switch says it again.
- `src/locales/zh-CN.json` must contain every chrome key:
  `locales/locales.test.ts` scans the sources and fails on a missing key or a
  mismatched `{{placeholder}}`.
- Locale ids: BCP 47 in the app (`zh-CN`, `<html lang>`), underscore form for
  the server (`zh_CN`). The user's choice persists (`remi-locale`); without
  one, the server default (`Bootstrap.locale`) then the browser language is
  used. Switching does not reload the page. A language whose chrome
  catalog (a chunk) could not be fetched is not switched to; it is asked
  for again when the browser is back online or a failed load is retried
  ("Try again"), unless another language was chosen meanwhile.
- **Writing direction**: `activateLocale` sets `<html dir>` from the
  locale (`textDirection`: CLDR through `Intl.Locale` text info, with a list
  of right-to-left languages as a fallback), so an Arabic, Hebrew, Persian or
  Urdu catalog (`localizations/ar_SA.json`, …) lays the app out right to
  left; `App` passes it to Base UI's `DirectionProvider` (sliders, menus,
  keyboard arrows). This replaces the Gradio UI's `"rtl": true` key in a
  localization file (the server's catalog only carries strings): a catalog
  is named after its language, which already determines the direction.
  Numbers in inputs and key combinations (`<kbd>`) stay left to right in a
  right-to-left layout (`index.css`), and the prompt editor uses `dir=auto`.
- The language menu names a catalog by the server's name for it, else by its
  own name from CLDR (`localeName`: a user-supplied `ar.json` reads
  "العربية", not "ar").

## Mock mode

`pnpm dev:mock` (`--mode mock`, or `VITE_MOCK=1` in the environment):
`api/mock/server.ts` answers every `/api/v2` call from
`api/mock/fixtures.ts`; `api/mock/socket.ts` replaces the
WebSocket and the simulated worker emits `job.*` events and real WebP
preview frames; a dev-server plugin serves generated SVG artwork for
`/api/v2/files|thumbs/mock-*` and an empty `user.css` (the bootstrap fixture
lists one; end-to-end tests route their own). The txt2img fixture mirrors the shape of the
real schema — `sampler`, `seed` and `refiner` are scripts in the `sampler`,
`seed` and `accordions` slots, Hires. fix is a core section whose enable
toggle is exclusive, has `on_change: hr_resolution` and an output line, and
`conflict_params` lists the lockable params — and covers every node kind and
condition operator, cross-owner conditions, an unknown slot, a selectable
script, a script with an untitled group of JSON params, a
bound tabs node, an Anima-like section (enable toggle, preset action,
exclusive locks, custom widget) and a ControlNet-like section (uploads,
`on_change`, sanitised HTML). The mock `__conflicts__` reads the `scripts`
context (Anima locks one-of-two, Hires. fix vs refiner, LCM pins its
schedule). `POST /jobs` validates like §3 and answers 422 `{"errors": …}`,
each refused value in the server's words (`remi/ui/nodes.py`: the sentence,
its template and values), and the sample extensions' toasts name their
values as `{text, params}`, so the zh preview translates both through the
catalog sample (`serverCatalogZh`; `api/mock/server.test.ts`)
(try `#invalid` in the prompt, or in a text setting, which `PATCH /options`
refuses the same way; the schema's `core.required` rules are
checked by the client before it sends anything, so img2img without a
source image marks the field and shows the toast without a request);
`GET /jobs` is paged (`cursor`, `limit`), `repeat` queues copies and retry
works; interrupted jobs end as `job.completed` with status `interrupted`.
`/prompt/tokens`, `/styles` (+ `/styles/apply`), `/infotext/parse`,
`/infotext/image` and the `__forge_preset__` action answer too; as on the
server, a model change (`PATCH /options`, a preset) only selects the model and
the next job loads it, sending `model.loading` (stage "loading") at its start
and then `model.loaded` (`api/mock/model.ts`; `/models/reload` loads it at
once). There is a
schema-rendered extension page (`Toolbox`) and a zh_CN server catalog
sample. `api/mock/library.ts` serves the Gallery (260 outputs of varied
sizes, forms, models and dates; stars, removal, zip, reveal), PNG Info and
the Queue tools (duplicate, edit, star, clear, an import report); the
simulated worker honours pause / resume. Production builds contain no mock
code.

`src/api/mock/contract/{txt2img,img2img,extras}.json` are the real
`tab_schema()` outputs (`tools/dump_schemas.py`, default options and the
built-in processing scripts). Tests only; never bundled.

## Tests

`tools/check.sh` runs every gate (Python and SPA) in one command; see
[testing.md](testing.md) for the steps, the end-to-end and axe suites, the
real-server smoke test and CI.

`pnpm test` runs Vitest in jsdom: condition evaluator (incl. Python
semantics), form store (drafts and their coercion, reset, patches,
validation errors and reveal), action requests and response splitting,
change effects (conflict flow with the scripts context, `on_change`
debounce, bulk resets, `__settings__`, stale-response drop), HTTP error
mapping (422), WS event validation, frame parser, WS store (handshake,
dispatch, preview URL revocation, reconnect, overlapping connects),
ws-bridge (schema refresh, preview clearing, per-form results, model
loading), router (incl. malformed hashes) and nav, Markdown/HTML
sanitising, attention/reorder editing and the CodeMirror key bindings,
renderer smoke tests for every node kind, tabs visibility/enablement,
upload preview revocation, `jobRequest`, the generate page (submit, 422
marking, Ctrl+Enter, interrupt), locale switching, zh-CN catalog coverage,
and **contract tests** rendering the real txt2img / img2img / extras
schemas (no console errors, every default round-trips, conflict params and
slots resolve). Generate view: prompt tokenizer and bracket problems,
editor commands on a real `EditorView` (weights with precisions, tag moves,
undo, lint, completion registry), the run machine (every transition),
restore after reload (incl. events during the lookup and submits during
the restore), resync after a reconnect / missed events, double submits with
interrupt, infotext paste planning (seed / size filters, one
`on_change` per action, one conflicts pass), send-to mapping and delivery,
image metadata reading (PNG chunks incl. zlib, JPEG / WebP EXIF), the
curated arrangement of the real schemas (and the user's order), size
helpers, lightbox zoom / pinch math, keyboard and pinch on the component,
gamepad mapping, history grid keys and `aria-activedescendant`, model bar
modules commit (incl. the picker nested in the narrow header's popover),
token counter debounce and recount after a model load, notifications,
reset confirmation, queue position,
form reload carry-over, and the page (submit, 422 marking, keys, history).
Library pages: justified / masonry layout and the above / below neighbour,
selection ranges and pruning, gallery filters → query (local day bounds),
the gallery page (rows at aspect ratio, check boxes and bulk star / delete,
Shift-click ranges, Escape, the roving keyboard focus, filters sent), the
compare slider (keyboard, clamping), Extras runs (single-source tracking,
finish / restore after a reload) and page, PNG Info (upload, tables, send),
queue reorder planning across priorities, the pending list (keyboard move
and its requests), edit sessions (update, 422 marking, a job that started)
and "Add to queue".
Image editor: undo history (order, ≥ 32 states, byte budget, per-layer
removal), patches round-trip, mask export (white + exact alpha whatever the
tint, luminance fallback), flood fill, dab placement (batch-independent,
pressure), view maths, overlays against the server's crop maths, shortcuts,
the document on memory rasters (strokes, opacity, erase, invert, clear, fill,
external loads, sampling, per-layer undo / redo, tracked bounds, upload sync
with hash reuse, its expiry and flush), the undo budget, input (a stroke
committed on unmount, one-finger touch left to the page), toolbar keys and
`aria-keyshortcuts`, the editor's value binding (`Editor.test.tsx`: uploaded
mask loading, send-to, a failed load with Retry, layer commits), schema
layers through the renderer, the real img2img schema → mode layers, the
Source panel (balanced mode grid, no nested frames, mode options, errors,
the "Resize by" size), source / size writes (undo toast, flush first),
stage view switches (first job of a run only, never mid-stroke) and the
submit waiting for layer uploads; pytest
(`tests/test_remi_canvas_layers.py`) for the server's layer composition and
the Source area's zh-CN catalog.
`tests/e2e` (Playwright, mock mode): generate + history, Escape interrupt,
editor keys and bracket warning, mode switch, viewer, send to img2img, the
phone layout, the image editor (`canvas.spec.ts`: paint / undo / redo a
mask kept across the stage views with the job carrying it, sketch with Shift
erasing, zoom / fit / full screen / shortcuts, an Inpaint-upload mask on the
canvas across mode switches with per-layer undo and the job's layer ids,
drop and paste, image size and removal, touch scrolling on phones), and the library (`library.spec.ts`): gallery search and viewer
details, two-image compare, PNG Info → txt2img, and the queue (pause, "Add
to queue" twice, keyboard reorder, resume, history). TypeScript runs with `noUncheckedIndexedAccess`.

## Performance

Budget (scorecard dimension 5, `webui/tools/perf/budget.json`, checked by
`tools/check.sh` step `perf:bundle`; measurements and method in
[docs/research/ui-performance.md](../research/ui-performance.md#1-bundle)):
initial JS ≤ 300 KB gzip; every route and every other lazy entry ≤ 150 KB
on top of it, counted with all its static imports; the editor chunks
(canvas editor, CodeMirror prompt editor) exempt; no exception: the default
Generate route together with its parameter form (a lazy part it fetches as
it renders, `defaultRouteParts` in `webui/tools/perf/budget.json`) is held
to the same 150 KB. The build groups everything the
first paint needs into `react`, `vendor` (other node_modules) and `shell`
chunks (`codeSplitting` groups with the `$initial` tag in
`vite.config.ts`); everything else is lazy: pages, the prompt editor, the
command palette, the model picker, widgets, locale catalogs.

Keeping a first view small:

- Menus and popovers: `@/components/ui/dropdown-menu` and
  `@/components/ui/popover` render their trigger themselves and load the
  Base UI menu / popover (`dropdown-menu-impl.tsx`, `popover-impl.tsx`) on
  hover, focus or open. Code that builds on `Menu.Root` directly imports its
  parts from `dropdown-menu-impl`.
- Selects: `@/components/ui/select`, `@/components/SearchSelect` and the
  chips field of a multiple choice (`ui/inputs/MultiChoice.tsx`) render a
  stand-in of their closed state (same markup and roles) until used; hover
  or focus fetches the Base UI code (`select-impl.tsx`,
  `SearchSelectImpl.tsx`, `ui/inputs/ChipsSelect.tsx`, through
  `lib/lazy-module.ts`), a press or an opening key puts the real control in
  place opened, keyboard focus puts it there focused (typed text and the
  caret carry over); a stand-in is never replaced under a pressing pointer.
- The history grid (`features/gallery/HistoryGrid.tsx`, TanStack Virtual)
  loads with the session's first result; an empty session shows
  `HistoryEmpty`.
- Form views most first views do not show (image, file, color, JSON, tabs,
  outputs) are deferred in `ui/lazy-views.tsx`; a test that renders them
  synchronously calls `preloadViews()` first. A view's placeholder is as
  tall as the view it stands for in its usual first state, so what follows
  does not move when the code arrives: an output's is its label over the
  "No output yet" line, each a line of the same text size (nothing for an
  output without a label); an image upload field's is its label line over the
  empty drop zone (`ImageFieldSkeleton`), and a layer painted on an editor
  (which renders nothing there) has none. Tabs stand in as a strip-high bar
  over the selected tab's content, rendered as it will be
  (`layout/tab-selection.ts` picks the tab for both).
- Dialogs and surfaces that open on demand (img2img's source panel and
  stage, the networks item dialogs, the lightbox, the style editor) are
  `retryableLazy()` and mounted on first use.
- Every code-split component is `retryableLazy()` (`lib/retryable-lazy.ts`),
  never a plain `lazy()`: `lazy()` keeps a failed import for good, so a chunk
  that failed once (offline, a build replaced under the page) stayed failed
  until a reload. A failed load reaches the error boundary as usual; its
  "Try again", and the browser coming back online, load the chunk again.
  Each boundary also resets the suspense queries that failed under it
  (`QueryErrorResetBoundary`), so "Try again", and coming back to the page,
  ask for them again.
  `retryable-lazy.test.tsx` fails on a plain `lazy(` in `src`.

Fonts load per unicode range (Geist latin ≈ 28 KB woff2; a Chinese page loads a dozen Noto Sans SC slices of 20 to 70 KB, cached after the first visit).

## Backend contract

What the client relies on, and where the server side is specified (the
generated `src/api/gen/schema.d.ts` types every route):

- `GET /app` returns `Bootstrap` (generated from the server's model;
  architecture.md "HTTP API v2").
- Schema-rendered extension pages: `GET /api/v2/pages/{id}` →
  `{version, id, title, owner: "page:<id>", layout, defaults, on_load?,
  send_to?}` ([schema-format.md §8](schema-format.md#8-page-schema-get-apiv2pagesid)).
- Action bodies carry `scripts: {owner id: values}` (every non-core owner
  except the caller) next to `values` / `core`, and `__settings__` in a
  patch reports settings the server already stored, so the client only
  updates its cache ([schema-format.md §5](schema-format.md#5-actions)).
- WS job events carry `job_id` and `seq` (= binary `job_seq`); `job_no` is
  0-based; `tab` is optional; binary preview headers are little-endian. A
  model load ends with `model.loaded {model?, error?}`. See [api.md "WebSocket"](api.md#websocket-apiv2ws)
  and architecture.md "WebSocket `/api/v2/ws`".
- `GET /jobs?cursor=&limit=` returns `{items, next_cursor, total}` newest
  first ([api.md "Conventions"](api.md#conventions), architecture.md route
  table).
- Uploaded images are previewed from a local object URL; ids from elsewhere
  are shown via `/api/v2/files/{id}`, where upload ids resolve too.
- The checkpoint quick-bar writes `sd_model_checkpoint` with `PATCH /options`
  and shows `model.loading{stage}` while a model loads.
