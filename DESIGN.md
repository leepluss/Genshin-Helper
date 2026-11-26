---
name: Genshin Helper
description: Nocturnal 520x580 Celestial Observatory Planisphere command deck for passive HoYoLAB Interactive Map synchronization.
colors:
  observatory-ground: "#0B0F19"
  observatory-surface: "#121826"
  observatory-well: "#0E131F"
  observatory-hover: "#182134"
  horizon-hairline: "#232D42"
  horizon-border-strong: "#32405D"
  starlight-ink: "#ECEFF4"
  slate-ink-secondary: "#A3B1C6"
  muted-ink: "#8B9BB5"
  gilt-brass: "#D9A441"
  gilt-brass-soft: "#F3D28B"
  celestial-sky: "#38BDF8"
  celestial-sky-deep: "#0369A1"
  emerald-active: "#4ADE80"
  amber-standby: "#FBBF24"
  crimson-offline: "#F87171"
  badge-ice-cyan: "#7DD3FC"
  pill-sky-tint: "#BAE6FD"
typography:
  display:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif"
    fontSize: "14px"
    fontWeight: 700
    lineHeight: 1.25
    letterSpacing: "-0.02em"
  title:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: "0em"
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.35
    letterSpacing: "0em"
    fontFeature: "lnum, tnum"
  label:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif"
    fontSize: "11px"
    fontWeight: 700
    lineHeight: 1.25
    letterSpacing: "0.04em"
  mono:
    fontFamily: "Consolas, 'JetBrains Mono', monospace"
    fontSize: "11px"
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: "0em"
    fontFeature: "lnum, tnum"
rounded:
  xs: "4px"
  chip: "5px"
  sm: "6px"
  ctrl: "7px"
  md: "9px"
  pill: "999px"
spacing:
  xs: "4px"
  sm: "6px"
  md: "9px"
  lg: "11px"
  xl: "14px"
components:
  button-primary:
    backgroundColor: "{colors.celestial-sky-deep}"
    textColor: "#F8FAFC"
    typography: "{typography.title}"
    rounded: "{rounded.ctrl}"
    padding: "6px 10px"
  button-secondary:
    backgroundColor: "#182032"
    textColor: "{colors.starlight-ink}"
    typography: "{typography.title}"
    rounded: "{rounded.ctrl}"
    padding: "6px 10px"
  button-brass:
    backgroundColor: "rgba(217, 164, 65, 0.16)"
    textColor: "{colors.gilt-brass-soft}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "4px 11px"
  observatory-card:
    backgroundColor: "{colors.observatory-surface}"
    textColor: "{colors.starlight-ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "11px 11px"
  inset-well:
    backgroundColor: "{colors.observatory-well}"
    textColor: "{colors.slate-ink-secondary}"
    typography: "{typography.label}"
    rounded: "{rounded.ctrl}"
    padding: "7px 8px"
  tab-active:
    backgroundColor: "rgba(217, 164, 65, 0.18)"
    textColor: "{colors.gilt-brass-soft}"
    typography: "{typography.title}"
    rounded: "{rounded.sm}"
    padding: "5px 8px"
  sticky-footer:
    backgroundColor: "{colors.observatory-well}"
    textColor: "{colors.slate-ink-secondary}"
    typography: "{typography.label}"
    padding: "0 14px"
    height: "38px"
---

# Design System: Genshin Helper

## Overview

**Creative North Star: "Celestial Observatory Planisphere"**

Genshin Helper's Chrome Extension Control Center (`extension/popup.html`, `extension/popup.css`, `extension/popup.js`) is engineered as a nocturnal precision cartography instrument built for 2–3 second mid-game adjustments. Floating cleanly over dim-room gameplay and the HoYoLAB Interactive Map, the interface trades cramped vertical SaaS scrolling and decorative emoji clutter for a wide `520×580px` 2-column command deck framed by deep indigo slate panels, fine hairline instrument rulings, gilt-brass astrolabe accents, and celestial hydro-sky states.

Every surface earns its pixel budget through immediate scannability and single-click directness. Live engine state and Teyvat telemetry share one unified Horizon Strip above balanced functional columns, while connection health and the voluntary support trigger remain permanently anchored in a sticky bottom bar so scrolling never hides navigation or contribution links.

**Key Characteristics:**
- Fixed `520×580px` zero-scroll Control and Tools viewports using a 2-column grid instead of a narrow scrolling popup.
- Nocturnal observatory palette built around midnight indigo (`#0B0F19`), slate surfaces (`#121826`), and inset wells (`#0E131F`).
- Crisp `1.75px`–`1.9px` monochrome inline SVG iconography across every control, strictly zero decorative emojis in popup chrome or feedback toasts.
- Strict bilingual (`VI | EN`) parity with tabular lining numerals for live coordinates, audio percentages, and PIN codes.
- Anchored slide-up popovers (`#diag-drawer` Bug Report & `#donate-drawer` Support) above a persistent `38px` footer so feedback banners, privacy-safe bug reporting, and voluntary support QR drawers never cause layout shift.

## Colors

The palette balances low-glare dark observatory neutrals with cool celestial cyan for primary actions and warm gilt brass strictly reserved for instrumentation highlights and voluntary support.

### Primary
- **Celestial Sky** (`#38BDF8`): Active tracking switches (`Auto-Sync`, `Live Minimap 360°`, `Audio Chimes`), focus-visible accessibility rings, checked switch thumbs, and volume slider accents.
- **Deep Observatory Sky** (`#0369A1`): Primary CTA button gradient base (`Open HoYoLAB Map`, `Start LAN`) paired with `#F8FAFC` foreground text for accessible `AA`+ button contrast.
- **Ice Cyan & Sky Tint** (`#7DD3FC` / `#BAE6FD`): Monospace live telemetry coordinates, version pill text, LAN URL readouts, and secondary conflict suggestion chips.

### Secondary
- **Gilt Brass & Soft Gold** (`#D9A441` / `#F3D28B`): Active navigation tab frame, Horizon Strip perimeter hairline (`34%` alpha), brand compass emblem border, Mini-HUD & Compact UI child accents, utility preset buttons (`Drag`, `Test`, `Copy`), and the permanently anchored Support pill & drawer.

### Tertiary
- **Emerald Signal** (`#4ADE80`): Live `Active` engine status pill, `LAN Running` badge, `sync` primary hotkey dashed accent, and clean save/restore confirmation states.
- **Amber Standby** (`#FBBF24`): `Standby (0% CPU)` warning state on both header and footer status indicators plus non-blocking conflict cautions.
- **Crimson Offline** (`#F87171`): `Offline` disconnect indicators, active hotkey recording capture rings, UIPI admin elevation warnings, and destructive `Quit Desktop App` hover controls.

### Neutral
- **Midnight Observatory Ground** (`#0B0F19`): Base viewport backdrop topped with a subtle radial hydro-sky glow at the upper-right horizon.
- **Layered Observatory Surface** (`#121826`): Card containers, tab bar housing, language switcher track, and the slide-up Donate drawer surface.
- **Recessed Instrument Well** (`#0E131F`): Inset sub-groups (HUD positioning/capture child group, audio slider tray), hotkey recorder slots, neutral buttons, and the sticky footer bar.
- **Hover Slate & Hairline Borders** (`#182134` / `#232D42` / `#32405D`): Interactive row hover wash and `1px` crisp structural separators.
- **Starlight Ink Scale** (`#ECEFF4` / `#A3B1C6` / `#8B9BB5`): Primary titles, secondary control descriptions, and uppercase section labels maintaining `6.29:1`+ minimum text contrast over cards.

### Named Rules
**The Brass Reserve Rule.** Gilt brass (`#D9A441` / `#F3D28B`) marks active navigation, astrolabe framing, secondary quick-action chips, and the voluntary Support surface; reserve celestial cyan (`#38BDF8` / `#0369A1`) for primary navigation/tracking actions and live active switches.
**The Honest Semantic Rule.** Never paint decorative chrome emerald, amber, or crimson; those three hues strictly report live Engine power, UIPI risk, recording state, or validation outcome.

## Typography

**Display & UI Font:** System UI Geometric Sans (`-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif`) for instant zero-network MV3 startup and sharp Vietnamese diacritics.
**Telemetry & Keycap Font:** Tabular Monospace (`Consolas, 'JetBrains Mono', monospace`) for Teyvat coordinates, 6-digit LAN PINs, URLs, and keycap bindings.

**Character:** Compact utility typography locked to lining tabular numerals (`font-variant-numeric: lining-nums tabular-nums`) so numeric coordinates, volume readouts, and client counters never jitter horizontally during live polling.

### Hierarchy
- **Display** (`700`, `14px`, `1.25`, `-0.02em`): Extension brand wordmark (`Genshin Helper`) paired with `>= 1.27×` ratio over body copy to maintain clear header hierarchy inside a `520px` popup.
- **Title** (`600`–`700`, `12px`, `1.25`): Navigation tabs, card toggle titles, action buttons, and primary status titles.
- **Body** (`400`–`500`, `12px`, `1.35`): Tool card descriptions, support copy, and toast feedback messages.
- **Label** (`600`–`700`, `11px`, `1.25`, `+0.04em` uppercase on section headers): Dense utility captions, inset select controls, switches, chips, compact `3×2` hotkey titles, and the `38px` sticky footer.
- **Mono** (`600`–`700`, `11px`–`13px`, `1.25`): Coordinates (`sync-coords`), hotkey keycaps (`recorder-value`), LAN PIN (`13px`, `+0.12em`), and LAN URLs.

### Named Rules
**The Compact Floor Rule.** Never drop any interactive label, caption, badge, or select option below `11px` anywhere in the popup; reserve `12px`+ for primary interactive row titles and buttons.
**The Tabular Stability Rule.** Keep `lining-nums tabular-nums` enabled globally so 2-second status polling updates never shift timestamps, coordinates, or volume readouts sideways.

## Layout

The extension popup fixes its root canvas at `520×580px` (`overflow: hidden`) as a vertical flexbox shell divided into four bands:
1. **Sticky Header (`48px`):** Brand emblem (`28×28px` brass-framed icon), `14px` title + `v5.7.0` badge, inline `VI | EN` pill toggle, and live semantic status pill (`Active` / `Standby` / `Offline`).
2. **Navigation Tab Bar (`35px`):** Equal 3-column grid (`minmax(0, 1fr)`) with ARIA APG keyboard arrow navigation (`Control`, `Tools`, `Hotkeys`).
3. **Flexible Viewport (`flex: 1; min-height: 0; overflow-y: auto;`):**
   - **Tab 1 (`Control`):** Sets `min-height: 100%` so the Horizon Strip sits cleanly above a full-height 2-column Command Deck (`minmax(0, 1.08fr) minmax(0, 0.92fr)`, `9px` gap, `11px` internal card padding), filling flush down to `10px` above the sticky footer with zero scroll at rest.
   - **Tab 2 (`Tools`):** Balanced `2-column` grid (`TSP` | `Export`) above two full-width cards (`LAN Companion QR` and `Launch Borderless`), zero vertical scroll in both LAN OFF and expanded QR states.
   - **Tab 3 (`Hotkeys`):** `2-column` primary hotkeys (`Sync` full-span + aligned `Switch` & `Debug` half-cards) above a dense `3×2` in-game navigation hotkey matrix and bottom `Diagnostic` + `Quit` utility row, scrolling gracefully inside `.tab-viewport` with a `5px` slate thumb when conflict panels expand.
4. **Sticky Footer (`38px`):** Permanently anchored `#0E131F` strip showing connected `Map Tabs` status dot + count on the left and the gilt-brass `Support` trigger on the right.

### Named Rules
**The No-Scroll Deck Rule.** Tab 1 and Tab 2 must operate comfortably within `520×580px` with zero vertical scrollbars in both Vietnamese and English locales; only Tab 3's dense key-binding matrix and dynamic conflict advisors use the viewport scrollbar.
**The Anchored Layer Rule.** Feedback banners and the Donate QR drawer render `position: absolute` `6px` (`bottom: 44px`) above the sticky footer (`z-index: 30` and `40`), never pushing grid geometry or shifting underlying buttons.

## Elevation & Depth

Surface hierarchy relies primarily on **tonal step separation plus crisp `1px` structural borders** rather than blurry drop shadows, preserving instrument sharpness over high-contrast game windows. Only floating layers pinned above the sticky footer carry calibrated obsidian ambient shadows:

### Shadow Vocabulary
- **Floating Feedback Toast (`box-shadow: 0 8px 24px rgba(2, 6, 23, 0.72)`):** Separates transient `4s` status confirmations and errors above the live command deck without obscuring navigation tabs.
- **Anchored Support Popover (`box-shadow: 0 14px 36px rgba(2, 6, 23, 0.85)`):** Deep obsidian shadow paired with a gilt-brass border (`1px solid rgba(217, 164, 65, 0.55)`) separating the interactive Ko-fi / MoMo VietQR panel from the underlying cards.

### Named Rules
**The Hairline Depth Rule.** Rest cards stay flat on `#121826` over `#0B0F19` with `#232D42` `1px` borders, and nested controls step down into `#0E131F` inset wells; reserve real shadows exclusively for `z-index` floating overlays.

## Shapes

The system uses disciplined technical corner rounding scaled strictly by container hierarchy:
- **Cards & Drawers (`9px`):** Outer command cards, Horizon Strip, Tab Bar housing, and Donate Popover.
- **Interactive Actions & Wells (`7px`):** Primary/secondary buttons, inset sub-groups, feedback toast bars, brand emblem tile, and hotkey recorder cards.
- **Inner Tabs & Recorder Inputs (`6px`):** Active tab pill (`tab-btn`), dashed key-capture slots, conflict alert items, and drawer close buttons.
- **Micro Controls (`4px`–`5px`):** Inline selects (`5px`), preset utility chips (`5px`), version badges (`4px`), floor badges (`4px`), and recorder status badges (`4px`).
- **Switches & Status Pills (`999px`):** Language selector, live connection pill, pill switches, conflict suggestion pills, and footer Support trigger.

All SVG line icons follow a **unified `1.75px`–`1.9px` stroke grammar** (`stroke-linecap="round"`, `stroke-linejoin="round"`), matching UI crispness across every tab, toggle row, and button.

## Components

Every interactive primitive pairs clear rest contrast with accessible `:focus-visible` (`2px solid #38BDF8`, `offset: 2px`) rings and discrete hover feedback.

### Buttons
- **Shape:** `7px` rounded (`--radius-ctrl`), full-width or compact inline flex with centered `13px` stroke icon + `.btn-label` span.
- **Primary (`.btn-primary`):** Linear gradient `#0369A1 → #075985`, translucent `#38BDF8` border (`52%`), `#F8FAFC` bold `12px` text; lightens toward `#0284C7` on hover.
- **Secondary (`.btn-secondary`):** Deep slate `#182032`, `#32405D` border, `#ECEFF4` text; shifts to `#202B42` with sky border on hover.
- **Utility Chip / Support Pill (`.preset-btn--brass`, `.donate-trigger`):** Translucent brass wash (`12%`–`16%`), warm `#F3D28B` label, crisp border highlight on hover; `.preset-btn.is-active` inverts to solid `#D9A441` on `#0B0F19`.
- **Hover / Focus:** Active press scales gently (`scale(0.98)`); disabled states dim to `opacity: 0.52` (`cursor: not-allowed`).

### Cards / Containers
- **Corner Style:** `9px` outer cards wrapping `7px` inset child wells (`.sub-group`).
- **Background:** `#121826` primary cards (radial `#131A2A → #0E1422` gradient on the Horizon Strip) stepping down to `#0E131F` inside recessed sub-groups.
- **Shadow Strategy:** Flat hairline borders at rest; shadows reserved for floating layers above the footer.
- **Border:** `1px solid #232D42` default (`rgba(217, 164, 65, 0.34)` on Horizon Strip).
- **Internal Padding:** `11px` inside Control Deck columns (`8px 10px` on compact Tool cards, `7px 8px` inside sub-groups).

### Inputs / Fields
- **Style:** Native `<select>` (`#161E2F` fill, `1px solid #32405D`, `5px` radius, `11px` text) with single-line ellipsis truncation, plus `4px` range slider (`accent-color: #38BDF8`).
- **Focus:** Borders shift to `#38BDF8` on hover alongside global `2px` sky focus ring.
- **Keyboard Hotkey Recorder (`.recorder-box`):** `6px` `#0B0F19` slot with `1.5px dashed` semantic border (`emerald` for Sync, `brass` for Guide/HUD toggles, `sky` default); activates via mouse click OR `Enter`/`Space`, flips to solid `#F87171` while recording (`Esc` cancels), and truncates cleanly inside compact `3×2` cells.

### Navigation
- **Style:** Anchored 3-slot equal grid (`#tab-bar`) inside `#121826` with `#232D42` border; inactive tabs show muted `#A3B1C6` text and hover wash (`#182134`).
- **Active State:** Warm gilt-brass translucent wash (`rgba(217, 164, 65, 0.18)`), brass border (`rgba(217, 164, 65, 0.48)`), `#F3D28B` text, plus full ARIA `tablist` / `ArrowLeft` / `ArrowRight` / `Home` / `End` support.

### Celestial Horizon Strip & Pill Switches (Signature Components)
- **Celestial Horizon Strip:** Full-width split instrument pairing a clear 3-state segmented Engine Power control (`BẬT / Nghỉ 0% / Tắt hẳn` | `ON / Idle 0% / Quit`) and multi-line state readout on the left with right-aligned tabular coordinate telemetry and floor badge (`Surface` cyan vs `Underground` gilt-brass) across a vertical `1px` divider.
- **Sticky Utility Footer & Bug Report Popover:** The `38px` persistent footer pairs live tab telemetry on the left with a single one-click `Báo lỗi / Report Bug` (`#btn-diag-toggle`) trigger alongside `Ủng hộ / Support` (`#btn-donate-toggle`) on the right. Opening `#diag-drawer` provides privacy assurances, category chips, note input, optional masked HoYoLAB Map screenshot toggle, and 1-click `.zip` packaging that automatically copies the PII-free Discord `#support-and-bugs` template and opens Explorer highlighting the generated `.zip` (with manual `Reopen Folder` & `Copy Form Again` fallbacks).
- **Observatory Pill Switches:** `30×17px` (`26×15px` compact child variant) `#1E2738` tracks with `180ms` cubic-bezier thumb slide; hydro sky (`#38BDF8`) marks tracking/audio switches while brass (`#D9A441`) marks UI/HUD customization switches.

## Do's and Don'ts

### Do:
- **Do** preserve every DOM element ID verified by `tests/verify_i18n_and_ext_v5.py` and keep `i18n` labels wrapped inside `<span class="btn-label" data-i18n="...">` beside inline SVGs so runtime language switches never erase SVG icons.
- **Do** validate every popup change across both `VI` and `EN` strings at fixed `520×580px`, ensuring zero ellipsis clipping on Horizon Strip descriptions, switch labels (`flex-wrap: wrap`), and compact `3×2` keycap slots.
- **Do** keep every visual asset 100% bundled and MV3-CSP compliant (`1.75px` inline SVGs, local `.png`/`.jpg` assets, system font stack, zero `eval` or `innerHTML`).

### Don't:
- **Don't** reintroduce decorative emojis (`🚀`, `⚡`, `💤`, `✨`, `☕`) into popup headers, tabs, buttons, toggle labels, or `#popup-feedback` banners.
- **Don't** collapse the popup back into a single-column scrolling stack (`< 480px`) or hide the bottom `Map Tabs` / voluntary `Support` footer behind scrolling.
- **Don't** inject hard-coded local filesystem paths or unapproved MV3 host permissions anywhere inside `extension/` or design documentation.
