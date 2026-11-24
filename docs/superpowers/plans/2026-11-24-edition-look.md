# Edition look (world S) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw the nightly edition plate in the seven templates David chose, with rotation, colour on half the nights and an on-plate colophon, falling back to the current look on any error.

**Architecture:** A new pure module `src/zeitgeist/edition_look.py` measures the photograph, plans the night (template, lettering, gloss face, colour device, alignment) and draws light and dark plates with Pillow. `visual_router.select_and_render_pulse_assets` calls it when `assets/edition_look.json` says `{"look": "s1"}`, and otherwise (or on any exception) keeps the current archetype path. The SVG fallback takes the plate's real size.

**Tech Stack:** Python 3, Pillow (basic layout, no raqm), numpy, pytest. Fonts OFL: Sorts Mill Goudy (roman, italic), Darker Grotesque (variable wght), Antonio (variable wght).

**Spec:** `docs/superpowers/specs/2026-11-24-edition-look-design.md`

## Global Constraints

- Plate width 1200 px; height varies by template; light page #ffffff ink #1b1e22; dark page #0d1117 ink #eef2f6; ink on photographs #ffffff or #14110f.
- Template names: S1_FRAME, S1_SUBTITLE, S1_PLATE, S1_SLEEVE, S1_BLEED, S1_ENTRY, S1_WALL (32 characters max, stored in curatorial_archetype).
- Words on a photograph: contrast at least 7 (Frame) or 6 (Subtitle), edge measure at most 0.009 (Frame) or 0.011 (Subtitle); colophon on a photograph contrast at least 4.5.
- Colour on half the nights (date ordinal parity); none on black-and-white photographs; the colour comes from the photograph only.
- No edits to adk_pipeline.py, understanding.py, state_store.py, private_run.py, any workflow, dispatch or relay file. No repository variable.
- Public text on the plate: word, gloss, edition number, date, photographer. Nothing else.

## Review Focus

- A word of 11+ letters (Climacteric, Insinuation) must shrink to fit its template, never overflow the plate.
- A gloss of 50 characters must stay within two balanced lines and never end a line on a preposition or article.
- A night where no photograph corner is calm must never pick Frame or Subtitle.
- A missing photographer name (generated images) must still give a sensible credit.
- Any exception inside edition_look must leave the current look's plates on disk and a valid archetype name.

---

### Task 1: Fonts, measurement and night planning

**Files:**
- Create: `assets/fonts/SortsMillGoudy-Regular.ttf`, `SortsMillGoudy-Italic.ttf`, `DarkerGrotesque-Variable.ttf`, `Antonio-Variable.ttf`, `OFL-SortsMillGoudy.txt`, `OFL-DarkerGrotesque.txt`, `OFL-Antonio.txt`
- Create: `src/zeitgeist/edition_look.py`
- Test: `tests/test_edition_look.py`

**Interfaces:**
- Produces: `analyse_photo(img: Image.Image) -> PhotoFacts` (dataclass: `colour: tuple[int,int,int] | None`, `symmetry: float`); `plan_night(date_iso: str, history: Sequence[str], facts: PhotoFacts) -> NightPlan` (dataclass: `template, voice, gloss_face, colour_on, device, centred`); constants `TEMPLATES`, `VOICES = ("italic","wide","condensed")`, `PAIRS = {"italic": ("sans","roman"), "wide": ("italic",), "condensed": ("italic",)}`.

- [ ] Test `test_plan_never_repeats_previous_template`: 30 consecutive dates, each plan's template differs from the previous; all seven appear.
- [ ] Test `test_colour_on_half_the_nights_and_never_on_grey`: colour_on true on exactly alternate dates for a colour photo, false on every date for a grey photo.
- [ ] Test `test_on_photo_templates_need_calm`: with `calm=False` for every placement, Frame and Subtitle are never chosen.
- [ ] Implement analyse_photo (port of the comps' median-cut colour pick: saturation at least 0.22, lightness 0.16 to 0.84, share at least 0.025; symmetry = mean mirrored luminance difference on a 60x40 thumbnail) and plan_night (least recently used of the eligible templates, voice by date ordinal mod 4 over italic, wide, condensed, italic; gloss face alternates by ordinal//4; device word on photographs, else word or rule alternating; Bleed centred on alternate uses).
- [ ] Run `pytest tests/test_edition_look.py -q`; commit.

### Task 2: Drawing the seven templates

**Files:** Modify `src/zeitgeist/edition_look.py`; Test `tests/test_edition_look.py`

**Interfaces:**
- Produces: `render_edition_pair(photo: Image.Image, *, word: str, gloss: str, date_iso: str, credit: str, plan: NightPlan) -> tuple[Image.Image, Image.Image]` (light, dark); `fits_on_photo(photo, word, gloss, template, voice, gloss_face) -> bool` used by plan_night's calm check.

- [ ] Test `test_every_template_renders_both_themes`: synthetic light, dark, grey and busy photographs, every template and voice, plates are 1200 wide, text pixels inside the plate.
- [ ] Test `test_long_word_shrinks`: "Climacteric" in every template never exceeds its measure.
- [ ] Test `test_gloss_breaks_balanced`: a 50-character gloss gives at most two lines and no line ends with of, in, the, a, an, to, and.
- [ ] Implement layouts ported from the approved comps (sizes in the spec's comps: Frame 1200x800 corners at 58/50 px; Subtitle gloss centred 52 px from the foot; Plate photograph 936x624; Sleeve padding 72/132/84 with catalogue number beside the word; Bleed photograph 1200x800 with word and gloss beneath; Entry photograph 816x544 with run-in entry; Wall photograph 735x490 at 96,110 with a museum label at x 872, width 290, aligned to the photograph's foot).
- [ ] Run tests; render the 23 archived nights to a private folder and compare by eye with the comps; commit.

### Task 3: Wiring, switch and fallback

**Files:** Create `assets/edition_look.json`; Modify `src/zeitgeist/visual_router.py` (archetype choice and plate rendering block, SVG size); Test `tests/test_edition_look.py`

- [ ] Test `test_router_uses_edition_look_when_switched_on`: offline demo run with the switch file writes S1_* archetype and plates 1200 wide.
- [ ] Test `test_router_falls_back_on_error`: edition_look.render_edition_pair patched to raise; the run still writes plates and a valid old archetype.
- [ ] Test `test_svg_matches_plate_size`: daily_pulse.svg width and height equal the dark plate's size.
- [ ] Implement; run the full suite (`pytest -q --basetemp=$TMPDIR/pt`), expecting only the known failure `tests/test_owner_alert_runner.py::test_success_step_recovered_only_after_prior_attempt`; commit; open the pull request.
