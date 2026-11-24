# Edition look: world S templates in production (design)

Date: 2026-11-24. Issue: SIY-204. Owner of decisions: David. Status: approved by David on 2026-11-24 ("yes", "wall is better than silence", "go with the critique"), with merge to main authorised once tests pass and an independent review approves, and retirement of the five old template names authorised.

## Intent

Replace the five current plate archetypes with the set David chose over five review rounds. The plate stays one raster per theme (light and dark) in the README. The photograph is shown whole and is the piece; the word, gloss, date and credit are small and exactly placed. The photograph choice is not touched: David's taste shapes the plate only.

## Goals

- Seven templates: Frame, Subtitle, Plate, Sleeve, Bleed, Entry, Wall. Wall's label is written the way a museum does (photographer, then word and year) and set larger so it reads at phone width. Key is dropped (David: "go with the critique", which said its swatches read as interface).
- Rotation: no template two nights running, least recently used first; Frame and Subtitle only when the photograph passes the measured calm test; no Key.
- Lettering: word in three voices (old-style italic most often, wide grotesque capitals, condensed capitals). The gloss face pairs with the word voice: italic word with sans or same-family roman gloss (alternating); wide capitals with old-style italic; condensed capitals with old-style italic. The Didone italic pairing and the condensed-with-sans pairing are dropped (critique, David agreed).
- Colour on half the nights. On a colour night: the full stop takes the night's colour, plus one device (the word in colour, or a short rule). On other nights no colour at all. Black-and-white photographs get no colour, since the colour must come from the photograph (critique, David agreed); they fall on the colourless half.
- Alignment rotates inside Bleed (split, centred). Frame centres its words only on a symmetric photograph.
- Date and credit on the plate as a colophon: "No. N · date · Photograph by Name", small tracked capitals, placed per template.
- Never lose an edition: if the new renderer raises, render the current look.

## Non-goals

- No change to photograph selection, scoring or the engine.
- No workflow, repository variable or env line change. No edits to adk_pipeline.py, understanding.py, state_store.py, private_run.py, dispatch or relay files.
- Momentum-driven loud and quiet: rule built later, when the engine's momentum figure is repaired (24h_shift reads 0.24 every night; breakout flag always true).

## Anti-goals

- Big or bold lettering, colour fields, generic faces, templates that differ only in spacing.
- Any public text beyond word, gloss, number, date, credit. No private orbit data.

## Constraints found in the code

- Plates are drawn with Pillow (raqm off). Fonts live in assets/fonts (OFL). New fonts to add: Sorts Mill Goudy (roman and italic), Darker Grotesque, Antonio, all OFL.
- README embeds plates by width only, so plates may vary in height. The SVG fallback in visual_router hard-codes 1200x800 and must take the plate's real size.
- Template names are stored in curatorial_archetype and archetype_history_30d. adk_pipeline treats them as plain strings, so new names pass through.
- publication.py checks the manifest archetype equals the entry's: unchanged behaviour.

## Approaches considered

1. Port the browser comps to Pillow inside the existing compositor seam (chosen). Keeps the nightly runner unchanged; testable in CI; one renderer.
2. Render the HTML comps with headless Chromium in the nightly. Exact parity, but needs a workflow change and a browser in the run, which tonight's constraints rule out.
3. Pre-render nothing; draw SVG text. GitHub shows SVG in img tags but font loading in SVG is unreliable; rejected.

## Design

- New module src/zeitgeist/edition_look.py: template layouts, the calm test (per-line contrast and edge measure, ported from the comps), colophon, colour derivation, pairing table, rotation. Pure functions returning a PIL image per theme.
- curatorial_compositor.render_curatorial_plate_pair: when assets/edition_look.json says {"look": "s1"}, call edition_look; on any exception log and fall back to the current archetype path.
- Selection: edition_look.select_template(history, photo analysis, night index) returns template, voice, gloss face, colour device, alignment. History read from archetype_history_30d (names like S1_FRAME).
- visual_router: SVG width and height from the rendered plate.
- Edition number: count of published editions plus one, read from the feed passed to the router (fallback: omit "No.").

## Tests

- Every template renders on synthetic photographs (light, dark, grey, busy, symmetric) in both themes without error; text stays inside the plate; contrast of words on photographs at least 7 (Frame) and 6 (Subtitle); colophon at least 4.5.
- Rotation: 30 simulated nights never repeat a template on consecutive nights and colour falls on half the nights.
- Fallback: a forced exception in edition_look yields the current look.
- Golden check against the browser comps on three archived nights: text boxes within a few pixels.

## Left to the builder

Exact Pillow metrics (tracking, line height) to match the comps; file layout inside edition_look.py.
