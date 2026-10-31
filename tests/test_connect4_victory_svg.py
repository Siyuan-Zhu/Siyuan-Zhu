"""Unit tests for Connect 4 Victory Celebration SVG renderer (WS4 Seam 3)."""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from src.game_engine import (
    build_initial_game_state,
    render_board_svg,
    render_readme_game_section,
)


class TestConnect4VictorySvg(unittest.TestCase):
    """Verify winning 4-disc aura, golden strike-through beam, non-winning dimming, and celebration banner."""

    def _build_completed_winning_state(self) -> dict:
        state = build_initial_game_state("connect4", round_id=7)
        state["status"] = "COMPLETED"
        state["round_over"] = True
        state["mode"] = "VS_BOT"
        state["winner"] = "red"
        state["finisher_handle"] = "maya-lin"
        state["last_mover"] = "maya-lin"
        state["moves_this_round"] = 11
        state["last_elo_delta"] = 27
        # 4 winning Red discs at row 5, cols 1..4 -> (5, 1), (5, 2), (5, 3), (5, 4)
        state["board"][5][1] = 1
        state["board"][5][2] = 1
        state["board"][5][3] = 1
        state["board"][5][4] = 1
        # Non-winning Bot discs at row 4, cols 1..3
        state["board"][4][1] = 2
        state["board"][4][2] = 2
        state["board"][4][3] = 2
        state["winning_cells"] = [(5, 1), (5, 2), (5, 3), (5, 4)]
        return state

    def test_completed_board_renders_victory_glow_beam_dimming_and_banners(self) -> None:
        state = self._build_completed_winning_state()
        svg = render_board_svg(state)

        # 1. XML well-formedness
        root = ET.fromstring(svg)
        self.assertEqual(root.attrib.get("viewBox"), "0 0 580 360")

        # 2. Defs include #win-glow filter and #victory-beam gradient
        self.assertIn('id="win-glow"', svg)
        self.assertIn('id="victory-beam"', svg)

        # 3. Winning 4-disc outer aura: stroke="#9B482D" stroke-width="3.5" and r="23" outer ring
        self.assertIn('stroke="#9B482D"', svg)
        self.assertIn('stroke-width="3.5"', svg)
        self.assertIn('r="23" fill="none" stroke="#C85A32" stroke-width="2" opacity="0.85"', svg)

        # 4. Golden strike-through line from (5, 1) -> (cx=160, cy=306) to (5, 4) -> (cx=355, cy=306)
        self.assertIn(
            '<line x1="160" y1="306" x2="355" y2="306" '
            'stroke="url(#victory-beam)" stroke-width="6" stroke-linecap="round" filter="url(#win-glow)"/>',
            svg,
        )

        # 5. Non-winning discs on COMPLETED board are dimmed with opacity="0.35"
        self.assertIn('opacity="0.35"', svg)

        # 6. Top celebration banner pill & footer next-round CTA
        self.assertNotIn("CONNECTS 4 IN", svg)
        self.assertIn("Congratulations, @maya-lin!", render_readme_game_section(state))
        self.assertIn("Next round: choose a column.", render_readme_game_section(state))

    def test_active_board_has_no_victory_line_or_dimming(self) -> None:
        state = build_initial_game_state("connect4", round_id=3)
        state["board"][5][3] = 1
        state["board"][4][3] = 2
        svg = render_board_svg(state)

        ET.fromstring(svg)
        self.assertNotIn('stroke="url(#victory-beam)"', svg)
        self.assertNotIn('opacity="0.35"', svg)
        self.assertNotIn("CONNECTS 4 IN", svg)
        self.assertNotIn("Drop in Any Column to Start Round", svg)

        readme_md = render_readme_game_section(state)
        self.assertIn('width="580"', readme_md)
        self.assertIn("game:move:connect4:col:1:turn=1", readme_md)
        self.assertIn("game:mode:connect4:new_pvp:turn=1", readme_md)


if __name__ == "__main__":
    unittest.main()
