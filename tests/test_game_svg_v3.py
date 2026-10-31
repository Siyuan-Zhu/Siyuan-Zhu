"""Approved v3 board contract at the public SVG and README builder seams."""
import xml.etree.ElementTree as ET
import json

from src.game_engine import build_initial_game_state, render_board_svg, render_readme_game_section

NS = {"s": "http://www.w3.org/2000/svg"}


def test_board_has_editorial_frame_and_dark_palette_without_external_requests():
    state = build_initial_game_state()
    state["board"][5][0] = 1
    state["board"][5][1] = 2
    svg = render_board_svg(state)
    root = ET.fromstring(svg)
    assert root.attrib["viewBox"] == "0 0 580 360"
    assert "THE DAILY COLUMN" not in "".join(root.itertext())
    texts = [node.text for node in root.findall(".//s:text", NS)]
    assert all(label in texts for label in ("I", "II", "III", "IV", "V", "VI", "VII"))
    css = "".join(node.text or "" for node in root.findall("s:style", NS))
    assert "@media (prefers-color-scheme: dark)" in css
    assert "#E7E2DA" in css and "#2A2926" in css
    assert root.findall(".//s:path[@class='board-grid']", NS)
    assert "http" not in svg.replace('xmlns="http://www.w3.org/2000/svg"', "")


def test_board_footer_ranks_only_actual_ledger_ratings_and_escapes_names():
    ledger = {"players": {
        "fourth": {"elo": 810},
        "second": {"elo": 920},
        "first": {"elo": 1234, "wins": 5},
        "a&b": {"elo": 850},
        "unrated": {},
    }}
    md = render_readme_game_section(build_initial_game_state(),elo_ledger=ledger)
    assert '| Player | Rating |' in md
    assert md.index('@first | 1234') < md.index('@second | 920') < md.index('@a&amp;b | 850')
    assert 'unrated' not in md
    assert 'No recorded ratings' in render_readme_game_section(build_initial_game_state())
    assert 'elo-leaderboard' not in render_board_svg(build_initial_game_state(),elo_ledger=ledger)


def test_readme_builder_uses_supplied_store_ratings_in_board_asset(tmp_path):
    from scripts.build_readme import render_profile_markdown
    from src.state_store import InMemoryStateStore

    (tmp_path / "data").mkdir()
    (tmp_path / "data/zeitgeist_state.json").write_text(json.dumps({"analysis": {}}))
    # A stale file must never supply the visible scores when another store is selected.
    (tmp_path / "data/hall_of_records.json").write_text(json.dumps({"players": {"stale": {"elo": 9999}}}))
    store = InMemoryStateStore()
    store.save_game_state(build_initial_game_state())
    store.save_elo_ledger({"players": {"actual-player": {"elo": 948}}})
    md = render_profile_markdown(tmp_path, state_store=store)
    root = ET.parse(tmp_path / "assets/game_board.svg").getroot()
    assert "| @actual-player | 948 |" in md
    assert "stale" not in md
    assert "game:move:connect4:col:1:turn=1" in md
    assert "game:mode:connect4:new_pvp:turn=1" in md


def test_victory_keeps_editorial_labels_ratings_and_dashed_winning_halos():
    state = build_initial_game_state()
    state.update(status="COMPLETED", round_over=True, winner="red", finisher_handle="finisher",
                 winning_cells=[(5, col) for col in range(4)], moves_this_round=7)
    state["board"][5][:4] = [1, 1, 1, 1]
    state["board"][4][0] = 2
    root = ET.fromstring(render_board_svg(state, elo_ledger={"players": {"finisher": {"elo": 827}}}))
    text = "".join(root.itertext())
    assert "THE DAILY COLUMN" not in text and "VICTORY" not in text
    assert "| @finisher | 827 |" in render_readme_game_section(state,elo_ledger={"players":{"finisher":{"elo":827}}})
    assert len(root.findall(".//s:circle[@stroke-dasharray]", NS)) == 4
    assert root.findall(".//s:circle[@opacity='0.35']", NS)
    assert "VII" in [node.text for node in root.findall(".//s:text", NS)]
