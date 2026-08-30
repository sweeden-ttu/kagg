"""CompetitiveRewardShaper asymmetric stake tests."""

from __future__ import annotations

from kaggriculture_path_b_rebuild import CompetitiveRewardShaper, KaggricultureJSONParser


def test_behind_margin_not_stake_amplified():
    shaper = CompetitiveRewardShaper(
        KaggricultureJSONParser(),
        stake_reference=1000.0,
        margin_scale=100.0,
        mix_bonus_scale=0.0,
        schedule_affinity=0.0,
    )
    shaper.reset_episode()
    obs = {
        "player": 0,
        "day": 1,
        "hour": 0,
        "farms": [{"money": 1000.0}, {"money": 5000.0}],
    }
    # margin = (1000-5000)/100 = -40; stake would amplify to -240 — we keep -40, then clip
    shaped = shaper.shape_reward(obs, raw_reward=0.0)
    assert shaped == -20.0  # clipped to shaper.clip


def test_ahead_margin_is_stake_amplified():
    shaper = CompetitiveRewardShaper(
        KaggricultureJSONParser(),
        stake_reference=1000.0,
        margin_scale=100.0,
        mix_bonus_scale=0.0,
        schedule_affinity=0.0,
    )
    shaper.reset_episode()
    obs = {
        "player": 0,
        "day": 1,
        "hour": 0,
        "farms": [{"money": 5000.0}, {"money": 1000.0}],
    }
    # margin = 40, stake = 6 → 240, clipped to clip=20
    shaped = shaper.shape_reward(obs, raw_reward=0.0)
    assert shaped == 20.0
