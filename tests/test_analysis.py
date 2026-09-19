import pandas as pd

from analysis.analyze import (
    anonymous_baseline_tables,
    blocked_label_permutation,
    complete_scenario_effects,
    holm_adjust,
    matched_country_gaps,
)


def test_country_gap_requires_two_repetitions_per_country():
    rows = [
        {
            "condition": "labelled",
            "scenario_id": "S1",
            "model_id": "M1",
            "country": "Kyrgyzstan",
            "investment_score_0_100": 40,
        },
        {
            "condition": "labelled",
            "scenario_id": "S1",
            "model_id": "M1",
            "country": "United States",
            "investment_score_0_100": 50,
        },
        {
            "condition": "labelled",
            "scenario_id": "S1",
            "model_id": "M1",
            "country": "United States",
            "investment_score_0_100": 52,
        },
    ]

    df = pd.DataFrame(rows)

    gaps = matched_country_gaps(
        df,
        "investment_score_0_100",
        "labelled",
        min_repetitions=2,
    )

    assert gaps.empty


def test_scenario_effect_requires_every_model():
    index = pd.MultiIndex.from_tuples(
        [
            ("S1", "M1"),
            ("S1", "M2"),
            ("S1", "M3"),
            ("S2", "M1"),
            ("S2", "M2"),
        ],
        names=[
            "scenario_id",
            "model_id",
        ],
    )

    gaps = pd.Series(
        [-2.0, -4.0, -6.0, -1.0, -3.0],
        index=index,
    )

    result = complete_scenario_effects(
        gaps,
        ["M1", "M2", "M3"],
    )

    assert list(result.index) == ["S1"]
    assert result.loc["S1"] == -4.0



def test_holm_adjustment():
    adjusted = holm_adjust(
        {
            "a": 0.01,
            "b": 0.04,
            "c": 0.03,
        }
    )

    assert abs(
        adjusted["a"] - 0.03
    ) < 1e-12

    assert abs(
        adjusted["b"] - 0.06
    ) < 1e-12

    assert abs(
        adjusted["c"] - 0.06
    ) < 1e-12


def test_blocked_permutation_is_deterministic():
    rows = []

    for country, scores in [
        (
            "Kyrgyzstan",
            [10, 11, 12],
        ),
        (
            "United States",
            [5, 6, 7],
        ),
    ]:
        for score in scores:
            rows.append(
                {
                    "condition": "labelled",
                    "scenario_id": "S1",
                    "model_id": "M1",
                    "country": country,
                    "investment_score_0_100": score,
                }
            )

    df = pd.DataFrame(rows)

    first, blocks_first = (
        blocked_label_permutation(
            df,
            n_simulations=1000,
            seed=123,
        )
    )

    second, blocks_second = (
        blocked_label_permutation(
            df,
            n_simulations=1000,
            seed=123,
        )
    )

    assert first == second
    assert len(blocks_first) == 1
    assert len(blocks_second) == 1
    assert first["n_complete_blocks"] == 1
    assert (
        0
        < first[
            "plus_one_corrected_two_sided_p"
        ]
        <= 1
    )



def test_anonymous_baseline_requires_two_valid_repetitions():
    rows = []

    # Complete anonymous and Kyrgyzstan cells,
    # but only one US response: this model must be excluded.
    for score in [50, 52]:
        rows.append(
            {
                "condition": "anonymous",
                "scenario_id": "S1",
                "model_id": "M1",
                "country": "Anonymous",
                "investment_score_0_100": score,
            }
        )

    for score in [45, 47]:
        rows.append(
            {
                "condition": "labelled",
                "scenario_id": "S1",
                "model_id": "M1",
                "country": "Kyrgyzstan",
                "investment_score_0_100": score,
            }
        )

    rows.append(
        {
            "condition": "labelled",
            "scenario_id": "S1",
            "model_id": "M1",
            "country": "United States",
            "investment_score_0_100": 55,
        }
    )

    model_level, scenario_level = (
        anonymous_baseline_tables(
            pd.DataFrame(rows),
            ["M1"],
            min_repetitions=2,
        )
    )

    assert model_level.empty
    assert scenario_level.empty


def test_anonymous_scenario_average_requires_complete_model_set():
    rows = []

    for scenario_id, models in [
        ("S1", ["M1", "M2", "M3"]),
        ("S2", ["M1", "M2"]),
    ]:
        for model_id in models:
            for score in [50, 50]:
                rows.append(
                    {
                        "condition": "anonymous",
                        "scenario_id": scenario_id,
                        "model_id": model_id,
                        "country": "Anonymous",
                        "investment_score_0_100": score,
                    }
                )

            for score in [45, 45]:
                rows.append(
                    {
                        "condition": "labelled",
                        "scenario_id": scenario_id,
                        "model_id": model_id,
                        "country": "Kyrgyzstan",
                        "investment_score_0_100": score,
                    }
                )

            for score in [55, 55]:
                rows.append(
                    {
                        "condition": "labelled",
                        "scenario_id": scenario_id,
                        "model_id": model_id,
                        "country": "United States",
                        "investment_score_0_100": score,
                    }
                )

    model_level, scenario_level = (
        anonymous_baseline_tables(
            pd.DataFrame(rows),
            ["M1", "M2", "M3"],
            min_repetitions=2,
        )
    )

    assert len(model_level) == 5
    assert list(
        scenario_level["scenario_id"]
    ) == ["S1"]

    assert (
        scenario_level.loc[
            0,
            "kyrgyzstan_minus_anonymous",
        ]
        == -5.0
    )

    assert (
        scenario_level.loc[
            0,
            "united_states_minus_anonymous",
        ]
        == 5.0
    )
