"""Scenario engine: load, blend, and manage macro scenario sets."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from src.hybrid_macro_vol.models import Scenario, ScenarioSet


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_scenario_maps(path: str | Path | None = None) -> dict[str, ScenarioSet]:
    """
    Load scenario maps from a YAML file.

    Returns a dict of event_type → ScenarioSet.
    """
    if path is None:
        path = _CONFIG_DIR / "scenario_maps.yaml"
    path = Path(path)

    with open(path) as f:
        raw = yaml.safe_load(f)

    result: dict[str, ScenarioSet] = {}
    for key, data in raw.items():
        scenarios = [Scenario(**s) for s in data["scenarios"]]
        ss = ScenarioSet(
            name=data["name"],
            event_type=data["event_type"],
            scenarios=scenarios,
            source_description=data.get("source_description", ""),
        )
        result[key] = ss
    return result


def blend_with_prediction_market(
    scenario_set: ScenarioSet,
    event_probability: float,
    event_scenario_names: list[str] | None = None,
) -> ScenarioSet:
    """
    Adjust scenario probabilities using a prediction market event probability.

    The prediction market gives us P(event). We redistribute that probability
    across the "event" scenarios (e.g. mild/severe/crisis recession) proportionally,
    and assign 1 - P(event) to the "no event" scenarios.

    Args:
        scenario_set: Base scenario set to adjust
        event_probability: Probability from prediction market (0-1)
        event_scenario_names: Which scenarios represent the event occurring.
            If None, all scenarios except the first are treated as event scenarios.
    """
    ss = copy.deepcopy(scenario_set)

    if event_scenario_names is None:
        # Convention: first scenario is "no event", rest are "event" scenarios
        no_event = [ss.scenarios[0]]
        event_scenarios = ss.scenarios[1:]
    else:
        no_event = [s for s in ss.scenarios if s.name not in event_scenario_names]
        event_scenarios = [s for s in ss.scenarios if s.name in event_scenario_names]

    if not event_scenarios:
        return ss

    # Current total probability of event scenarios
    current_event_prob = sum(s.probability for s in event_scenarios)

    if current_event_prob < 1e-9:
        # All event probs are zero, distribute evenly
        for s in event_scenarios:
            s.probability = event_probability / len(event_scenarios)
    else:
        # Scale event scenarios proportionally to match prediction market prob
        scale = event_probability / current_event_prob
        for s in event_scenarios:
            s.probability *= scale

    # Distribute remaining probability to no-event scenarios
    remaining = 1.0 - event_probability
    current_no_event_prob = sum(s.probability for s in no_event)
    if current_no_event_prob > 1e-9 and no_event:
        scale = remaining / current_no_event_prob
        for s in no_event:
            s.probability *= scale
    elif no_event:
        for s in no_event:
            s.probability = remaining / len(no_event)

    return ss


def create_custom_scenario_set(
    name: str,
    event_type: str,
    scenario_defs: list[dict[str, Any]],
) -> ScenarioSet:
    """Create a ScenarioSet from a list of dicts."""
    scenarios = [Scenario(**sd) for sd in scenario_defs]
    return ScenarioSet(name=name, event_type=event_type, scenarios=scenarios)
