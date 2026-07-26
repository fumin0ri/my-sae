import importlib.util
import random
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "make_research_data.py"
SPEC = importlib.util.spec_from_file_location("make_research_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
research_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(research_data)


def test_all_task_families_reach_requested_final_state() -> None:
    shifts = [1, 3, 2, 1]
    for family in ("fsm", "arithmetic", "logic"):
        for final_state in range(4):
            start = research_data.initial_state_for_final(
                family,
                final_state,
                shifts,
            )
            if family == "logic":
                recovered = start ^ research_data.logic_toggle_mask(shifts)
            else:
                recovered = (start + sum(shifts)) % 4
            assert recovered == final_state


def test_all_task_families_have_four_paraphrase_templates() -> None:
    for family in ("fsm", "arithmetic", "logic"):
        prompts = {
            research_data.render_task_prompt(
                family,
                problem_id=1,
                start=0,
                shifts=[1, 2, 3],
                template_id=template,
                rng=random.Random(template),
            )
            for template in range(4)
        }
        assert len(prompts) == 4
        assert all(prompt for prompt in prompts)
