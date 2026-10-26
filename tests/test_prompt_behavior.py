from app.generation.prompt import SYSTEM

def test_grounding_prompt_present():
    assert isinstance(SYSTEM, str) and len(SYSTEM) > 100
