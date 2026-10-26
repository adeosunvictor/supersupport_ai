from llm import _extract

def test_cloudflare_response_parsing():
    assert _extract({"result": {"response": "You can publish a space."}}) == "You can publish a space."
    assert _extract({"result": {"response": "<think>private</think>Final."}}) == "Final."
    assert _extract({"result": {"choices": [{"message": {"content": "Okay"}}]}}) == "Okay"
    assert _extract({"result": {}}) == ""
