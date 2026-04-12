from types import SimpleNamespace

from core.chat import collect_stream_response, extract_chunk_text


def test_extract_chunk_text_reads_delta_content():
    chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))]
    )

    assert extract_chunk_text(chunk) == "hello"


def test_extract_chunk_text_reads_structured_delta_content():
    chunk = {
        "choices": [
            {
                "delta": {
                    "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "text", "text": {"value": "world"}},
                    ]
                }
            }
        ]
    }

    assert extract_chunk_text(chunk) == "hello world"


def test_collect_stream_response_joins_all_text_and_calls_callback():
    seen = []
    response_stream = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" world"))]),
    ]

    result = collect_stream_response(response_stream, on_chunk=seen.append)

    assert result == "hello world"
    assert seen == ["hello", " world"]
