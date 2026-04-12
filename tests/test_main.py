from types import SimpleNamespace

from click.testing import CliRunner

import core.chat as chat_module
import llm_cli.db_utils as db_utils
import llm_cli.main as main_module
from llm_cli.main import cli
from llm_cli.utils import make_session_title


def test_chat_command_logs_history(monkeypatch, tmp_path):
    db_path = tmp_path / "chat_history.db"

    monkeypatch.setattr(db_utils, "DB_PATH", db_path)
    monkeypatch.setattr(
        main_module,
        "get_config",
        lambda: SimpleNamespace(model="test-model", temperature=0.2),
    )
    monkeypatch.setattr(
        chat_module,
        "stream_chat",
        lambda **kwargs: [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hello back"))]
            )
        ],
    )

    result = CliRunner().invoke(cli, ["chat", "hello there"])

    assert result.exit_code == 0

    sessions = db_utils.get_chat_history()
    assert sessions is not None
    assert len(sessions) == 1

    session_id = sessions[0][0]
    _, _, title, history = db_utils.get_chat_history(session_id)

    assert title == "hello there"
    assert history == [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hello back"},
    ]


def test_make_session_title_shortens_to_first_words():
    assert (
        make_session_title("please help me debug this sqlite locking issue today")
        == "please help me debug this sqlite..."
    )


def test_make_session_title_reads_structured_user_content():
    assert (
        make_session_title(
            [
                {"type": "text", "text": "explain this image of the error"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
            ]
        )
        == "explain this image of the error"
    )
