from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cli.main import agent


@pytest.fixture
def mock_get_agent():
    with patch('core.agents.sml_agents.get_agent') as mock:
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Task completed successfully"
        mock.return_value = mock_agent
        yield mock

@pytest.fixture
def runner():
    return CliRunner()

def test_agent_direct_input(runner, mock_get_agent):
    """Test agent command with direct input"""
    result = runner.invoke(agent, ["write a hello world program"])
    assert result.exit_code == 0
    assert "Task completed successfully" in result.output

def test_agent_file_input(runner, mock_get_agent, tmp_path):
    """Test agent command with file input"""
    # Create temporary file with task
    task_file = tmp_path / "task.txt"
    task_file.write_text("write a hello world program")
    
    result = runner.invoke(agent, ["--file", str(task_file)])
    assert result.exit_code == 0
    assert "Task completed successfully" in result.output

def test_agent_pipe_input(runner, mock_get_agent):
    """Test agent command with piped input"""
    result = runner.invoke(agent, input="write a hello world program")
    assert result.exit_code == 0
    assert "Task completed successfully" in result.output

def test_agent_no_input(runner, mock_get_agent):
    """Test agent command with no input"""
    result = runner.invoke(agent)
    assert result.exit_code == 0
    assert "No task provided" in result.output

def test_agent_with_model_override(runner, mock_get_agent):
    """Test agent command with model override"""
    result = runner.invoke(agent, ["--model", "gpt-4", "write a hello world program"])
    assert result.exit_code == 0
    assert "Task completed successfully" in result.output

def test_agent_error_handling(runner):
    """Test agent command error handling"""
    with patch('core.agents.sml_agents.get_agent') as mock:
        mock.side_effect = Exception("Agent error")
        result = runner.invoke(agent, ["write a hello world program"])
        assert result.exit_code == 0
        assert "Error: Agent error" in result.output

@pytest.mark.integration
def test_agent_real_integration(runner):
    """Integration test with real agent (marked to be run explicitly)"""
    result = runner.invoke(agent, ["write a python hello world program"])
    assert result.exit_code == 0
    assert len(result.output) > 0
