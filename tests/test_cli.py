import os
import sys

from typer.testing import CliRunner

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import cli


def test_verbose_flag_in_help():
    runner = CliRunner()
    result = runner.invoke(cli.app, ["merge", "--help"])
    assert result.exit_code == 0
    assert "--verbose" in result.output
    assert "-v" in result.output
