import argparse
from pathlib import Path


def test_cli_builds_and_has_run_subcommand():
    import fargo2radmc3d.cli as cli

    parser: argparse.ArgumentParser = cli.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)

    # Parse without executing: just ensure it wires up correctly
    args = parser.parse_args(["run", "--chdir", "."])
    assert hasattr(args, "func") and callable(args.func)
    assert isinstance(args.chdir, Path)
