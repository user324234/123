#!/usr/bin/env python
# -*- coding: utf-8 -*-
from click.testing import CliRunner

from hydrus_dd.__main__ import main


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert result.output
