#!/usr/bin/env python

"""Tests for `silvio` package."""

import pytest

import os

from silvio import first, DATADIR


def test_datadir_exists () :
    """Check if directory of DATADIR actually exists."""
    datadir_exists = os.path.isdir(DATADIR)
    assert datadir_exists == True


def test_use_an_utility () :
    """Test usage of a simple utility, just to see if import is correct."""
    el = first( ["aaa","bbb","ccc"], lambda it : it == "bbb" )
    assert el == "bbb"
