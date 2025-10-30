import importlib
import pytest


def test_package_imports():
    mod = importlib.import_module("fargo2radmc3d")
    assert mod is not None


def test_dust_bhmie_imports():
    mod = importlib.import_module("fargo2radmc3d.dust.bhmie")
    assert mod is not None


def test_dust_makedustopac_imports():
    mod = importlib.import_module("fargo2radmc3d.dust.makedustopac")
    assert mod is not None


def test_imaging_beam_imports():
    pytest.importorskip("astropy")
    mod = importlib.import_module("fargo2radmc3d.imaging.beam")
    assert mod is not None
