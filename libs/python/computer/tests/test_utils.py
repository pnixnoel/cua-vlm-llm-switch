import pytest
import importlib.util
from pathlib import Path

UTILS_PATH = Path(__file__).resolve().parents[1] / "computer" / "utils.py"
spec = importlib.util.spec_from_file_location("utils", UTILS_PATH)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
parse_vm_info = utils.parse_vm_info


def test_parse_vm_info_valid():
    vm_info = {
        "ip_address": "192.168.1.1",
        "name": "test-vm",
        "provider": "lume",
        "status": "running",
        "extra": "ignore"
    }
    expected = {
        "ip_address": "192.168.1.1",
        "name": "test-vm",
        "provider": "lume",
        "status": "running",
    }
    assert parse_vm_info(vm_info) == expected


def test_parse_vm_info_empty():
    assert parse_vm_info({}) is None
