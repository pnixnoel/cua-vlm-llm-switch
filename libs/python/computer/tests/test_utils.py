from pathlib import Path
import importlib.util
import pytest

# Load utils module directly without importing the whole package
UTILS_PATH = Path(__file__).resolve().parents[1] / "computer" / "utils.py"
spec = importlib.util.spec_from_file_location("utils", UTILS_PATH)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
parse_vm_info = utils.parse_vm_info


def test_parse_vm_info_valid():
    vm_info = {
        "ip_address": "1.2.3.4",
        "name": "test-vm",
        "provider": "lume",
        "status": "running",
        "extra": "ignore"
    }
    assert parse_vm_info(vm_info) == {
        "ip_address": "1.2.3.4",
        "name": "test-vm",
        "provider": "lume",
        "status": "running",
    }


def test_parse_vm_info_empty():
    assert parse_vm_info(None) is None
    assert parse_vm_info({}) is None



