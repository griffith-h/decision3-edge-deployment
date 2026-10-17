import loader,runpy
import setup_helpers,merged_adapter
from variants import engine_with_library
setup_helpers.configure=lambda e:None
merged_adapter.PackedEngine=lambda mode:engine_with_library('/work/libconversion_bridge.so',mode)
runpy.run_path('/projection/test_packing.py',run_name='__main__')
from pathlib import Path
Path('/work/logs/packing-unit.json').rename('/work/logs/conversion-packing-unit.json')
