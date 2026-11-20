import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from unittest.mock import MagicMock
from h3vm.streaming_exact import StreamingExactRuntime


class TestTrimSafety(unittest.TestCase):
    def setUp(self):
        # Create runtime using __new__ to isolate _trim_one unit test
        self.runtime = StreamingExactRuntime.__new__(StreamingExactRuntime)
        self.runtime._warned_trim = False

    def test_trim_one_caps_to_resident_vbar(self):
        patcher = MagicMock()
        mock_vbar = MagicMock()
        mock_vbar.loaded_size.return_value = 500_000_000  # 500 MB resident
        patcher._vbar_get.return_value = mock_vbar
        patcher.loaded_size.return_value = 500_000_000
        patcher.offload_device = "cpu"
        patcher.partially_unload.return_value = 500_000_000

        # When force_all is True, request must equal resident bytes, NEVER 1e15!
        freed, dt = self.runtime._trim_one(patcher, target_bytes=0, force_all=True)
        self.assertEqual(freed, 500_000_000)

        # Verify partially_unload received exactly 500_000_000, NOT 1e15
        patcher.partially_unload.assert_called_once_with("cpu", 500_000_000)

    def test_trim_one_skips_when_empty(self):
        patcher = MagicMock()
        mock_vbar = MagicMock()
        mock_vbar.loaded_size.return_value = 0
        patcher._vbar_get.return_value = mock_vbar
        patcher.loaded_size.return_value = 0
        patcher.offload_device = "cpu"

        freed, dt = self.runtime._trim_one(patcher, target_bytes=0, force_all=True)
        self.assertEqual(freed, 0)
        # Must not even call partially_unload when empty
        patcher.partially_unload.assert_not_called()


if __name__ == "__main__":
    unittest.main()

