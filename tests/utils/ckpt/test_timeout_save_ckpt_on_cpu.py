import unittest
from unittest.mock import patch

from verl.utils.checkpoint.checkpoint_manager import convert_timeout_to_seconds, should_save_ckpt_timeout


class TestTimeoutCheckpointing(unittest.TestCase):
    def test_convert_timeout_to_seconds(self):
        self.assertEqual(convert_timeout_to_seconds("00:00:01:30"), 90)
        self.assertEqual(convert_timeout_to_seconds("01:00:00:00"), 86400)

    def test_no_deadline_returns_false(self):
        self.assertFalse(should_save_ckpt_timeout(max_steps_duration=100))

    @patch("verl.utils.checkpoint.checkpoint_manager.time.time", return_value=1000.0)
    def test_explicit_timeout_triggers_save(self, _mock_time):
        self.assertTrue(
            should_save_ckpt_timeout(
                max_steps_duration=30,
                save_ckpt_duration=60,
                checkpoint_must_save_by="00:00:18:00",
                start_time=0.0,
            )
        )

    @patch("verl.utils.checkpoint.checkpoint_manager.time.time", return_value=1000.0)
    def test_explicit_timeout_with_sufficient_headroom_does_not_trigger(self, _mock_time):
        self.assertFalse(
            should_save_ckpt_timeout(
                max_steps_duration=30,
                save_ckpt_duration=60,
                checkpoint_must_save_by="00:00:20:00",
                start_time=0.0,
            )
        )

    @patch.dict("os.environ", {"SLURM_JOB_END_TIME": "1085"}, clear=True)
    @patch("verl.utils.checkpoint.checkpoint_manager.time.time", return_value=1000.0)
    def test_slurm_end_time_triggers_save(self, _mock_time):
        self.assertTrue(
            should_save_ckpt_timeout(
                max_steps_duration=20,
                save_ckpt_duration=60,
            )
        )

    @patch.dict("os.environ", {"SLURM_JOB_END_TIME": "1200"}, clear=True)
    @patch("verl.utils.checkpoint.checkpoint_manager.time.time", return_value=1000.0)
    def test_slurm_end_time_with_headroom_does_not_trigger(self, _mock_time):
        self.assertFalse(
            should_save_ckpt_timeout(
                max_steps_duration=20,
                save_ckpt_duration=60,
            )
        )

    @patch.dict("os.environ", {"SLURM_JOB_END_TIME": "2000"}, clear=True)
    @patch("verl.utils.checkpoint.checkpoint_manager.time.time", return_value=1000.0)
    def test_explicit_timeout_overrides_slurm_end_time(self, _mock_time):
        self.assertTrue(
            should_save_ckpt_timeout(
                max_steps_duration=30,
                save_ckpt_duration=60,
                checkpoint_must_save_by="00:00:18:00",
                start_time=0.0,
            )
        )
