import asyncio
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

from friday_core import GraphStore, ReminderService, ReminderWorker
from reminder_daemon import deliver as deliver_desktop_notification


REPO = Path(__file__).resolve().parents[1]


class ReminderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reminders = ReminderService(
            GraphStore(Path(self.tmp.name) / "friday.db"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_timezone_is_required(self):
        with self.assertRaisesRegex(ValueError, "timezone"):
            self.reminders.create("test", "2030-01-01T12:00:00")

    def test_due_one_shot_fires_once(self):
        due = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        item = self.reminders.create("Stand up", due)
        self.assertEqual(self.reminders.due()[0]["reminder_id"], item["reminder_id"])
        receipt = self.reminders.mark_fired(item["reminder_id"])
        self.assertEqual(receipt["status"], "fired")
        self.assertEqual(self.reminders.due(), [])

    def test_recurring_reminder_advances_beyond_now(self):
        due = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        item = self.reminders.create("Hydrate", due, interval_seconds=60)
        receipt = self.reminders.mark_fired(item["reminder_id"])
        self.assertEqual(receipt["status"], "scheduled")
        next_due = datetime.fromisoformat(receipt["next_due_at"].replace("Z", "+00:00"))
        self.assertGreater(next_due, datetime.now(UTC))


class ReminderWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reminders = ReminderService(
            GraphStore(Path(self.tmp.name) / "friday.db"))
        due = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        self.item = self.reminders.create("Do not lose me", due)

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_failed_delivery_stays_scheduled_for_retry(self):
        attempts = 0

        async def fail(_receipt):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("notification transport unavailable")

        worker = ReminderWorker(self.reminders, fail, poll_seconds=0.01)
        await worker.start()
        await asyncio.sleep(0.03)
        await worker.stop()

        self.assertGreaterEqual(attempts, 1)
        self.assertEqual(
            self.reminders.due()[0]["reminder_id"], self.item["reminder_id"])

    async def test_running_state_tracks_delivery_loop(self):
        async def deliver(_receipt):
            return None

        worker = ReminderWorker(self.reminders, deliver, poll_seconds=60)
        self.assertFalse(worker.is_running)
        await worker.start()
        self.assertTrue(worker.is_running)
        await worker.stop()
        self.assertFalse(worker.is_running)

    async def test_successful_delivery_is_marked_after_transport_returns(self):
        observed_status = []
        confirmations = []

        async def deliver(_receipt):
            observed_status.append(
                self.reminders.list()[0]["status"])

        async def confirmed(receipt):
            confirmations.append((
                self.reminders.list()[0]["status"], receipt["status"]))

        worker = ReminderWorker(
            self.reminders, deliver, confirmation=confirmed, poll_seconds=0.01)
        await worker.start()
        await asyncio.sleep(0.03)
        await worker.stop()

        self.assertEqual(observed_status, ["scheduled"])
        self.assertEqual(self.reminders.list()[0]["status"], "fired")
        self.assertEqual(confirmations, [("fired", "fired")])


class ReminderDaemonTests(unittest.IsolatedAsyncioTestCase):
    async def test_nonzero_notify_send_exit_is_a_delivery_failure(self):
        failure = subprocess.CalledProcessError(1, ["notify-send"])
        with patch("reminder_daemon.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=failure) as to_thread:
            with self.assertRaises(subprocess.CalledProcessError):
                await deliver_desktop_notification({"text": "Do not lose me"})

        to_thread.assert_awaited_once_with(
            subprocess.run,
            ["notify-send", "Friday reminder", "Do not lose me"],
            capture_output=True, timeout=10, check=True)


class ReminderDeploymentTests(unittest.TestCase):
    def test_only_server_managed_worker_is_deployed(self):
        supervisor_unit = (REPO / "ops" / "friday-supervisor.service").read_text()
        self.assertNotIn("friday-reminders.service", supervisor_unit)

        standalone_units = [
            unit.name for unit in (REPO / "ops").glob("*.service")
            if "reminder_daemon.py" in unit.read_text()
        ]
        self.assertEqual(standalone_units, [])

        server = (REPO / "server.py").read_text()
        self.assertIn("REMINDER_WORKER = ReminderWorker(", server)
        self.assertIn("confirmation=_confirm_reminder", server)


if __name__ == "__main__":
    unittest.main()
