import unittest
from unittest.mock import Mock

from friday_core.task_orchestration import RecoveredBatchFinalizer
from friday_core.worker import BatchExecutionOutcome


class RecoveredBatchFinalizerTests(unittest.TestCase):
    def setUp(self):
        self.tasks = Mock()
        self.graph = Mock()
        self.finalizer = RecoveredBatchFinalizer(
            self.tasks, self.graph, Mock(), Mock())
        self.tasks.step_batch.return_value = {
            "task_id": "task", "steps": [{"status": "cancelled"}]}
        self.tasks.get.return_value = {"status": "running"}

    def test_cancelled_batch_closes_task_and_records_recovery_outbox(self):
        self.finalizer.complete(BatchExecutionOutcome(
            batch_id="batch", status="cancelled", outcomes=(),
            recovered_without_raw_results=False))

        self.tasks.transition.assert_called_once_with(
            "task", "cancelled", label="Recorded action batch cancelled")
        self.graph.record_node.assert_called_once()
        self.tasks.publish.assert_called_once()

    def test_unknown_external_outcome_is_never_converted_to_action_failure(self):
        self.tasks.step_batch.return_value = {
            "task_id": "task",
            "steps": [{"status": "succeeded"},
                      {"status": "abandoned_unknown"}],
        }

        self.finalizer.complete(BatchExecutionOutcome(
            batch_id="batch", status="failed", outcomes=(),
            recovered_without_raw_results=False))

        transition = self.tasks.transition.call_args
        self.assertEqual(transition.args, ("task", "failed"))
        self.assertEqual(
            transition.kwargs["error"],
            "external_action_outcome_unknown_acknowledged",
        )

    def test_terminal_or_missing_batches_have_no_side_effect(self):
        for batch, state in ((None, None), ({"task_id": "task"},
                                           {"status": "completed"})):
            with self.subTest(batch=batch, state=state):
                self.tasks.reset_mock()
                self.graph.reset_mock()
                self.tasks.step_batch.return_value = batch
                self.tasks.get.return_value = state
                self.finalizer.complete(BatchExecutionOutcome(
                    batch_id="batch", status="succeeded", outcomes=(),
                    recovered_without_raw_results=True))
                self.tasks.transition.assert_not_called()
                self.graph.record_node.assert_not_called()


if __name__ == "__main__":
    unittest.main()
