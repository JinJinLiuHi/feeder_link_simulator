"""Model numbering, legacy databases and transactional deletion of model pairs."""

from pathlib import Path
import tempfile
import unittest
import numpy as np

from link_sim.algorithm.device_models import from_pa
from link_sim.algorithm.model_store import ModelStore
from link_sim.algorithm.pa_simulation_core import PAModel
from link_sim.algorithm.pa_dpd_core import signature


def pair():
    pa = PAModel([[2.]], "memPoly", 1, sample_rate=100e6)
    dpd = PAModel([[.5]], "memPoly", 1, sample_rate=200e6, role="predistorter",
                  fit_info=dict(gainReal=1., gainImag=0., paSignature=signature(pa)))
    return from_pa(pa, "功放"), from_pa(dpd, "预失真")


class ModelStoreDeletionTests(unittest.TestCase):
    def test_existing_ids_survive_upgrade_and_deleted_numbers_are_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"models.sqlite3"
            store = ModelStore(path)
            pa, _ = pair()
            first, second = store.save(pa), store.save(pa)
            with store.connection() as db:
                db.execute("DROP TABLE model_id_counter")  # Existing v1 database.
            store = ModelStore(path)
            np.testing.assert_array_equal(store.get(first).arrays["coefficients"], pa.arrays["coefficients"])
            store.delete(second)
            third = store.save(pa)
            self.assertGreater(third, second)
            store.delete(first)
            store.delete(third)
            store = ModelStore(path)
            self.assertGreater(store.save(pa), third)

    def test_delete_removes_arrays_only_for_selected_record_and_keeps_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ModelStore(Path(folder)/"models.sqlite3")
            pa, _ = pair()
            first, second = store.save(pa), store.save(pa)
            snapshot = store.get(first)
            store.delete(first)
            self.assertEqual([row["id"] for row in store.list()], [second])
            np.testing.assert_array_equal(snapshot.arrays["coefficients"], pa.arrays["coefficients"])
            with store.connection() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM model_arrays WHERE model_id=?", (first,)).fetchone()[0], 0)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM model_arrays WHERE model_id=?", (second,)).fetchone()[0], 1)
            with self.assertRaisesRegex(ValueError, "不存在"):
                store.delete(first)

    def test_paired_pa_is_protected_until_dpd_deleted_and_rollback_preserves_data(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ModelStore(Path(folder)/"models.sqlite3")
            pa, dpd = pair()
            dpd_id = store.save(dpd, paired_pa=pa)
            pa_id = store.list("pa")[0]["id"]
            with self.assertRaisesRegex(ValueError, f"#{dpd_id}.*预失真"):
                store.delete(pa_id)
            self.assertEqual(len(store.list()), 2)
            self.assertEqual(store.paired_power_amplifier(dpd_id).name, pa.name)
            store.delete(dpd_id)
            with store.connection() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM dpd_pa_pairs").fetchone()[0], 0)
            self.assertEqual(store.get(pa_id).name, pa.name)
            store.delete(pa_id)
            self.assertEqual(store.list(), [])

    def test_legacy_dpd_pair_preserves_available_matching_pa(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ModelStore(Path(folder)/"models.sqlite3")
            pa, dpd = pair()
            first, second = store.save(pa), store.save(pa)
            dpd_id = store.save(dpd)
            with store.connection() as db:
                db.execute("DELETE FROM dpd_pa_pairs")
            store.delete(second)
            self.assertIsNotNone(store.paired_power_amplifier(dpd_id))
            with self.assertRaisesRegex(ValueError, "DPD"):
                store.delete(first)
            store.delete(dpd_id)
            store.delete(first)


if __name__ == "__main__":
    unittest.main()
