import csv
import unittest
import uuid
from pathlib import Path

from data.subject_metadata import (
    METADATA_FIELDS, MetadataValidationError, validate_metadata,
    assign_unique_subjects, sha256_file,
)


def workspace_dir(name):
    path = Path("tmp_tests_workspace") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


class SubjectMetadataTests(unittest.TestCase):
    def build_metadata(self):
        root = workspace_dir("subject_metadata")
        rows = []
        for label, folder in ((1, "positive"), (0, "negative")):
            (root / folder).mkdir()
            for index in range(3):
                relative = f"{folder}/video_{index}.mp4"
                (root / relative).write_bytes(f"{label}-{index}".encode("ascii"))
                rows.append(
                    {
                        "video_id": sha256_file(root / relative),
                        "relative_path": relative,
                        "label": str(label),
                        "subject_id": f"subject-{label}-{index}",
                        "session_id": f"session-{label}-{index}",
                        "capture_datetime": "",
                        "site_id": "",
                        "device_id": "",
                        "include": "1",
                        "notes": "",
                    }
                )
        metadata = root / "metadata.csv"
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return root, metadata, rows

    def test_complete_mapping_passes_subject_requirements(self):
        root, metadata, _rows = self.build_metadata()

        result = validate_metadata(root, metadata, verify_hashes=False)

        self.assertEqual(result["subjects_by_label"], {"0": 3, "1": 3})
        self.assertEqual(result["sessions"], 6)

    def test_owner_confirmed_unique_assignment_is_stable_and_preserves_backup(self):
        root, metadata, rows = self.build_metadata()
        for row in rows:
            row["subject_id"] = row["session_id"] = ""
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        before = metadata.read_bytes()
        result = assign_unique_subjects(root, metadata)
        assigned = metadata.read_bytes()
        assign_unique_subjects(root, metadata)
        self.assertEqual(result["subjects"], 6)
        self.assertEqual(result["sessions"], 6)
        self.assertEqual(metadata.read_bytes(), assigned)
        self.assertEqual(metadata.with_name("metadata.before_unique_subjects.csv").read_bytes(), before)
        self.assertTrue(metadata.with_suffix(".provenance.json").exists())

    def test_unique_assignment_refuses_to_overwrite_existing_mapping(self):
        root, metadata, _rows = self.build_metadata()
        before = metadata.read_bytes()
        with self.assertRaisesRegex(MetadataValidationError, "reconciliation"):
            assign_unique_subjects(root, metadata)
        self.assertEqual(metadata.read_bytes(), before)

    def test_subject_cannot_have_conflicting_labels(self):
        root, metadata, rows = self.build_metadata()
        rows[3]["subject_id"] = rows[0]["subject_id"]
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaisesRegex(MetadataValidationError, "inconsistent labels"):
            validate_metadata(root, metadata, verify_hashes=False)

    def test_session_cannot_belong_to_multiple_subjects(self):
        root, metadata, rows = self.build_metadata()
        rows[1]["session_id"] = rows[0]["session_id"]
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaisesRegex(MetadataValidationError, "exactly one subject"):
            validate_metadata(root, metadata, verify_hashes=False)


if __name__ == "__main__":
    unittest.main()
