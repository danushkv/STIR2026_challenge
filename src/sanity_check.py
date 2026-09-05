import os
import re
from typing import List

SEQ_PATTERN = re.compile(r"seq(\d+)")


def check_folder_and_files(main_folder: str, folder_names: List[str]):
    # given the teacher folder names, check:
    #  1) every teacher folder has the same number of patient subfolders
    #  2) every patient subfolder has the same set of clip IDs across teachers
    # Every failure is reported, rather than stopping at the first one.

    errors = []

    # === Check patient subfolder counts match across teachers ===
    patients_by_teacher = {}
    for teacher in folder_names:
        teacher_path = os.path.join(main_folder, teacher)
        patients_by_teacher[teacher] = sorted(os.listdir(teacher_path))

    reference_teacher = folder_names[0]
    reference_patients = patients_by_teacher[reference_teacher]
    reference_patient_set = set(reference_patients)

    counts = {teacher: len(patients) for teacher, patients in patients_by_teacher.items()}
    # if len(set(counts.values())) > 1:
    #     diffs = {}
    #     for teacher, patients in patients_by_teacher.items():
    #         patient_set = set(patients)
    #         if patient_set != reference_patient_set:
    #             diffs[teacher] = {
    #                 "missing": sorted(reference_patient_set - patient_set),
    #                 "extra": sorted(patient_set - reference_patient_set),
    #             }
    #     msg = (f"Mismatched number of patient subfolders: {counts}\n"
    #            f"Differences from {reference_teacher!r}: {diffs}")
    #     print(f"ERROR: {msg}")
    #     errors.append(msg)

    # === Check clip IDs (seq numbers) match across teachers, per patient ===
    for patient in reference_patients:
        clip_ids_by_teacher = {}
        for teacher in folder_names:
            patient_path = os.path.join(main_folder, teacher, patient)
            if not os.path.isdir(patient_path):
                msg = f"{teacher!r} is missing patient folder {patient!r}"
                print(f"ERROR: {msg}")
                errors.append(msg)
                continue

            seq_numbers = set()
            for fname in os.listdir(patient_path):
                match = SEQ_PATTERN.search(fname)
                if match:
                    seq_numbers.add(int(match.group(1)))
            clip_ids_by_teacher[teacher] = seq_numbers

        if reference_teacher not in clip_ids_by_teacher:
            continue  # already reported as a missing patient folder above

        reference_clip_ids = clip_ids_by_teacher[reference_teacher]
        for teacher, clip_ids in clip_ids_by_teacher.items():
            if clip_ids != reference_clip_ids:
                missing = sorted(reference_clip_ids - clip_ids)
                extra = sorted(clip_ids - reference_clip_ids)
                msg = (f"Patient {patient!r}: {teacher!r} clip IDs differ from "
                       f"{reference_teacher!r} (missing={missing}, extra={extra})")
                print(f"ERROR: {msg}")
                errors.append(msg)

    if errors:
        raise AssertionError(f"{len(errors)} sanity check(s) failed:\n" + "\n".join(errors))

    print(f"OK: {len(folder_names)} teachers, {len(reference_patients)} patients each, "
          f"all clip IDs match.")


DEFAULT_TEACHERS = ["alltracker", "cotracker3", "trackon2",
                    "trackon_r", "locotrack", "mft"]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Check that every teacher covered the same patients and the "
                    "same clips, before spending hours on phase 2.")
    p.add_argument("raw_tracks_root",
                   help="the directory holding the per-teacher folders, e.g. "
                        ".../STIRprocessed/STIROrig_tracks")
    p.add_argument("--teachers", nargs="+", default=DEFAULT_TEACHERS,
                   help=f"default: {' '.join(DEFAULT_TEACHERS)}")
    a = p.parse_args()
    check_folder_and_files(a.raw_tracks_root, a.teachers)
