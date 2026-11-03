#!/usr/bin/env python3
"""
14_purge_ghost_and_reaggregate.py — Clean up Phase-1 ghost artifacts and re-aggregate.

Phase-2 bug fixes F-4 (and prerequisite to F-1/F-2/F-3 reruns):
* Deletes the stale OpenFlamingo JSON skeleton that's polluting master CSVs.
* Deletes any other measurement JSONs whose hf_id is NOT in the current
  MODELS roster (defensive — catches manually-created stale files).
* Re-runs scripts 06 / 07 / 08 to rebuild master.csv, master_wide.csv,
  within_family_consistency.csv, family_summary.csv, finding_evidence.{csv,md}
  from the cleaned set.

All deletions are logged (file name + sha256 of contents before deletion in
case the user wants to restore). All operations are wrapped in try/except;
errors are logged with full traceback and the script attempts to continue.

Usage
-----
    python 14_purge_ghost_and_reaggregate.py            # delete + re-aggregate
    python 14_purge_ghost_and_reaggregate.py --dry-run  # just report
    python 14_purge_ghost_and_reaggregate.py --no-reaggregate  # delete only
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vlm_audit import MODELS, ExperimentLogger
from vlm_audit.logging_utils import sanitize_for_filename


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", default="outputs/measurements")
    p.add_argument("--log-dir", default="outputs/logs")
    p.add_argument("--dry-run", action="store_true", help="Don't delete anything; just report what would happen")
    p.add_argument("--no-reaggregate", action="store_true", help="Skip the post-deletion aggregation pass")
    p.add_argument("--keep-backup", action="store_true",
                   help="Move ghost JSONs to outputs/measurements/_purged/ instead of deleting outright")
    return p.parse_args()


def _sha256(path: Path, max_bytes: int = 1_000_000) -> str:
    """Hash a file up to ``max_bytes`` so we have a fingerprint on the log."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()
    except Exception:
        return "<unreadable>"


def main():
    args = parse_args()
    measurement_dir = Path(args.measurement_dir)

    with ExperimentLogger(args.log_dir, "14_purge_ghost_and_reaggregate") as log:
        log.section("Configuration")
        log.info(f"measurement_dir: {measurement_dir.resolve()}")
        log.info(f"dry_run: {args.dry_run}")
        log.info(f"no_reaggregate: {args.no_reaggregate}")
        log.info(f"keep_backup: {args.keep_backup}")

        log.section("Building valid hf_id set from MODELS")
        valid_filenames = {f"{sanitize_for_filename(m.hf_id)}.json" for m in MODELS}
        for m in MODELS:
            log.info(f"  KEEP  {m.name:30s}  -> {sanitize_for_filename(m.hf_id)}.json")
        log.info(f"Total valid JSON filenames in current roster: {len(valid_filenames)}")

        if not measurement_dir.exists():
            log.error(f"measurement_dir does not exist: {measurement_dir}")
            print(f"ERROR: measurement_dir does not exist: {measurement_dir}", file=sys.stderr)
            sys.exit(1)

        log.section("Scanning measurement_dir for ghost JSONs")
        all_jsons = sorted(measurement_dir.glob("*.json"))
        log.info(f"Found {len(all_jsons)} .json files total")

        ghosts = []
        valid = []
        for p in all_jsons:
            if p.name in valid_filenames:
                valid.append(p)
                continue
            # CSV outputs and meta files (within_family_*.csv etc.) are .csv not
            # .json so they don't appear here. Anything that is .json and not in
            # the valid set is a ghost (orphaned model JSON).
            ghosts.append(p)

        log.info(f"Valid JSONs:  {len(valid)}")
        log.info(f"Ghost JSONs:  {len(ghosts)}")
        for g in ghosts:
            sha = _sha256(g)
            size = g.stat().st_size
            log.iteration(f"ghost {g.name}", size_bytes=size, sha256_prefix=sha[:16])

        if not ghosts:
            log.info("No ghosts found — measurement_dir is clean.")

        # --- Delete (or back up) ghosts ---
        log.section("Deletion phase")
        backup_dir = measurement_dir / "_purged"
        deleted = []
        for g in ghosts:
            if args.dry_run:
                log.info(f"[DRY-RUN] would delete {g.name}")
                continue
            try:
                if args.keep_backup:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    dst = backup_dir / g.name
                    g.rename(dst)
                    log.info(f"Backed up {g.name} -> {dst}")
                else:
                    g.unlink()
                    log.info(f"Deleted {g.name}")
                deleted.append(g.name)
            except Exception as e:
                log.error(f"Could not delete {g.name}", e)

        # --- Re-aggregate ---
        if args.no_reaggregate or args.dry_run:
            log.info("Skipping re-aggregation per flag.")
        else:
            log.section("Re-running scripts 06 -> 07 -> 08")
            scripts_to_rerun = [
                "06_aggregate_master_table.py",
                "07_within_family_consistency.py",
                "08_finding_evidence_revision.py",
            ]
            here = Path(__file__).parent
            for script_name in scripts_to_rerun:
                script_path = here / script_name
                if not script_path.exists():
                    log.warn(f"Script not found, skipping: {script_path}")
                    continue
                log.iteration(f"running {script_name}")
                try:
                    res = subprocess.run(
                        [sys.executable, str(script_path)],
                        cwd=str(here),
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    log.info(f"{script_name} exit={res.returncode}")
                    # Echo stdout/stderr into the log so the diagnostic trail is complete
                    if res.stdout:
                        for line in res.stdout.splitlines():
                            log.info(f"  [stdout] {line}")
                    if res.stderr:
                        for line in res.stderr.splitlines():
                            log.warn(f"  [stderr] {line}")
                    if res.returncode != 0:
                        log.error(f"{script_name} returned non-zero exit code")
                except subprocess.TimeoutExpired as e:
                    log.error(f"{script_name} timed out after 5 minutes", e)
                except Exception as e:
                    log.error(f"Could not run {script_name}", e)

        log.section("Summary")
        summary = {
            "ghosts_found": [g.name for g in ghosts],
            "deleted": deleted,
            "dry_run": args.dry_run,
            "kept_backup": args.keep_backup,
        }
        log.info(json.dumps(summary, indent=2))

    # Also write a machine-readable summary so downstream scripts can read what happened.
    summary_path = measurement_dir / "_purge_summary.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written: {summary_path}")
    except Exception as e:
        print(f"Could not write summary: {e}", file=sys.stderr)

    print(f"Ghosts found:  {len(ghosts)}")
    print(f"Deleted:       {len(deleted)}")
    print(f"Re-aggregated: {not (args.no_reaggregate or args.dry_run)}")


if __name__ == "__main__":
    main()
