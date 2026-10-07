"""Seed the local database with deterministic, clearly synthetic demo cases.

Run ``python -m app.seed_demo`` to add the demo set once, or pass ``--reset``
to replace only records whose acknowledgement starts with ``DEMO-``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import random
from pathlib import Path

from app.app import create_app
from app.models_db import AuditLog, Grievance, db


ROOT = Path(__file__).resolve().parents[1]
DEMO_PREFIX = "DEMO-"
SEED = 4132026
DEMO_RECORDS = 297  # 9 examples for each of the taxonomy's 33 subcategories.
RECURRING_GROUPS = 18
HUMAN_REVIEW_COUNT = 36
CROSS_DOMAIN_COUNT = 50

CONTEXT = {
    "en": [
        "This is also affecting access to scheduled classes and coursework.",
        "The disruption is creating a safety concern for students returning after class.",
        "It is also delaying access to the lab resources needed for assignments.",
        "Students who rely on the bus service are missing their next scheduled lecture.",
    ],
    "hinglish": [
        "Is wajah se scheduled classes aur coursework bhi affect ho rahe hain.",
        "Class ke baad students ke liye wapas jaana safety concern ban raha hai.",
        "Assignments ke liye lab resources tak access bhi delay ho raha hai.",
        "Bus service par depend karne wale students ki next lecture miss ho rahi hai.",
    ],
    "hi": [
        "इससे निर्धारित कक्षाओं और पढ़ाई पर भी असर पड़ रहा है।",
        "कक्षा के बाद लौटने वाले विद्यार्थियों के लिए सुरक्षा चिंता बन रही है।",
        "असाइनमेंट के लिए प्रयोगशाला संसाधनों तक पहुँच में भी देरी हो रही है।",
        "बस सेवा पर निर्भर विद्यार्थियों की अगली कक्षा छूट रही है।",
    ],
}

NEAR_DUPLICATE_SUFFIX = {
    "en": " We reported the same issue earlier, and it has continued this week.",
    "hinglish": " Same issue pehle bhi report kiya tha, par is week bhi continue ho raha hai.",
    "hi": " यही समस्या पहले भी बताई थी, लेकिन इस सप्ताह भी जारी है।",
}


def _read_taxonomy() -> dict:
    return json.loads((ROOT / "config" / "taxonomy.json").read_text(encoding="utf-8"))["categories"]


def _source_rows(taxonomy: dict) -> list[dict]:
    path = ROOT / "data" / "processed" / "grievances_synthetic.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(SEED)
    selected = []
    for category, definition in taxonomy.items():
        for subcategory in definition["subcategories"]:
            eligible = [row for row in rows if row["category"] == category and row["subcategory"] == subcategory]
            by_language = {language: [row for row in eligible if row["language"] == language] for language in ("en", "hinglish", "hi")}
            if any(len(by_language[language]) < 3 for language in by_language):
                raise RuntimeError(f"Not enough synthetic source examples for {category}/{subcategory} in all three languages.")
            for language in ("en", "hinglish", "hi"):
                choices = sorted(by_language[language], key=lambda row: int(row["id"]))
                rng.shuffle(choices)
                selected.extend(dict(row) for row in choices[:3])
    if len(selected) != DEMO_RECORDS:
        raise RuntimeError(f"Expected {DEMO_RECORDS} source rows, found {len(selected)}.")
    return selected


def _build_records(taxonomy: dict) -> list[dict]:
    records = _source_rows(taxonomy)
    rng = random.Random(SEED + 1)

    # Eighteen three-record groups make duplicate and recurring signals explicit
    # and reproducible while retaining hundreds of distinct synthetic reports.
    category_blocks = [round(i * 32 / (RECURRING_GROUPS - 1)) for i in range(RECURRING_GROUPS)]
    group_starts = [block * 9 for block in category_blocks]
    for group_number, start in enumerate(group_starts, 1):
        base = records[start]
        language = base["language"]
        theme = f"{DEMO_PREFIX}CLUSTER-{group_number:02d}"
        records[start]["recurring_cluster_id"] = theme
        records[start + 1]["text"] = base["text"]
        records[start + 1]["recurring_cluster_id"] = theme
        records[start + 1]["duplicate_kind"] = "exact_duplicate"
        records[start + 1]["duplicate_base_index"] = start
        records[start + 2]["text"] = base["text"] + NEAR_DUPLICATE_SUFFIX[language]
        records[start + 2]["recurring_cluster_id"] = theme
        records[start + 2]["duplicate_kind"] = "near_duplicate"
        records[start + 2]["duplicate_base_index"] = start
        records[start + 2]["language"] = language

    # Cross-domain reports name a second, concrete campus impact in the same
    # language, without changing the primary taxonomy label.
    grouped = {n for start in group_starts for n in (start, start + 1, start + 2)}
    candidates = [i for i in range(len(records)) if i not in grouped]
    for index in rng.sample(candidates, CROSS_DOMAIN_COUNT):
        record = records[index]
        record["text"] = f"{record['text'].rstrip()} {rng.choice(CONTEXT[record['language']])}"
        record["cross_domain"] = True

    return records


def seed_demo(reset: bool = False) -> int:
    app = create_app()
    with app.app_context():
        db.create_all()
        existing_query = Grievance.query.filter(Grievance.ack_number.startswith(DEMO_PREFIX))
        existing_count = existing_query.count()
        if existing_count and not reset:
            print(f"Demo data already exists ({existing_count} records); no changes made. Use --reset to replace it.")
            return existing_count

        if reset:
            ids = [row_id for (row_id,) in db.session.query(Grievance.id).filter(
                Grievance.ack_number.startswith(DEMO_PREFIX)
            ).all()]
            if ids:
                AuditLog.query.filter(AuditLog.grievance_id.in_(ids)).delete(synchronize_session=False)
                Grievance.query.filter(Grievance.id.in_(ids)).delete(synchronize_session=False)
                db.session.commit()

        taxonomy = _read_taxonomy()
        records = _build_records(taxonomy)
        departments = json.loads((ROOT / "config" / "departments.json").read_text(encoding="utf-8"))["mapping"]
        sla = json.loads((ROOT / "config" / "sla_rules.json").read_text(encoding="utf-8"))["priority_hours"]
        now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
        date_tag = now.strftime("%Y%m%d")
        rng = random.Random(SEED + 2)
        statuses = ("routed", "in_progress", "resolved", "rejected", "closed", "submitted")
        created = []
        by_ack = {}

        for index, source in enumerate(records, 1):
            category = source["category"]
            subcategory = source["subcategory"]
            language = source["language"]
            priority = int(source["priority"])
            submitted_at = now - timedelta(days=(index * 17 + SEED) % 46, hours=(index * 7) % 24)
            manual_review = index <= HUMAN_REVIEW_COUNT
            status = "submitted" if manual_review else statuses[(index - HUMAN_REVIEW_COUNT - 1) % len(statuses)]
            department = departments.get(category)
            if manual_review or status == "submitted":
                department = None
            confidence = round(rng.uniform(0.50, 0.69), 3) if manual_review else round(rng.uniform(0.72, 0.99), 3)
            ack = f"{DEMO_PREFIX}{date_tag}-{index:03d}"
            resolved_at = submitted_at + timedelta(days=1 + (index % 8)) if status in {"resolved", "closed"} else None
            row = Grievance(
                ack_number=ack,
                text=source["text"],
                language=language,
                script="devanagari" if language == "hi" else "latin",
                language_confidence=0.99,
                category=category,
                subcategory=subcategory,
                priority=priority,
                predicted_category=category,
                predicted_subcategory=subcategory,
                predicted_priority=priority,
                priority_raw=float(priority),
                confidence=confidence,
                confidence_threshold=0.70,
                subcategory_confidence=round(max(0.45, confidence - 0.05), 3),
                model_version="synthetic_demo_seed_v1",
                status=status,
                routed_department=department,
                model_department=departments.get(category),
                manual_review=manual_review,
                automated_route=bool(department and not manual_review),
                recurring_cluster_id=source.get("recurring_cluster_id"),
                duplicate_similarity=(1.0 if source.get("duplicate_kind") == "exact_duplicate" else 0.91) if source.get("duplicate_kind") else None,
                duplicate_method=source.get("duplicate_kind"),
                llm_summary=f"Synthetic demo summary for {category} / {subcategory}.",
                llm_analysis={
                    "summary": f"Synthetic demo summary for {category} / {subcategory}.",
                    "root_cause": "Synthetic training example; cause has not been verified.",
                    "recommended_action": "Review the report and confirm details with the relevant campus team.",
                    "department_recommendation": departments.get(category),
                    "provider": "synthetic_demo_seed",
                    "used_fallback": True,
                    "recurring_interpretation": "Synthetic recurring group for dashboard demonstration." if source.get("recurring_cluster_id") else "No seeded recurring group.",
                },
                summary_provider="synthetic_demo_seed",
                submitted_at=submitted_at,
                prediction_at=submitted_at,
                routing_at=submitted_at if department and not manual_review else None,
                manual_review_at=submitted_at if manual_review else None,
                resolved_at=resolved_at,
                sla_deadline=submitted_at + timedelta(hours=int(sla.get(str(priority), 72))),
            )
            db.session.add(row)
            by_ack[ack] = row
            created.append((row, source))

        db.session.flush()
        for index, (row, source) in enumerate(created, 1):
            if source.get("duplicate_kind"):
                group_start = int(source["duplicate_base_index"]) + 1
                base_ack = f"{DEMO_PREFIX}{date_tag}-{group_start:03d}"
                base = by_ack[base_ack]
                row.duplicate_of_id = base.id
                row.related_matches = [{
                    "grievance_id": base.id,
                    "similarity": row.duplicate_similarity,
                    "relationship": source["duplicate_kind"],
                    "method": "synthetic_demo_seed",
                }]
            else:
                row.related_matches = []
            db.session.add(AuditLog(
                grievance_id=row.id,
                action="submitted",
                actor="synthetic_demo_seed",
                detail="Synthetic local demo case; not a real student submission.",
                timestamp=row.submitted_at,
            ))
            if row.manual_review:
                db.session.add(AuditLog(
                    grievance_id=row.id,
                    action="manual_review_queued",
                    actor="synthetic_demo_seed",
                    detail="Synthetic review-queue example for local dashboard demonstration.",
                    timestamp=row.submitted_at,
                ))
            elif row.routed_department:
                db.session.add(AuditLog(
                    grievance_id=row.id,
                    action="routed",
                    actor="synthetic_demo_seed",
                    detail=f"Synthetic demo routing to {row.routed_department}.",
                    timestamp=row.submitted_at,
                ))
            if row.status in {"resolved", "closed"}:
                db.session.add(AuditLog(
                    grievance_id=row.id,
                    action="status_updated",
                    actor="synthetic_demo_seed",
                    detail=f"Synthetic demo case marked {row.status}.",
                    timestamp=row.resolved_at or row.submitted_at,
                ))
        db.session.commit()
        print(f"Seeded {len(created)} synthetic demo grievances; {RECURRING_GROUPS} recurring groups, {2 * RECURRING_GROUPS} duplicate links, {HUMAN_REVIEW_COUNT} queued for human review.")
        return len(created)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed clearly synthetic local GrieveAI demo records.")
    parser.add_argument("--reset", action="store_true", help="replace only existing DEMO-* synthetic rows")
    args = parser.parse_args()
    seed_demo(reset=args.reset)


if __name__ == "__main__":
    main()
