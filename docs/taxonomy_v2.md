# GrieveAI Working Taxonomy v2

**Status:** working taxonomy derived from the HOD-reviewed seven-category scope and project-owner clarification. It must be validated with the VCET grievance cell before a live pilot.

## Design rule

The classifier's primary category/subcategory should represent the **actionable institutional issue**. Cross-domain information should not create artificial subcategories. A later extraction/LLM layer should retain:

- root cause or contributing factor
- affected area
- related category/domain
- requested action
- recommended responsible department

Example: “The canteen is overcrowded because every department has the same lunch break.”

- Primary: `Academics / timetable_scheduling`
- Affected area: Canteen
- Contributing factor: common break timing
- Candidate action: stagger breaks

## Categories and subcategories

### Academics
- `timetable_scheduling`
- `teaching_quality`
- `faculty_conduct`
- `syllabus_course_progress`
- `attendance`

### Examinations
- `hall_ticket`
- `marks_discrepancy`
- `revaluation`
- `examination_schedule`
- `examination_process`

### Fees / Accounts
- `fee_discrepancy`
- `refund`
- `scholarship`
- `payment_transaction`
- `accounts_financial_documentation`

### IT / Library
- `portal_account_access`
- `wifi_network`
- `software_license`
- `library_book_availability`
- `library_digital_resources`

### Infrastructure
- `classroom_lab_maintenance`
- `electrical_issues`
- `sanitation_cleanliness`
- `parking_access`
- `construction_facility_disruption`

### Transport
Transport is retained because it is part of the HOD-reviewed seven-category scope, but it does **not** mean VCET operates buses or controls external trains. It covers student commuting/accessibility concerns that can have an institutional impact.

- `commuting_accessibility`
- `transport_academic_conflict`
- `travel_safety_access`

External bus/train delays are not themselves VCET transport grievances. When the actionable institutional issue is timetable design, the primary label can instead be `Academics / timetable_scheduling` with transportation retained as context.

### Canteen
- `food_quality`
- `food_hygiene`
- `pricing`
- `menu_variety`
- `food_service`

Canteen overcrowding is intentionally not a classifier subcategory. If overcrowding is caused by common break scheduling, the primary actionable issue is `Academics / timetable_scheduling`, with Canteen as the affected area.

## Dataset migration

The previous synthetic taxonomy had 28 subcategories. The revised synthetic seed has **33 subcategories** across the same seven top-level categories. The old synthetic CSV should not be mixed with the new taxonomy without relabeling.

A new MuRIL/LoRA training run is required before using the revised taxonomy for model evaluation.
