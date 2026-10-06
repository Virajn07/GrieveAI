"""Classical evaluation baselines required by the project plan."""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge

CATEGORY_KEYWORDS = {
    "Academics": ["syllabus", "faculty", "professor", "lecture", "attendance", "timetable", "class"],
    "Examinations": ["exam", "examination", "hall ticket", "revaluation", "marks", "datesheet", "paper"],
    "Fees_Accounts": ["fee", "refund", "scholarship", "payment", "amount", "receipt", "gateway"],
    "IT_Library": ["wifi", "wi-fi", "library", "book", "portal", "login", "software", "license", "network"],
    "Infrastructure": ["classroom", "hostel", "washroom", "sanitation", "projector", "wiring", "power", "maintenance"],
    "Transport": ["bus", "route", "driver", "transport", "overcrowded", "crowd", "delay", "overspeed"],
    "Canteen": ["canteen", "food", "menu", "meal", "price", "hygiene", "insects", "clean"],
}

SUBCATEGORY_KEYWORDS = {
    "syllabus_pace": ["syllabus", "pace", "fast", "rushed", "जल्दी"],
    "faculty_conduct": ["faculty", "professor", "rude", "behaviour", "conduct"],
    "timetable_conflict": ["timetable", "schedule", "clash", "same time"],
    "attendance_dispute": ["attendance", "present", "medical certificate"],
    "revaluation_request": ["revaluation", "re-evaluation"],
    "hall_ticket_issue": ["hall ticket", "seat number", "download hall ticket"],
    "exam_schedule_conflict": ["exam", "same day", "datesheet", "schedule conflict"],
    "marks_discrepancy": ["marks", "mismatch", "answer sheet", "internal", "external"],
    "fee_discrepancy": ["fee", "extra", "discrepancy", "receipt"],
    "refund_delay": ["refund", "pending", "caution deposit"],
    "scholarship_query": ["scholarship", "application status"],
    "payment_gateway_issue": ["gateway", "payment", "deducted", "transaction"],
    "portal_access": ["portal", "login", "log in", "password", "access"],
    "book_availability": ["book", "reference book", "available", "issue"],
    "wifi_network": ["wifi", "wi-fi", "network"],
    "software_license": ["software", "license", "licence", "lab"],
    "classroom_maintenance": ["classroom", "projector", "furniture", "repair"],
    "hostel_issue": ["hostel", "room", "water supply"],
    "sanitation": ["washroom", "clean", "sanitation", "safai"],
    "electrical_fault": ["wiring", "power cut", "electrical", "safety hazard"],
    "bus_routing": ["route", "stop", "cover", "changed"],
    "schedule_delay": ["late", "delay", "timing", "30 minutes"],
    "safety_concern": ["safety", "unsafe", "overspeed", "driver"],
    "overcrowding": ["crowd", "overcrowded", "stand", "peak hours"],
    "food_quality": ["food quality", "undercooked", "cold", "stale"],
    "hygiene": ["hygiene", "clean", "insects", "dirty"],
    "pricing": ["price", "pricing", "expensive", "cost"],
    "menu_variety": ["menu", "variety", "options", "dietary"],
}


class KeywordRuleClassifier:
    def __init__(self, taxonomy):
        self.taxonomy = taxonomy

    def predict(self, text):
        t = text.lower()
        cat_scores = {cat: sum(1 for kw in kws if kw.lower() in t) for cat, kws in CATEGORY_KEYWORDS.items()}
        category = max(cat_scores, key=cat_scores.get)
        valid_subs = self.taxonomy["categories"][category]["subcategories"]
        sub_scores = {sub: sum(1 for kw in SUBCATEGORY_KEYWORDS.get(sub, []) if kw.lower() in t) for sub in valid_subs}
        subcategory = max(sub_scores, key=sub_scores.get)
        return category, subcategory


class TfidfSVMClassifier:
    def __init__(self, taxonomy):
        self.taxonomy = taxonomy
        self.vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 3), sublinear_tf=True)
        self.category_clf = LinearSVC(C=1.5)
        self.subcategory_clf = LinearSVC(C=1.5)
        self.priority_reg = Ridge(alpha=1.0)

    def fit(self, texts, categories, subcategories, priorities):
        X = self.vectorizer.fit_transform(texts)
        self.category_clf.fit(X, categories)
        self.subcategory_clf.fit(X, subcategories)
        self.priority_reg.fit(X, np.asarray(priorities, dtype=float))
        return self

    def predict(self, text):
        X = self.vectorizer.transform([text])
        category = self.category_clf.predict(X)[0]
        valid_subs = self.taxonomy["categories"][category]["subcategories"]
        decision = self.subcategory_clf.decision_function(X)[0]
        classes = list(self.subcategory_clf.classes_)
        valid_pairs = [(c, float(decision[i])) for i, c in enumerate(classes) if c in valid_subs]
        subcategory = max(valid_pairs, key=lambda x: x[1])[0]
        priority = float(self.priority_reg.predict(X)[0])
        return category, subcategory, priority
