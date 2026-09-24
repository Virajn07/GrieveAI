"""
GrieveAI - Synthetic Grievance Generator
==========================================
Bootstraps a labeled training set from the taxonomy so the ML pipeline
(language ID -> classifier -> priority -> dedup -> SHAP) can be built and
tested end-to-end WITHOUT waiting on real VCET data access.

When real data arrives: replace/augment rows in the output CSV. No code
elsewhere in the pipeline needs to change - it just reads a CSV with the
same column schema.

Run:  python generate_synthetic_grievances.py --per_combo 15
Output: grievances_synthetic.csv
"""

import argparse
import csv
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Placeholder value pools (used to add lexical variety to templates)
# ---------------------------------------------------------------------------
PLACEHOLDERS = {
    "building": ["Block A", "Block B", "the main building", "the new wing", "Block C", "the library building"],
    "course": ["Data Structures", "AI/ML", "DBMS", "Mumbai University semester exam", "the elective course", "Computer Networks"],
    "days": ["3 days", "a week", "10 days", "two weeks", "5 days"],
    "amount": ["Rs. 2,500", "Rs. 5,000", "Rs. 1,200", "Rs. 8,000", "Rs. 500"],
    "faculty": ["the faculty", "a professor", "the HOD", "a visiting lecturer"],
    "route": ["Route 4", "Route 7", "the Vasai route", "Route 2"],
    "floor": ["2nd floor", "3rd floor", "ground floor", "4th floor"],
}

def fill(template):
    out = template
    for key, values in PLACEHOLDERS.items():
        if "{" + key + "}" in out:
            out = out.replace("{" + key + "}", random.choice(values))
    return out

# ---------------------------------------------------------------------------
# Templates: TEMPLATES[category][subcategory][lang] = list of template strings
# lang in {"en", "hi", "hinglish"}
# ---------------------------------------------------------------------------
TEMPLATES = {
 "Academics": {
  "syllabus_pace": {
   "en": ["The {course} syllabus is being rushed and we are not able to understand the concepts.",
          "Faculty is covering {course} too fast, students are struggling to keep up."],
   "hi": ["{course} का सिलेबस बहुत जल्दी पढ़ाया जा रहा है, समझ नहीं आ रहा।",
          "पढ़ाई की गति बहुत तेज़ है, कृपया धीमा करें।"],
   "hinglish": ["{course} ka syllabus bohot jaldi cover ho raha hai, samajh nahi aa raha.",
                "Pace bahut fast hai sir, thoda slow karo please."]
  },
  "faculty_conduct": {
   "en": ["{faculty} was rude to students during the {course} lecture.",
          "There was inappropriate behaviour by {faculty} in class today."],
   "hi": ["{faculty} ने क्लास में छात्रों के साथ बहुत बुरा व्यवहार किया।",
          "आज क्लास में {faculty} का व्यवहार सही नहीं था।"],
   "hinglish": ["{faculty} ne class mein bohot rudely behave kiya students ke saath.",
                "Aaj {faculty} ka behaviour theek nahi tha class mein."]
  },
  "timetable_conflict": {
   "en": ["Two lectures for {course} and another subject are scheduled at the same time.",
          "The revised timetable clashes with our lab sessions."],
   "hi": ["{course} की क्लास और लैब का समय एक साथ है, टकराव हो रहा है।",
          "नया टाइमटेबल पुराने शेड्यूल से टकरा रहा है।"],
   "hinglish": ["{course} ka lecture aur lab dono same time pe hai, clash ho raha hai.",
                "New timetable purane schedule se clash kar raha hai."]
  },
  "attendance_dispute": {
   "en": ["My attendance for {course} is marked wrong despite being present.",
          "Attendance was not updated even after submitting a medical certificate."],
   "hi": ["{course} में मेरी अटेंडेंस गलत दिखाई जा रही है जबकि मैं क्लास में था।",
          "मेडिकल सर्टिफिकेट देने के बाद भी अटेंडेंस अपडेट नहीं हुई।"],
   "hinglish": ["{course} mein meri attendance galat show ho rahi hai, main present tha.",
                "Medical certificate submit karne ke baad bhi attendance update nahi hui."]
  }
 },
 "Examinations": {
  "revaluation_request": {
   "en": ["I want to request revaluation for my {course} paper, marks seem incorrect.",
          "Please process my revaluation application for the {course} exam, it has been {days}."],
   "hi": ["मुझे {course} पेपर का रीवैल्यूएशन चाहिए, मार्क्स गलत लग रहे हैं।",
          "रीवैल्यूएशन एप्लीकेशन दिए हुए {days} हो गए, कृपया प्रोसेस करें।"],
   "hinglish": ["{course} paper ka revaluation chahiye, marks galat lag rahe hain.",
                "Revaluation application diye {days} ho gaye, please process karo."]
  },
  "hall_ticket_issue": {
   "en": ["My hall ticket for the {course} exam has an incorrect seat number.",
          "I am unable to download my hall ticket from the portal."],
   "hi": ["मेरे हॉल टिकट में सीट नंबर गलत है {course} परीक्षा के लिए।",
          "पोर्टल से हॉल टिकट डाउनलोड नहीं हो रहा है।"],
   "hinglish": ["Mera hall ticket mein seat number galat hai {course} exam ke liye.",
                "Portal se hall ticket download nahi ho raha hai."]
  },
  "exam_schedule_conflict": {
   "en": ["Two of my exams including {course} are scheduled on the same day.",
          "The exam datesheet clashes with another subject's exam."],
   "hi": ["मेरी दो परीक्षाएं {course} सहित एक ही दिन पर हैं।",
          "परीक्षा डेटशीट दूसरे विषय की परीक्षा से टकरा रही है।"],
   "hinglish": ["Meri do exams including {course} same din pe hain.",
                "Exam datesheet doosre subject ke exam se clash kar raha hai."]
  },
  "marks_discrepancy": {
   "en": ["The marks uploaded for {course} do not match what was shown on the answer sheet.",
          "There is a mismatch between internal and external marks for {course}."],
   "hi": ["{course} के लिए अपलोड किए गए मार्क्स आंसर शीट से मेल नहीं खाते।",
          "{course} के इंटरनल और एक्सटर्नल मार्क्स में अंतर है।"],
   "hinglish": ["{course} ke marks jo upload hue hain wo answer sheet se match nahi karte.",
                "{course} ke internal aur external marks mein mismatch hai."]
  }
 },
 "Fees_Accounts": {
  "fee_discrepancy": {
   "en": ["I was charged {amount} extra in my semester fee without explanation.",
          "There is a discrepancy of {amount} between the fee receipt and portal amount."],
   "hi": ["मुझसे सेमेस्टर फीस में {amount} अतिरिक्त लिया गया है बिना किसी कारण के।",
          "फीस रसीद और पोर्टल राशि में {amount} का अंतर है।"],
   "hinglish": ["Mujhse semester fee mein {amount} extra charge hua hai bina reason ke.",
                "Fee receipt aur portal amount mein {amount} ka difference hai."]
  },
  "refund_delay": {
   "en": ["My refund of {amount} has been pending for {days}.",
          "I still have not received my caution deposit refund after {days}."],
   "hi": ["मेरा {amount} का रिफंड {days} से पेंडिंग है।",
          "{days} बाद भी मेरा कॉशन डिपॉजिट रिफंड नहीं मिला।"],
   "hinglish": ["Mera {amount} ka refund {days} se pending hai.",
                "{days} baad bhi caution deposit refund nahi mila."]
  },
  "scholarship_query": {
   "en": ["My scholarship status has not been updated on the portal for {days}.",
          "I have not received any update about my scholarship application."],
   "hi": ["मेरी स्कॉलरशिप स्थिति पोर्टल पर {days} से अपडेट नहीं हुई है।",
          "मुझे स्कॉलरशिप एप्लीकेशन के बारे में कोई अपडेट नहीं मिला।"],
   "hinglish": ["Meri scholarship status portal pe {days} se update nahi hui hai.",
                "Scholarship application ke baare mein koi update nahi mila."]
  },
  "payment_gateway_issue": {
   "en": ["The fee payment gateway failed but {amount} was deducted from my account.",
          "I am unable to complete fee payment due to a gateway error."],
   "hi": ["फीस पेमेंट गेटवे फेल हो गया लेकिन {amount} मेरे अकाउंट से कट गया।",
          "गेटवे एरर की वजह से फीस पेमेंट पूरा नहीं हो पा रहा है।"],
   "hinglish": ["Fee payment gateway fail ho gaya lekin {amount} account se kat gaya.",
                "Gateway error ki wajah se fee payment complete nahi ho raha."]
  }
 },
 "IT_Library": {
  "portal_access": {
   "en": ["I am unable to log into the student portal for {days}.",
          "The portal shows an error every time I try to access my results."],
   "hi": ["मैं {days} से स्टूडेंट पोर्टल में लॉगिन नहीं कर पा रहा हूँ।",
          "रिजल्ट देखने की कोशिश करने पर पोर्टल एरर दिखाता है।"],
   "hinglish": ["Main {days} se student portal mein login nahi kar pa raha.",
                "Result dekhne ki koshish karne pe portal error deta hai."]
  },
  "book_availability": {
   "en": ["The reference book for {course} is not available in the library.",
          "I have been waiting {days} for a library book to be issued."],
   "hi": ["{course} के लिए संदर्भ पुस्तक लाइब्रेरी में उपलब्ध नहीं है।",
          "लाइब्रेरी की किताब मिलने में {days} हो गए।"],
   "hinglish": ["{course} ki reference book library mein available nahi hai.",
                "Library book issue hone mein {days} ho gaye."]
  },
  "wifi_network": {
   "en": ["Wi-Fi in {building} has not been working for {days}.",
          "The internet connection in the {floor} lab keeps disconnecting."],
   "hi": ["{building} में वाईफाई {days} से काम नहीं कर रहा है।",
          "{floor} लैब में इंटरनेट बार-बार डिस्कनेक्ट हो रहा है।"],
   "hinglish": ["{building} mein WiFi {days} se kaam nahi kar raha hai.",
                "{floor} lab mein internet baar baar disconnect ho raha hai."]
  },
  "software_license": {
   "en": ["The licensed software for {course} lab has expired and needs renewal.",
          "We do not have access to the required software license in the lab."],
   "hi": ["{course} लैब का लाइसेंस्ड सॉफ्टवेयर एक्सपायर हो गया है, रिन्यू चाहिए।",
          "लैब में जरूरी सॉफ्टवेयर लाइसेंस उपलब्ध नहीं है।"],
   "hinglish": ["{course} lab ka licensed software expire ho gaya hai, renew chahiye.",
                "Lab mein zaroori software license available nahi hai."]
  }
 },
 "Infrastructure": {
  "classroom_maintenance": {
   "en": ["The projector in {building} {floor} classroom has been broken for {days}.",
          "Furniture in the classroom is damaged and needs repair."],
   "hi": ["{building} {floor} की क्लास का प्रोजेक्टर {days} से खराब है।",
          "क्लासरूम का फर्नीचर खराब है, मरम्मत चाहिए।"],
   "hinglish": ["{building} {floor} classroom ka projector {days} se kharab hai.",
                "Classroom ka furniture damaged hai, repair chahiye."]
  },
  "hostel_issue": {
   "en": ["The hostel water supply has been irregular for {days}.",
          "Hostel room maintenance requests are not being addressed."],
   "hi": ["हॉस्टल में पानी की सप्लाई {days} से अनियमित है।",
          "हॉस्टल रूम मेंटेनेंस रिक्वेस्ट पर कोई कार्रवाई नहीं हुई।"],
   "hinglish": ["Hostel mein paani ki supply {days} se irregular hai.",
                "Hostel room maintenance request pe koi action nahi hua."]
  },
  "sanitation": {
   "en": ["Washrooms in {building} are not being cleaned regularly.",
          "There is a sanitation issue near the {floor} corridor."],
   "hi": ["{building} के वॉशरूम नियमित रूप से साफ नहीं किए जा रहे हैं।",
          "{floor} गलियारे के पास सफाई की समस्या है।"],
   "hinglish": ["{building} ke washroom regularly clean nahi ho rahe hain.",
                "{floor} corridor ke paas safai ki problem hai."]
  },
  "electrical_fault": {
   "en": ["There is exposed wiring near {building} {floor}, this is a safety hazard.",
          "Frequent power cuts in {building} are disrupting lab sessions."],
   "hi": ["{building} {floor} के पास खुली वायरिंग है, यह खतरनाक है।",
          "{building} में बार-बार बिजली कटौती से लैब सेशन बाधित हो रहे हैं।"],
   "hinglish": ["{building} {floor} ke paas exposed wiring hai, ye safety hazard hai.",
                "{building} mein baar baar power cut se lab session disrupt ho raha hai."]
  }
 },
 "Transport": {
  "bus_routing": {
   "en": ["{route} bus does not cover our stop anymore, please review the route.",
          "The college bus route was changed without prior notice."],
   "hi": ["{route} बस अब हमारे स्टॉप पर नहीं आती, कृपया रूट देखें।",
          "कॉलेज बस रूट बिना सूचना के बदल दिया गया।"],
   "hinglish": ["{route} bus ab hamare stop pe nahi aati, please route check karo.",
                "College bus route bina notice ke change kar diya gaya."]
  },
  "schedule_delay": {
   "en": ["The {route} bus has been consistently late by 30+ minutes for {days}.",
          "Bus timing is unreliable, we keep missing the first lecture."],
   "hi": ["{route} बस {days} से लगातार 30 मिनट से ज्यादा देर हो रही है।",
          "बस का समय अनिश्चित है, हमारी पहली क्लास छूट जाती है।"],
   "hinglish": ["{route} bus {days} se consistently 30+ minute late ho rahi hai.",
                "Bus timing unreliable hai, first lecture miss ho jata hai."]
  },
  "safety_concern": {
   "en": ["The {route} bus driver was overspeeding and it felt unsafe.",
          "There is no proper safety check being done on the college buses."],
   "hi": ["{route} बस का ड्राइवर तेज़ गति से चला रहा था, असुरक्षित महसूस हुआ।",
          "कॉलेज बसों की उचित सुरक्षा जांच नहीं हो रही है।"],
   "hinglish": ["{route} bus driver overspeed kar raha tha, unsafe laga.",
                "College buses ki proper safety check nahi ho rahi hai."]
  },
  "overcrowding": {
   "en": ["The {route} bus is severely overcrowded during peak hours.",
          "Students are forced to stand for the entire journey on {route}."],
   "hi": ["{route} बस पीक आवर्स में बहुत भीड़भाड़ वाली होती है।",
          "छात्रों को {route} पर पूरी यात्रा खड़े होकर करनी पड़ती है।"],
   "hinglish": ["{route} bus peak hours mein bohot overcrowded hoti hai.",
                "Students ko {route} pe poori journey khade hoke karni padti hai."]
  }
 },
 "Canteen": {
  "food_quality": {
   "en": ["The food quality in the canteen has dropped significantly this month.",
          "I found the food undercooked and cold at the canteen today."],
   "hi": ["कैंटीन में खाने की क्वालिटी इस महीने काफी गिर गई है।",
          "आज कैंटीन में खाना अधपका और ठंडा मिला।"],
   "hinglish": ["Canteen mein khane ki quality is mahine kaafi gir gayi hai.",
                "Aaj canteen mein khana undercooked aur thanda mila."]
  },
  "hygiene": {
   "en": ["The canteen seating area is not being cleaned properly.",
          "There were insects near the food counter at the canteen."],
   "hi": ["कैंटीन का बैठने का क्षेत्र ठीक से साफ नहीं किया जा रहा है।",
          "कैंटीन के फूड काउंटर के पास कीड़े दिखे।"],
   "hinglish": ["Canteen ka seating area properly clean nahi ho raha hai.",
                "Canteen ke food counter ke paas insects dikhe."]
  },
  "pricing": {
   "en": ["Canteen prices were increased without any notice to students.",
          "The price for a basic meal is too high compared to quality offered."],
   "hi": ["कैंटीन की कीमतें बिना किसी सूचना के बढ़ा दी गईं।",
          "बेसिक खाने की कीमत क्वालिटी के मुकाबले बहुत ज्यादा है।"],
   "hinglish": ["Canteen ki prices bina notice ke badha di gayi hain.",
                "Basic meal ki price quality ke comparison mein bohot zyada hai."]
  },
  "menu_variety": {
   "en": ["The canteen menu has not changed in months, we want more variety.",
          "There are very few options for students with dietary restrictions."],
   "hi": ["कैंटीन का मेनू महीनों से नहीं बदला, हमें और विविधता चाहिए।",
          "डाइटरी रिस्ट्रिक्शन वाले छात्रों के लिए बहुत कम विकल्प हैं।"],
   "hinglish": ["Canteen ka menu mahino se nahi badla, hume aur variety chahiye.",
                "Dietary restriction wale students ke liye bohot kam options hain."]
  }
 }
}

SAFETY_SUBCATS = {"safety_concern", "electrical_fault"}

def compute_priority(base_priority, subcategory):
    p = base_priority + random.choice([-1, 0, 0, 0, 1])
    if subcategory in SAFETY_SUBCATS:
        p += 1
    return max(1, min(5, p))

def main(per_combo, taxonomy, dup_rate, out_path):
    rows = []
    row_id = 1
    for category, meta in taxonomy["categories"].items():
        for subcat in meta["subcategories"]:
            for lang in taxonomy["languages"]:
                templates = TEMPLATES[category][subcat][lang]
                for _ in range(per_combo):
                    template = random.choice(templates)
                    text = fill(template)
                    priority = compute_priority(meta["base_priority"], subcat)
                    rows.append({
                        "id": row_id,
                        "text": text,
                        "language": lang,
                        "category": category,
                        "subcategory": subcat,
                        "priority": priority,
                        "duplicate_of": ""
                    })
                    row_id += 1

    # Inject near-duplicates for testing the dedup module
    n_dupes = int(len(rows) * dup_rate)
    dupe_targets = random.sample(rows, n_dupes)
    for target in dupe_targets:
        rows.append({
            "id": row_id,
            "text": target["text"],  # exact/near copy - real pipeline should catch this
            "language": target["language"],
            "category": target["category"],
            "subcategory": target["subcategory"],
            "priority": target["priority"],
            "duplicate_of": target["id"]
        })
        row_id += 1

    random.shuffle(rows)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "language", "category", "subcategory", "priority", "duplicate_of"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"  Base grievances: {row_id - 1 - n_dupes}")
    print(f"  Injected near-duplicates: {n_dupes}")
    print(f"  Categories: {len(taxonomy['categories'])}, Sub-categories: {sum(len(m['subcategories']) for m in taxonomy['categories'].values())}")

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_combo", type=int, default=15, help="rows per (subcategory x language) combo")
    parser.add_argument("--dup_rate", type=float, default=0.05, help="fraction of rows duplicated for dedup testing")
    parser.add_argument("--taxonomy", type=str, default="config/taxonomy.json")
    parser.add_argument("--out", type=str, default="data/processed/grievances_synthetic.csv")
    args = parser.parse_args()

    with open(args.taxonomy, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    main(args.per_combo, taxonomy, args.dup_rate, args.out)
