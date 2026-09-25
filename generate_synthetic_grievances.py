"""
GrieveAI - Synthetic Grievance Generator
========================================
Bootstraps a labeled training set from the current seven-category VCET-oriented
working taxonomy so the ML pipeline can be developed before de-identified VCET
records are available.

Important: these are synthetic demo/training labels, not VCET-validated labels.
The taxonomy is intentionally based on actionable institutional issues. When a
complaint spans domains, the primary label should represent the actionable issue;
root cause, affected area, and other cross-domain context belong in later pipeline
metadata rather than being forced into extra classifier classes.
"""

import argparse
import csv
import random

random.seed(42)

PLACEHOLDERS = {
    "building": ["Block A", "Block B", "the main building", "the new wing", "Block C", "the library building"],
    "course": ["Data Structures", "AI/ML", "DBMS", "Computer Networks", "Operating Systems", "the elective course"],
    "days": ["3 days", "a week", "10 days", "two weeks", "5 days"],
    "amount": ["Rs. 2,500", "Rs. 5,000", "Rs. 1,200", "Rs. 8,000", "Rs. 500"],
    "faculty": ["the faculty", "a professor", "the HOD", "a visiting lecturer"],
    "floor": ["2nd floor", "3rd floor", "ground floor", "4th floor"],
    "location": ["Virar", "Nalasopara", "Palghar", "Boisar", "Thalasari", "Bhiwandi"],
}


def fill(template):
    out = template
    for key, values in PLACEHOLDERS.items():
        if "{" + key + "}" in out:
            out = out.replace("{" + key + "}", random.choice(values))
    return out


# TEMPLATES[category][subcategory][language] = list of templates
TEMPLATES = {
    "Academics": {
        "timetable_scheduling": {
            "en": [
                "Two lectures for {course} and another subject are scheduled at the same time.",
                "The common lunch break is causing heavy crowding at the canteen; please review the break timetable.",
            ],
            "hi": [
                "{course} की क्लास और लैब का समय एक साथ है, टकराव हो रहा है।",
                "सभी विभागों का लंच ब्रेक एक साथ होने से कैंटीन में बहुत भीड़ होती है, ब्रेक टाइम बदला जाए।",
            ],
            "hinglish": [
                "{course} ka lecture aur lab dono same time pe hai, clash ho raha hai.",
                "Sab departments ka lunch break same hai, canteen mein bohot crowd hota hai, break timing review karo.",
            ],
        },
        "teaching_quality": {
            "en": [
                "The professor is not explaining {course} concepts clearly and students are struggling to understand.",
                "Students are unable to get their doubts addressed properly during {course} lectures.",
            ],
            "hi": [
                "{course} के कॉन्सेप्ट ठीक से समझाए नहीं जा रहे हैं और छात्रों को समझने में परेशानी हो रही है।",
                "{course} की क्लास में छात्रों के डाउट्स ठीक से क्लियर नहीं किए जाते हैं।",
            ],
            "hinglish": [
                "Professor {course} ke concepts clearly explain nahi kar rahe, samajhne mein problem ho rahi hai.",
                "{course} lecture mein doubts properly address nahi hote.",
            ],
        },
        "faculty_conduct": {
            "en": [
                "{faculty} was rude to students during the {course} lecture.",
                "There was inappropriate behaviour by {faculty} in class today.",
            ],
            "hi": [
                "{faculty} ने क्लास में छात्रों के साथ बहुत बुरा व्यवहार किया।",
                "आज क्लास में {faculty} का व्यवहार सही नहीं था।",
            ],
            "hinglish": [
                "{faculty} ne class mein bohot rudely behave kiya students ke saath.",
                "Aaj {faculty} ka behaviour theek nahi tha class mein.",
            ],
        },
        "syllabus_course_progress": {
            "en": [
                "The {course} syllabus is being rushed and we are not able to understand the concepts.",
                "The {course} syllabus is not progressing as planned and important portions are still incomplete.",
            ],
            "hi": [
                "{course} का सिलेबस बहुत जल्दी पढ़ाया जा रहा है, समझ नहीं आ रहा।",
                "{course} का जरूरी सिलेबस अभी तक पूरा नहीं हुआ है।",
            ],
            "hinglish": [
                "{course} ka syllabus bohot jaldi cover ho raha hai, samajh nahi aa raha.",
                "{course} ka important syllabus abhi tak complete nahi hua hai.",
            ],
        },
        "attendance": {
            "en": [
                "My attendance for {course} is marked wrong despite being present.",
                "Attendance was not updated even after submitting a medical certificate.",
            ],
            "hi": [
                "{course} में मेरी अटेंडेंस गलत दिखाई जा रही है जबकि मैं क्लास में था।",
                "मेडिकल सर्टिफिकेट देने के बाद भी अटेंडेंस अपडेट नहीं हुई।",
            ],
            "hinglish": [
                "{course} mein meri attendance galat show ho rahi hai, main present tha.",
                "Medical certificate submit karne ke baad bhi attendance update nahi hui.",
            ],
        },
    },
    "Examinations": {
        "hall_ticket": {
            "en": [
                "My hall ticket for the {course} exam has an incorrect seat number.",
                "I am unable to download my hall ticket from the portal.",
            ],
            "hi": [
                "मेरे हॉल टिकट में सीट नंबर गलत है {course} परीक्षा के लिए।",
                "पोर्टल से हॉल टिकट डाउनलोड नहीं हो रहा है।",
            ],
            "hinglish": [
                "Mera hall ticket mein seat number galat hai {course} exam ke liye.",
                "Portal se hall ticket download nahi ho raha hai.",
            ],
        },
        "marks_discrepancy": {
            "en": [
                "The marks uploaded for {course} do not match what was shown on the answer sheet.",
                "There is a mismatch between internal and external marks for {course}.",
            ],
            "hi": [
                "{course} के लिए अपलोड किए गए मार्क्स आंसर शीट से मेल नहीं खाते।",
                "{course} के इंटरनल और एक्सटर्नल मार्क्स में अंतर है।",
            ],
            "hinglish": [
                "{course} ke marks jo upload hue hain wo answer sheet se match nahi karte.",
                "{course} ke internal aur external marks mein mismatch hai.",
            ],
        },
        "revaluation": {
            "en": [
                "I want to request revaluation for my {course} paper because the marks seem incorrect.",
                "Please process my revaluation application for the {course} exam; it has been {days}.",
            ],
            "hi": [
                "मुझे {course} पेपर का रीवैल्यूएशन चाहिए, मार्क्स गलत लग रहे हैं।",
                "रीवैल्यूएशन एप्लीकेशन दिए हुए {days} हो गए, कृपया प्रोसेस करें।",
            ],
            "hinglish": [
                "{course} paper ka revaluation chahiye, marks galat lag rahe hain.",
                "Revaluation application diye {days} ho gaye, please process karo.",
            ],
        },
        "examination_schedule": {
            "en": [
                "Two of my exams including {course} are scheduled on the same day.",
                "The exam datesheet has a timing or date conflict with another subject's exam.",
            ],
            "hi": [
                "मेरी दो परीक्षाएं {course} सहित एक ही दिन पर हैं।",
                "परीक्षा डेटशीट दूसरे विषय की परीक्षा के समय या तारीख से टकरा रही है।",
            ],
            "hinglish": [
                "Meri do exams including {course} same din pe hain.",
                "Exam datesheet doosre subject ke exam ke time ya date se clash kar rahi hai.",
            ],
        },
        "examination_process": {
            "en": [
                "I have an administrative problem with the examination process that is not related to marks or the timetable.",
                "There is an issue with the examination procedure and I need guidance on the correct process.",
            ],
            "hi": [
                "परीक्षा प्रक्रिया से जुड़ी प्रशासनिक समस्या है जो मार्क्स या डेटशीट से संबंधित नहीं है।",
                "परीक्षा की प्रक्रिया में समस्या है और सही प्रक्रिया के बारे में मार्गदर्शन चाहिए।",
            ],
            "hinglish": [
                "Exam process se related administrative problem hai, marks ya timetable ka issue nahi hai.",
                "Examination procedure mein issue hai aur correct process ki guidance chahiye.",
            ],
        },
    },
    "Fees_Accounts": {
        "fee_discrepancy": {
            "en": [
                "I was charged {amount} extra in my semester fee without explanation.",
                "There is a discrepancy of {amount} between the fee receipt and portal amount.",
            ],
            "hi": [
                "मुझसे सेमेस्टर फीस में {amount} अतिरिक्त लिया गया है बिना किसी कारण के।",
                "फीस रसीद और पोर्टल राशि में {amount} का अंतर है।",
            ],
            "hinglish": [
                "Mujhse semester fee mein {amount} extra charge hua hai bina reason ke.",
                "Fee receipt aur portal amount mein {amount} ka difference hai.",
            ],
        },
        "refund": {
            "en": [
                "My refund of {amount} has been pending for {days}.",
                "I still have not received my caution deposit refund after {days}.",
            ],
            "hi": [
                "मेरा {amount} का रिफंड {days} से पेंडिंग है।",
                "{days} बाद भी मेरा कॉशन डिपॉजिट रिफंड नहीं मिला।",
            ],
            "hinglish": [
                "Mera {amount} ka refund {days} se pending hai.",
                "{days} baad bhi caution deposit refund nahi mila.",
            ],
        },
        "scholarship": {
            "en": [
                "My scholarship status has not been updated on the portal for {days}.",
                "I have not received any update about my scholarship application.",
            ],
            "hi": [
                "मेरी स्कॉलरशिप स्थिति पोर्टल पर {days} से अपडेट नहीं हुई है।",
                "मुझे स्कॉलरशिप एप्लीकेशन के बारे में कोई अपडेट नहीं मिला।",
            ],
            "hinglish": [
                "Meri scholarship status portal pe {days} se update nahi hui hai.",
                "Scholarship application ke baare mein koi update nahi mila.",
            ],
        },
        "payment_transaction": {
            "en": [
                "The fee payment failed but {amount} was deducted from my account.",
                "I am unable to complete fee payment because the transaction is failing.",
            ],
            "hi": [
                "फीस पेमेंट फेल हो गया लेकिन {amount} मेरे अकाउंट से कट गया।",
                "ट्रांजैक्शन फेल होने की वजह से फीस पेमेंट पूरा नहीं हो पा रहा है।",
            ],
            "hinglish": [
                "Fee payment fail ho gaya lekin {amount} account se kat gaya.",
                "Transaction fail hone ki wajah se fee payment complete nahi ho raha.",
            ],
        },
        "accounts_financial_documentation": {
            "en": [
                "I need a copy of my fee receipt and payment record for official documentation.",
                "My financial receipt or payment record is missing from the student account.",
            ],
            "hi": [
                "मुझे आधिकारिक दस्तावेज के लिए फीस रसीद और पेमेंट रिकॉर्ड की कॉपी चाहिए।",
                "मेरे स्टूडेंट अकाउंट में फीस रसीद या पेमेंट रिकॉर्ड दिखाई नहीं दे रहा है।",
            ],
            "hinglish": [
                "Official documentation ke liye fee receipt aur payment record ki copy chahiye.",
                "Student account mein fee receipt ya payment record show nahi ho raha.",
            ],
        },
    },
    "IT_Library": {
        "portal_account_access": {
            "en": [
                "I am unable to log into the student portal for {days}.",
                "The portal shows an error every time I try to access my results.",
            ],
            "hi": [
                "मैं {days} से स्टूडेंट पोर्टल में लॉगिन नहीं कर पा रहा हूँ।",
                "रिजल्ट देखने की कोशिश करने पर पोर्टल एरर दिखाता है।",
            ],
            "hinglish": [
                "Main {days} se student portal mein login nahi kar pa raha.",
                "Result dekhne ki koshish karne pe portal error deta hai.",
            ],
        },
        "wifi_network": {
            "en": [
                "Wi-Fi in {building} has not been working for {days}.",
                "The internet connection in the {floor} lab keeps disconnecting.",
            ],
            "hi": [
                "{building} में वाईफाई {days} से काम नहीं कर रहा है।",
                "{floor} लैब में इंटरनेट बार-बार डिस्कनेक्ट हो रहा है।",
            ],
            "hinglish": [
                "{building} mein WiFi {days} se kaam nahi kar raha hai.",
                "{floor} lab mein internet baar baar disconnect ho raha hai.",
            ],
        },
        "software_license": {
            "en": [
                "The licensed software for {course} lab has expired and needs renewal.",
                "We do not have access to the required software license in the lab.",
            ],
            "hi": [
                "{course} लैब का लाइसेंस्ड सॉफ्टवेयर एक्सपायर हो गया है, रिन्यू चाहिए।",
                "लैब में जरूरी सॉफ्टवेयर लाइसेंस उपलब्ध नहीं है।",
            ],
            "hinglish": [
                "{course} lab ka licensed software expire ho gaya hai, renew chahiye.",
                "Lab mein zaroori software license available nahi hai.",
            ],
        },
        "library_book_availability": {
            "en": [
                "The reference book for {course} is not available in the library.",
                "I have been waiting {days} for a library book to be issued.",
            ],
            "hi": [
                "{course} के लिए संदर्भ पुस्तक लाइब्रेरी में उपलब्ध नहीं है।",
                "लाइब्रेरी की किताब मिलने में {days} हो गए।",
            ],
            "hinglish": [
                "{course} ki reference book library mein available nahi hai.",
                "Library book issue hone mein {days} ho gaye.",
            ],
        },
        "library_digital_resources": {
            "en": [
                "I cannot access the online journal or digital library resource required for my course.",
                "The e-library resource is not accessible with my student account.",
            ],
            "hi": [
                "मेरे कोर्स के लिए जरूरी ऑनलाइन जर्नल या डिजिटल लाइब्रेरी संसाधन नहीं खुल रहा है।",
                "मेरे स्टूडेंट अकाउंट से ई-लाइब्रेरी संसाधन एक्सेस नहीं हो रहा है।",
            ],
            "hinglish": [
                "Course ke liye required online journal ya digital library resource access nahi ho raha.",
                "Student account se e-library resource access nahi ho raha.",
            ],
        },
    },
    "Infrastructure": {
        "classroom_lab_maintenance": {
            "en": [
                "The projector in {building} {floor} classroom has been broken for {days}.",
                "Furniture or physical equipment in the classroom lab is damaged and needs repair.",
            ],
            "hi": [
                "{building} {floor} की क्लास का प्रोजेक्टर {days} से खराब है।",
                "क्लासरूम या लैब का फर्नीचर और भौतिक उपकरण खराब हैं, मरम्मत चाहिए।",
            ],
            "hinglish": [
                "{building} {floor} classroom ka projector {days} se kharab hai.",
                "Classroom ya lab ka furniture aur physical equipment damaged hai, repair chahiye.",
            ],
        },
        "electrical_issues": {
            "en": [
                "There is exposed wiring near {building} {floor}, which is unsafe.",
                "Frequent electrical failures in {building} are disrupting lab sessions.",
            ],
            "hi": [
                "{building} {floor} के पास खुली वायरिंग है, यह असुरक्षित है।",
                "{building} में बार-बार बिजली की समस्या से लैब सेशन बाधित हो रहे हैं।",
            ],
            "hinglish": [
                "{building} {floor} ke paas exposed wiring hai, ye unsafe hai.",
                "{building} mein baar baar electrical problem se lab session disrupt ho raha hai.",
            ],
        },
        "sanitation_cleanliness": {
            "en": [
                "Washrooms in {building} are not being cleaned regularly.",
                "There is a sanitation and cleanliness problem near the {floor} corridor.",
            ],
            "hi": [
                "{building} के वॉशरूम नियमित रूप से साफ नहीं किए जा रहे हैं।",
                "{floor} गलियारे के पास सफाई और सैनिटेशन की समस्या है।",
            ],
            "hinglish": [
                "{building} ke washroom regularly clean nahi ho rahe hain.",
                "{floor} corridor ke paas safai aur sanitation ki problem hai.",
            ],
        },
        "parking_access": {
            "en": [
                "There is not enough two-wheeler parking space for students on campus.",
                "Students are facing parking access problems because available parking space has been reduced.",
            ],
            "hi": [
                "कॉलेज में छात्रों के दोपहिया वाहनों के लिए पर्याप्त पार्किंग की जगह नहीं है।",
                "पार्किंग की जगह कम होने से छात्रों को वाहन पार्क करने में समस्या हो रही है।",
            ],
            "hinglish": [
                "College mein students ke two-wheeler ke liye enough parking space nahi hai.",
                "Parking space kam hone ki wajah se students ko vehicle park karne mein problem ho rahi hai.",
            ],
        },
        "construction_facility_disruption": {
            "en": [
                "Construction work is blocking access to a common college facility.",
                "Ongoing construction is disrupting normal use of the campus area.",
            ],
            "hi": [
                "निर्माण कार्य से कॉलेज की एक सामान्य सुविधा तक पहुंच बाधित हो रही है।",
                "चल रहे निर्माण कार्य से कैंपस क्षेत्र का सामान्य उपयोग प्रभावित हो रहा है।",
            ],
            "hinglish": [
                "Construction work ki wajah se common college facility ka access block ho raha hai.",
                "Ongoing construction se campus area ka normal use disturb ho raha hai.",
            ],
        },
    },
    "Transport": {
        "commuting_accessibility": {
            "en": [
                "Students travelling from {location} spend three hours commuting to college every day and need better accessibility consideration.",
                "Long-distance commuting is making it difficult for students from {location} to attend the full college schedule regularly.",
            ],
            "hi": [
                "{location} से आने वाले छात्रों को कॉलेज पहुंचने में लगभग तीन घंटे लगते हैं और उनकी यात्रा की समस्या पर ध्यान देने की जरूरत है।",
                "दूर से आने वाले {location} के छात्रों के लिए रोज पूरा कॉलेज शेड्यूल करना मुश्किल हो रहा है।",
            ],
            "hinglish": [
                "{location} se students ko college aane mein around three hours lagte hain, long commute ki problem consider honi chahiye.",
                "{location} se daily travel ki wajah se students ke liye full college schedule follow karna difficult ho raha hai.",
            ],
        },
        "transport_academic_conflict": {
            "en": [
                "Because of the long commute from {location}, students sometimes have to leave lectures early to catch their return train.",
                "The current lecture timings create a conflict for students travelling long distances from {location}.",
            ],
            "hi": [
                "{location} से लंबी यात्रा के कारण छात्रों को कभी-कभी वापसी की ट्रेन पकड़ने के लिए लेक्चर जल्दी छोड़ना पड़ता है।",
                "मौजूदा लेक्चर टाइमिंग {location} से दूर से आने वाले छात्रों के लिए समस्या पैदा करती है।",
            ],
            "hinglish": [
                "{location} se long commute ki wajah se students ko return train pakadne ke liye kabhi lecture early leave karna padta hai.",
                "Current lecture timings {location} se long-distance travel karne wale students ke liye conflict create karti hain.",
            ],
        },
        "travel_safety_access": {
            "en": [
                "Students travelling long distances have raised a safety or accessibility concern about reaching campus regularly.",
                "The journey to and from campus creates an access or safety concern for some students who travel from far away.",
            ],
            "hi": [
                "दूर से आने वाले छात्रों ने कैंपस तक नियमित पहुंच को लेकर सुरक्षा या पहुंच संबंधी चिंता बताई है।",
                "कॉलेज आने-जाने की यात्रा कुछ दूर रहने वाले छात्रों के लिए सुरक्षा या पहुंच की समस्या पैदा करती है।",
            ],
            "hinglish": [
                "Far se travel karne wale students ne campus reach karne ko lekar safety ya accessibility concern raise kiya hai.",
                "College aane-jaane ka journey far-away students ke liye safety ya access problem create karta hai.",
            ],
        },
    },
    "Canteen": {
        "food_quality": {
            "en": [
                "The food quality in the canteen has dropped significantly this month.",
                "I found the food undercooked and cold at the canteen today.",
            ],
            "hi": [
                "कैंटीन में खाने की क्वालिटी इस महीने काफी गिर गई है।",
                "आज कैंटीन में खाना अधपका और ठंडा मिला।",
            ],
            "hinglish": [
                "Canteen mein khane ki quality is mahine kaafi gir gayi hai.",
                "Aaj canteen mein khana undercooked aur thanda mila.",
            ],
        },
        "food_hygiene": {
            "en": [
                "The canteen seating area is not being cleaned properly.",
                "There were insects near the food counter at the canteen.",
            ],
            "hi": [
                "कैंटीन का बैठने का क्षेत्र ठीक से साफ नहीं किया जा रहा है।",
                "कैंटीन के फूड काउंटर के पास कीड़े दिखे।",
            ],
            "hinglish": [
                "Canteen ka seating area properly clean nahi ho raha hai.",
                "Canteen ke food counter ke paas insects dikhe.",
            ],
        },
        "pricing": {
            "en": [
                "Canteen prices were increased without any notice to students.",
                "The price for a basic meal is too high compared to the quality offered.",
            ],
            "hi": [
                "कैंटीन की कीमतें बिना किसी सूचना के बढ़ा दी गईं।",
                "बेसिक खाने की कीमत क्वालिटी के मुकाबले बहुत ज्यादा है।",
            ],
            "hinglish": [
                "Canteen ki prices bina notice ke badha di gayi hain.",
                "Basic meal ki price quality ke comparison mein bohot zyada hai.",
            ],
        },
        "menu_variety": {
            "en": [
                "The canteen menu has not changed in months, and students want more variety.",
                "There are very few options for students with dietary restrictions.",
            ],
            "hi": [
                "कैंटीन का मेनू महीनों से नहीं बदला, हमें और विविधता चाहिए।",
                "डाइटरी रिस्ट्रिक्शन वाले छात्रों के लिए बहुत कम विकल्प हैं।",
            ],
            "hinglish": [
                "Canteen ka menu mahino se nahi badla, hume aur variety chahiye.",
                "Dietary restriction wale students ke liye bohot kam options hain.",
            ],
        },
        "food_service": {
            "en": [
                "The canteen serving process is too slow during normal service hours.",
                "A food item listed on the menu is repeatedly unavailable when students order it.",
            ],
            "hi": [
                "कैंटीन में सामान्य समय पर खाना सर्व करने की प्रक्रिया बहुत धीमी है।",
                "मेनू में मौजूद खाना ऑर्डर करने पर बार-बार उपलब्ध नहीं होता।",
            ],
            "hinglish": [
                "Canteen mein normal service time par food serving bohot slow hai.",
                "Menu mein listed food item order karne par baar baar available nahi hota.",
            ],
        },
    },
}

SAFETY_SUBCATS = {"electrical_issues", "travel_safety_access"}


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
            if subcat not in TEMPLATES.get(category, {}):
                raise ValueError(f"Missing templates for {category}/{subcat}")
            for lang in taxonomy["languages"]:
                templates = TEMPLATES[category][subcat].get(lang)
                if not templates:
                    raise ValueError(f"Missing {lang} templates for {category}/{subcat}")
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
                        "duplicate_of": "",
                    })
                    row_id += 1

    n_dupes = int(len(rows) * dup_rate)
    dupe_targets = random.sample(rows, n_dupes)
    for target in dupe_targets:
        rows.append({
            "id": row_id,
            "text": target["text"],
            "language": target["language"],
            "category": target["category"],
            "subcategory": target["subcategory"],
            "priority": target["priority"],
            "duplicate_of": target["id"],
        })
        row_id += 1

    random.shuffle(rows)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "text", "language", "category", "subcategory", "priority", "duplicate_of"],
        )
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
