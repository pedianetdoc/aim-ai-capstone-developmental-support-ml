"""Project-wide variable definitions and model constants."""

RANDOM_STATE = 42
TARGET = "recognized_dev_concern_support"

DIAGNOSIS_VARS = [
    "k2q36a","k2q37a","k2q35a","k2q31a","k2q30a","k2q60a",
    "k2q34a","k2q33a","k2q32a","k2q38a","k2q61a","downsyn","fasd"
]

SUPPORT_VARS = ["sc_k2q19","sc_k2q22","k6q15","k4q36"]

BASIC_VARS = [
    "sc_age_years","sc_sex","agepos4","totkids_r","hhcount","famcount",
    "birthwt","k2q05"
]

DEVELOPMENT_VARS = [
    "oneword","twowords","threewords","askquestion","askquestion2","tellstory",
    "understand","directions","point","directions2","understand2","clearexp",
    "bounceaball","drawacircle","drawaface","drawaperson","recogbegin","samesound",
    "writename","focuson","readonedigit","simpleaddition","groupofobjects","recogabc",
    "countto_r","rhymeword_r","nameemotions","startnewact","temper_r","playwell",
    "distracted","hurtsad","calmdown_r","waitforturn","hardwork","sharetoys"
]

MEDICAL_VARS = [
    "breathing","swallowing","stomach","physicalpain","hands","coordination",
    "toothaches","gumbleed","cavities","k2q43b","blindness","allergies","k2q40a",
    "autoimmune","diabetes","k2q42a","heart","headache","blood","cystfib",
    "concussion","overweight"
]

ACE_VARS = [
    "ace1","ace3","ace4","ace5","ace6","ace7","ace8","ace9","ace10","ace11",
    "everhomeless","homeevic","moves"
]

ROUTINE_VARS = [
    "hoursleep05","outdoorswkday","outdoorswkend","screentime","bedtime",
    "foodsit","sugardrink","vegetables","fruit"
]

FAMILY_VARS = [
    "higrade","family_r","a1_employed_r","a1_physhealth","a1_menthealth",
    "k8q35","talkabout","wktosolve","strengths","hopeful","goforhelp",
    "k8q31","k8q32","k8q34"
]

MODEL_A_FEATURES = (
    BASIC_VARS + DEVELOPMENT_VARS + MEDICAL_VARS + ACE_VARS +
    ROUTINE_VARS + FAMILY_VARS
)

MODEL_B_BASIC = ["sc_age_years","sc_sex","birthwt","k2q05"]
MODEL_B_COMMUNICATION = [
    "oneword","twowords","threewords","askquestion","askquestion2","tellstory",
    "understand","directions","point","directions2","understand2","clearexp"
]
MODEL_B_LEARNING = ["focuson","rhymeword_r"]
MODEL_B_REGULATION = [
    "nameemotions","startnewact","temper_r","playwell","distracted","hurtsad",
    "calmdown_r","waitforturn","hardwork","sharetoys"
]
MODEL_B_MOTOR = [
    "bounceaball","drawacircle","drawaface","drawaperson","hands","coordination"
]
MODEL_B_MEDICAL = ["k2q43b","k2q42a","concussion"]
MODEL_B_ROUTINE = ["screentime"]

MODEL_B_FEATURES = (
    MODEL_B_BASIC + MODEL_B_COMMUNICATION + MODEL_B_LEARNING +
    MODEL_B_REGULATION + MODEL_B_MOTOR + MODEL_B_MEDICAL + MODEL_B_ROUTINE
)

# Top 44 predictors selected from the exploratory SHAP-ranking analysis.
MODEL_C_FEATURES = [
    "sc_age_years","k8q31","clearexp","tellstory","distracted","sc_sex",
    "focuson","allergies","k8q34","calmdown_r","stomach","startnewact",
    "famcount","recogbegin","k2q05","hoursleep05","a1_menthealth","askquestion2",
    "ace3","playwell","waitforturn","bounceaball",
    "askquestion","birthwt","drawacircle","hardwork","hurtsad","k2q43b",
    "nameemotions","rhymeword_r","understand2","agepos4","ace8","bedtime",
    "blindness","fruit","heart","hhcount","k2q40a","k8q35","recogabc",
    "simpleaddition","sugardrink","vegetables"
]

NUMERIC_MODEL_A = ["sc_age_years","hhcount","famcount","totkids_r"]
