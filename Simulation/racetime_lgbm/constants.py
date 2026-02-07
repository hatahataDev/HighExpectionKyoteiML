TARGET_COL = "レースタイム_refix"
RAW_TARGET_COL = "レースタイム"
TARGET_SEC_COL = "target_sec"
DATE_COL = "date"
RACE_ID_COL = "race_id"

RESULT_COLS = ["着", "決まり手", "PST", "TSC"]
IDENTIFIER_COLS = [RACE_ID_COL]
DROP_COLS = RESULT_COLS + IDENTIFIER_COLS + [RAW_TARGET_COL, TARGET_COL]

CATEGORICAL_COLS = [
    "place",
    "race_no",
    "ESC",
    "枠",
    "racer_no",
    "moter_no",
    "boat_no",
    "sibu",
    "born",
    "piston",
    "ring",
    "electric",
    "carburetor",
    "cylinder",
    "shafts",
    "gears",
    "carrier",
    "propera",
]

PRED_SAMPLE_ROWS_DEFAULT = 100000
PLOT_SAMPLE_ROWS_DEFAULT = 200000
ERROR_TOPN_PLACE_DEFAULT = 20
