"""Project-wide constants. No training, splits, or pipelines live here."""

SEED = 42

# Columns that must never be used as model inputs (leakage, identity, or post-event).
BANNED_COLUMNS = (
    "Completed_Date",
    "Changed_on",
    "Notification_type",
    "Notifications_Created",
    "resolution_days",
    "target",
    "Suburb",
    "Work_Center",
    "Notification",
)
