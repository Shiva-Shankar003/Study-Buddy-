import datetime as dt
import learning


def make_plan(subjects, exam_date, hours_per_day=2.0, sessions_per_day=3):
    """Spread study sessions until the exam. Weaker topics get more sessions."""
    today = dt.date.today()
    days = max((exam_date - today).days, 1)
    items = []
    for s in subjects:
        scores = learning.mastery(s)
        if scores:
            items += [(f"{s}: {t}", m) for t, m in scores.items()]
        else:
            items.append((f"{s}: all topics", 0.5))
    weights = {name: (1.0 - m) + 0.1 for name, m in items}
    total = sum(weights.values())
    slots = days * sessions_per_day
    counts = {n: max(1, round(slots * w / total)) for n, w in weights.items()}
    events = sorted(((j + 0.5) / c, n) for n, c in counts.items() for j in range(c))
    order = [n for _, n in events]  # topics spread evenly across the whole period
    minutes = int(hours_per_day * 60 / sessions_per_day)
    rows = []
    for d in range(days):
        todays = order[d * sessions_per_day:(d + 1) * sessions_per_day]
        if not todays:
            break
        rows.append({"date": (today + dt.timedelta(days=d)).isoformat(),
                     "sessions": " | ".join(todays), "minutes_each": minutes})
    return rows
