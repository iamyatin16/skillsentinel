"""
Add a new login (for example a second trainer or another officer).

Examples:
    python add_user.py trainer2 mypass123 trainer "Trainer Nashik" Centre-Nashik TR02
    python add_user.py officer2 mypass123 monitor "Officer Two"
Arguments:  username  password  role(trainer or monitor)  "Full name"  [centre]  [trainer_id]
For a new centre also add lines in data/centres.csv, data/roster.csv and data/gov_inventory.csv.
"""
import sys, uuid
import pandas as pd
import store

store.init_data()
if len(sys.argv) < 5:
    raise SystemExit(__doc__)
username, password, role, name = sys.argv[1:5]
centre = sys.argv[5] if len(sys.argv) > 5 else ""
tid = sys.argv[6] if len(sys.argv) > 6 else ""
if role not in ("trainer", "monitor"):
    raise SystemExit("role must be trainer or monitor")
df = pd.read_csv(store.USERS_CSV, dtype=str).fillna("")
df = df[df["username"].str.lower() != username.lower()]
salt = uuid.uuid4().hex[:8]
df.loc[len(df)] = [username, salt, store._hash(salt, password), role, name, centre, tid]
df.to_csv(store.USERS_CSV, index=False)
print(f"Saved login '{username}' ({role}).")
