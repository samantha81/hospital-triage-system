# Interface

# ============================================================
# 1. IMPORTS & CONFIG
# ============================================================
import os
import datetime
from collections import Counter

import pandas as pd
import altair as alt
import streamlit as st

import nlp_core as core

THRESH = core.CONFIDENCE_THRESHOLD

# Department list, from the corpus folder
DEPARTMENTS = sorted(
    os.path.splitext(f)[0] for f in os.listdir(core.CORPUS_DIR) if f.endswith(".txt")
)

st.set_page_config(page_title="Xiamen SmartTriage", page_icon="🏥", layout="centered")

# ============================================================
# 2. SESSION STATE (patient flow + records)
# ============================================================
if "stage" not in st.session_state:
    st.session_state.stage = "checkin"     # checkin -> intake -> result
if "patient" not in st.session_state:
    st.session_state.patient = {}
if "result" not in st.session_state:
    st.session_state.result = None
if "ticket_counter" not in st.session_state:
    st.session_state.ticket_counter = 0
if "patient_counter" not in st.session_state:
    st.session_state.patient_counter = 0   # gives each check-in a unique patient id
if "queue" not in st.session_state:
    st.session_state.queue = {}            # department -> count
if "log" not in st.session_state:
    st.session_state.log = []              # list of triaged visits (for staff view)


# ============================================================
# 3. HELPER FUNCTIONS (records + queue management)
# ============================================================
def reset_to_checkin():
    st.session_state.stage = "checkin"
    st.session_state.patient = {}
    st.session_state.result = None


def save_record(p, r, top_dept, top_conf):
    """Save/UPDATE this patient's record by patient id (no duplicates on amend)."""
    existing = next((e for e in st.session_state.log if e["id"] == p["id"]), None)
    if existing:                            # amend -> update in place, keep the ticket
        existing.update({
            "name": p["name"], "age": p["age"], "symptoms": r["symptoms_text"],
            "department": top_dept, "confidence": top_conf,
        })
        r["ticket"], r["queue_pos"] = existing["ticket"], existing["queue_pos"]
    else:                                   # first confirm -> new ticket + queue slot
        st.session_state.ticket_counter += 1
        st.session_state.queue[top_dept] = st.session_state.queue.get(top_dept, 0) + 1
        r["ticket"] = st.session_state.ticket_counter
        r["queue_pos"] = st.session_state.queue[top_dept]
        st.session_state.log.append({
            "id": p["id"], "name": p["name"], "age": p["age"],
            "symptoms": r["symptoms_text"], "department": top_dept,
            "confidence": top_conf,
            "ticket": r["ticket"], "queue_pos": r["queue_pos"],
        })
    r["confirmed"] = True


def reassign_department(entry, new_dept):
    """STAFF action: re-route a triaged patient to the correct department and
    rebalance the per-department queue counts."""
    old_dept = entry["department"]
    if old_dept == new_dept:
        return
    q = st.session_state.queue
    if old_dept in q:                       # free up the old department's slot
        q[old_dept] = max(0, q[old_dept] - 1)
        if q[old_dept] == 0:
            del q[old_dept]
    q[new_dept] = q.get(new_dept, 0) + 1    # take a slot in the new department
    entry["queue_pos"] = q[new_dept]
    entry["department"] = new_dept
    entry["reassigned"] = True


# ============================================================
# 4. SIDEBAR (view switch + live session snapshot)
# ============================================================
with st.sidebar:
    st.markdown("## 🏥 Xiamen SmartTriage")
    st.caption("Smart symptom triage & department routing for Xiamen Hospital")
    st.divider()

    view = st.radio("View", ["🧑 Patient", "🩺 Staff / Admin"])

    st.divider()
    _log = st.session_state.log
    st.markdown("#### 📋 This session")
    s1, s2 = st.columns(2)
    s1.metric("Triaged", len(_log))
    s2.metric("Departments", len({v["department"] for v in _log}))
    if _log:
        latest = _log[-1]
        st.caption(f"Last patient checked in: **{latest['name']}** → "
                   f"{latest['department'].title()}")

# ============================================================
# 5. STAFF DASHBOARD (charts, triage log, reassignment)
# ============================================================
if view == "🩺 Staff / Admin":
    st.title("🩺 Xiamen SmartTriage — Staff Dashboard")

    log = st.session_state.log
    if not log:
        st.info("No patients triaged yet this session. Switch to the Patient "
                "view to triage someone, then come back.")
    else:
        # Department distribution
        st.subheader("Department distribution")
        dept_counts = Counter(v["department"] for v in log)
        dist_df = pd.DataFrame({
            "Department": [d.title() for d in dept_counts],
            "Patients": list(dept_counts.values()),
        })
        pie = (
            alt.Chart(dist_df)
            .mark_arc()
            .encode(
                theta=alt.Theta("Patients:Q"),
                color=alt.Color("Department:N", title="Department"),
                tooltip=["Department", "Patients"],
            )
        )
        st.altair_chart(pie, use_container_width=True)

        # Average confidence per department
        st.subheader("Average confidence / dept")
        conf_by_dept = {}
        for v in log:
            if v.get("reassigned"):
                continue
            conf_by_dept.setdefault(v["department"], []).append(v["confidence"])
        conf_df = pd.DataFrame({
            "Department": [d.title() for d in conf_by_dept],
            "Avg confidence": [round(sum(vs) / len(vs), 3) for vs in conf_by_dept.values()],
        })
        conf_bar = (
            alt.Chart(conf_df)
            .mark_bar(color="#93C5FD")
            .encode(
                x=alt.X("Department:N", sort="-y", title=None),
                y=alt.Y("Avg confidence:Q"),
                tooltip=["Department", "Avg confidence"],
            )
        )
        avg_threshold = (
            alt.Chart(pd.DataFrame({"Avg confidence": [THRESH]}))
            .mark_rule(color="#EF4444", strokeDash=[5, 5])
            .encode(y="Avg confidence:Q")
        )
        if conf_by_dept:
            st.altair_chart(conf_bar + avg_threshold, use_container_width=True)
            st.caption("Manually reassigned patients are excluded.")
        else:
            st.info("No model-assigned patients yet — all current patients were "
                    "manually reassigned by staff.")

        st.subheader("Current queue per department")
        if st.session_state.queue:
            qcols = st.columns(len(st.session_state.queue))
            for col, (dept, n) in zip(qcols, st.session_state.queue.items()):
                col.metric(dept.title(), n)

        st.subheader("Triage log")
        # Per-row flags drive both the displayed confidence and the row colour
        reassigned = [bool(v.get("reassigned")) for v in log]
        low_conf = [v["confidence"] < THRESH and not reassigned[i]
                    for i, v in enumerate(log)]

        df = pd.DataFrame(log).rename(columns={
            "ticket": "Ticket", "name": "Name", "age": "Age", "department": "Department",
            "confidence": "Confidence", "symptoms": "Symptoms",
        })
        # Manually reassigned patients have no model confidence
        df["Confidence"] = ["-" if reassigned[i] else f"{c:.0%}"
                            for i, c in enumerate(df["Confidence"])]
        cols = ["Ticket", "Name", "Age", "Department", "Confidence", "Symptoms"]
        view = df[[c for c in cols if c in df.columns]].reset_index(drop=True)

        # Row colours: blue = staff-reassigned, red = below confidence threshold
        def _highlight(row):
            i = row.name
            if reassigned[i]:
                color = "background-color: rgba(59,130,246,0.25)"
            elif low_conf[i]:
                color = "background-color: rgba(239,68,68,0.25)"
            else:
                color = ""
            return [color] * len(row)

        st.dataframe(view.style.apply(_highlight, axis=1),
                     use_container_width=True, hide_index=True)
        st.caption(f"🔴 Below the {THRESH:.0%} confidence threshold (may need to be "
                   "reviewed)  •  🔵 Manually reassigned by staff")

        # --- Reassign / correct department (staff only) ---
        st.subheader("Reassign department")
        st.caption("If you think a patient has been wrongly assigned based on their "
                   "symptoms, you can manually reassign them to the correct "
                   "department here. The queue updates automatically.")
        options = {
            f"#{v['ticket']:03d} — {v['name']} (currently {v['department'].title()})": v
            for v in log
        }
        sel = st.selectbox("Select a patient", list(options.keys()))
        entry = options[sel]
        cur = entry["department"]
        new_dept = st.selectbox(
            "Correct department", DEPARTMENTS,
            index=DEPARTMENTS.index(cur) if cur in DEPARTMENTS else 0,
            format_func=lambda d: d.title(),
        )
        if st.button("Apply reassignment", type="primary"):
            if new_dept == cur:
                st.info("That patient is already in this department — nothing changed.")
            else:
                reassign_department(entry, new_dept)
                st.success(f"#{entry['ticket']:03d} {entry['name']} reassigned from "
                           f"{cur.title()} to {new_dept.title()}.")
                st.rerun()

    st.stop()


# ============================================================
# 6. PATIENT FLOW (check-in -> intake -> referral)
# ============================================================
st.title("🏥 Xiamen SmartTriage")
st.divider()


# ===== PAGE 1: CHECK-IN =====
if st.session_state.stage == "checkin":
    st.markdown(
        "#### 👋 Welcome to Xiamen Hospital\n"
        "This self-service kiosk helps you reach the **right department faster**. "
        "Instead of guessing which specialist to see, simply tell us how you feel "
        "in your own words — our system reads your symptoms and assigns the most "
        "suitable department for you."
    )
    st.markdown(
        "**How it works — 3 simple steps:**\n"
        "1. **Check in** — enter a few basic details below.\n"
        "2. **Describe your symptoms** — write what's bothering you, in plain language.\n"
        "3. **Get your referral** — receive a recommended department and a queue ticket."
    )
    st.info("🔒 Your information is used only for this triage session and is **not** "
            "stored permanently. This tool supports but does not replace a "
            "doctor's assessment.")
    st.divider()

    st.subheader("Patient Check-in")
    with st.form("checkin_form"):
        name = st.text_input("Full Name")
        col1, col2 = st.columns(2)
        age = col1.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        gender = col2.selectbox("Gender", ["Prefer not to say", "Female", "Male", "Other"])
        submitted = st.form_submit_button("Start triage →", type="primary")
    if submitted:
        if not name.strip():
            st.warning("Please enter the patient's name.")
        else:
            st.session_state.patient_counter += 1
            st.session_state.patient = {
                "id": st.session_state.patient_counter,
                "name": name.strip(), "age": int(age), "gender": gender,
                "checkin_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            st.session_state.stage = "intake"
            st.rerun()


# ===== PAGE 2: SYMPTOM INTAKE =====
elif st.session_state.stage == "intake":
    st.subheader("Symptom Intake")
    # Pre-fill previous text when amending an existing record.
    prev_text = (st.session_state.result or {}).get("symptoms_text", "")
    symptoms_text = st.text_area(
        "Describe your symptoms in your own words:",
        value=prev_text,
        placeholder="e.g. I've been having chest pain and feel short of breath",
        height=120,
    )
    # Back on the left, Submit on the right.
    col1, col2 = st.columns([1, 1])
    if col1.button("← Back to check-in"):
        reset_to_checkin()
        st.rerun()
    if col2.button("Submit for triage →", type="primary"):
        if not symptoms_text.strip():
            st.warning("Please describe the symptoms.")
        else:
            result = core.predict(symptoms_text)
            # Validate at the intake step: if nothing medical was recognised, keep the patient on this page instead of sending them to an empty slip.
            if result["status"] == "no_symptoms":
                st.warning("⚠ No recognised symptoms found. Please describe the "
                           "symptoms using clearer medical terms (e.g. \"chest pain\", "
                           "\"headache\", \"skin rash\").")
            elif result["status"] == "rejected":
                st.warning("⚠ That doesn't look like a medical complaint. Please "
                           "describe the symptoms you're experiencing.")
            else:
                result["symptoms_text"] = symptoms_text.strip()
                st.session_state.result = result
                st.session_state.stage = "result"
                st.rerun()


# ===== PAGE 3: REFERRAL SLIP =====
elif st.session_state.stage == "result":
    p = st.session_state.patient
    r = st.session_state.result

    # Safety net: intake (step 2) already blocks non-medical / no-symptom input so only reach this page with status == "ok".
    if r["status"] != "ok":
        st.subheader("🧾 Referral Slip")
        st.warning("No recognised symptoms found. Please go back and describe the "
                   "symptoms using clearer medical terms.")
    else:
        ranked = r["ranked"]
        top_dept, top_conf = ranked[0]

        # The formal referral slip (header + patient info + ticket) only appears after confirmation, before that it's the triage result.
        if r.get("confirmed"):
            st.markdown(
                f"✅ You've been assigned to the **{top_dept.title()}** department, "
                "and your record has been saved. Meanwhile, our admin will review "
                "whether you've been correctly sorted based on your symptoms — "
                "so no worries!"
            )
            st.subheader("🧾 Referral Slip")
            c1, c2, c3 = st.columns([1, 2, 1])
            c1.metric("Ticket", f"#{r['ticket']:03d}")
            c2.metric("Department", top_dept.title())
            c3.metric("Queue position", f"{r['queue_pos']}")
            st.write(f"**Patient Name:** {p['name']}")
            st.write(f"**Age:** {p['age']}")
            st.write(f"**Gender:** {p['gender']}")
            st.write(f"**Checked in:** {p['checkin_time']}")
        else:
            st.subheader("🩺 Triage Result")
            st.metric("Department", top_dept.title())

        if r["low_confidence"]:
            st.warning(f"⚠ Assigned to **{top_dept.title()}**, but our confidence is "
                       f"low ({top_conf:.0%}). You can **go back** and describe your "
                       "symptoms in more detail, or visit the **front counter** for "
                       "help. Either way, our staff will keep track of your case and "
                       "make sure you're seen by the right department.")
        else:
            st.success(f"Assigned department: **{top_dept.title()}**  "
                       f"({top_conf:.0%} confidence)")

        # "Other possible departments" + "How this was decided" are review aids
        if not r.get("confirmed"):
            with st.expander("Other possible departments"):
                for d, prob in ranked:
                    st.write(f"- {d.title()}: {prob:.0%}")

            st.markdown("#### How this was decided")

            st.caption("Department confidence")
            conf_df = pd.DataFrame({
                "Department": [d.title() for d, _ in ranked],
                "Confidence": [round(p, 3) for _, p in ranked],
            })
            conf_chart = (
                alt.Chart(conf_df)
                .mark_bar(color="#2563EB")
                .encode(
                    x=alt.X("Confidence:Q"),
                    y=alt.Y("Department:N", sort="-x", title=None),
                    tooltip=["Department", "Confidence"],
                )
            )
            conf_threshold = (
                alt.Chart(pd.DataFrame({"Confidence": [THRESH]}))
                .mark_rule(color="#EF4444", strokeDash=[5, 5])
                .encode(x="Confidence:Q")
            )
            st.altair_chart(conf_chart + conf_threshold, use_container_width=True)

            if r.get("reasons"):
                cap_col, info_col = st.columns([0.9, 0.1])
                cap_col.caption(f"Why {top_dept.title()}? (top contributing words)")
                with info_col.popover("ℹ️"):
                    st.markdown("**How is *contribution* calculated?**")
                    st.markdown(
                        "Each word's contribution is how strongly it pushed the text "
                        f"toward **{top_dept.title()}**:"
                    )
                    st.latex(r"\text{contribution} = \text{tf-idf}(word)\;\times\;"
                             r"\text{weight}(word,\,dept)")
                    st.markdown(
                        "- **tf-idf(word)** — how important the word is in this input "
                        "(frequent here, rare across all complaints → higher value).\n"
                        "- **weight(word, dept)** — the trained Logistic Regression "
                        f"coefficient linking the word to **{top_dept.title()}** "
                        "(learned from the corpus).\n\n"
                        "A larger bar = the word contributed more to this department."
                    )
                exp_df = pd.DataFrame({
                    "Term": [w for w, _ in r["reasons"]],
                    "Contribution": [round(v, 3) for _, v in r["reasons"]],
                })
                exp_chart = (
                    alt.Chart(exp_df)
                    .mark_bar(color="#2563EB")
                    .encode(
                        x=alt.X("Contribution:Q"),
                        y=alt.Y("Term:N", sort="-x", title=None),
                        tooltip=["Term", "Contribution"],
                    )
                )
                st.altair_chart(exp_chart, use_container_width=True)
            else:
                st.caption("Word-level explanation unavailable "
                           "(run the training script to create explainer_model.joblib).")

        # --- Confirm: nothing is saved into the record until this is clicked ---
        st.markdown("---")
        if r.get("confirmed"):
            st.success(f"✅ Saved & locked — Ticket #{r['ticket']:03d}, "
                       f"{top_dept.title()} (queue position {r['queue_pos']}).")
            st.caption(f"Please keep your ticket number **#{r['ticket']:03d}** and "
                       "wait patiently — you'll be called when it's your turn. "
                       "Thank you for your patience, and we wish you a speedy "
                       "recovery! 💙")
        else:
            if st.button("✅ Confirm & save to record", type="primary"):
                save_record(p, r, top_dept, top_conf)
                st.rerun()
            st.caption("Nothing is saved to the record until you confirm.")

    st.divider()
    # Once locked the patient can't change the record (amend is a staff action).
    # Before confirm they can still freely edit their own description.
    locked = r.get("status") == "ok" and r.get("confirmed")
    if locked:
        if st.button("✅ Done", type="primary"):
            reset_to_checkin()
            st.rerun()
    else:
        if st.button("← Edit symptoms"):
            st.session_state.stage = "intake"
            st.rerun()
