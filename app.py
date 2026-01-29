import os
import json
from typing import Dict, Any, Tuple
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Désactiver éventuellement le tracing LangSmith si ton env l'active
os.environ["LANGCHAIN_TRACING_V2"] = "false"

load_dotenv()


#  LLM FACTORY
def get_llm():
    """Return a ChatOpenAI instance configured for our app."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
    )

DEMO_FILE = Path("demo_data.json")

def load_demo_data():
    if not DEMO_FILE.exists():
        return []
    with open(DEMO_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


#  UTILITIES
def format_user_profile(profile: Dict[str, Any]) -> str:
    """Format the user profile as a readable text block for prompts."""
    return (
        f"Age: {profile['age']}\n"
        f"Gender: {profile['gender']}\n"
        f"Height (cm): {profile['height_cm']}\n"
        f"Weight (kg): {profile['weight_kg']}\n"
        f"Training level: {profile['training_level']}\n"
        f"Main goal: {profile['goal']}\n"
        f"Dietary preferences or restrictions: {profile['diet_prefs']}\n"
        f"Weekly available time for training (hours): {profile['hours_per_week']}\n"
        f"Available equipment: {profile['equipment']}\n"
        f"Injuries or constraints: {profile['constraints']}\n"
    )


# --------- STEP 1: TREE OF THOUGHTS – GENERATE STRATEGIES ---------
def generate_strategies(profile: Dict[str, Any]) -> str:
    """
    Use a Tree-of-Thoughts style prompt:
    Generate several distinct strategies (A, B, C) for this user.
    We return a markdown text block with Strategy A/B/C.
    """
    llm = get_llm()
    profile_text = format_user_profile(profile)

    prompt = f"""
You are an expert fitness and nutrition coach.

You receive this user profile:

{profile_text}

Your task is to think in a Tree-of-Thoughts manner and propose EXACTLY three distinct strategies:

- Strategy A
- Strategy B
- Strategy C

Each strategy must include:
1. A short name (1 line)
2. A high-level description (3–5 lines)
3. A training philosophy
4. A nutrition philosophy
5. Pros
6. Cons

IMPORTANT:
- Write your answer in English.
- Use clear markdown headings:
  ## Strategy A
  ...
  ## Strategy B
  ...
  ## Strategy C
    """

    response = llm.invoke(prompt)
    return response.content


#  STEP 2: EVALUATE & SELECT BEST STRATEGY
def evaluate_strategies(
    profile: Dict[str, Any], strategies_markdown: str
) -> Tuple[str, str]:
    """
    Ask the LLM to evaluate the three strategies and pick the best one
    (A, B, or C), with an explanation.
    Returns:
        best_label: "A" / "B" / "C"
        reasoning: explanation text
    """
    llm = get_llm()
    profile_text = format_user_profile(profile)

    prompt = f"""
You are now a critical evaluator.

User profile:
{profile_text}

Candidate strategies (markdown):
{strategies_markdown}

1. Carefully read all three strategies.
2. Evaluate them according to:
   - Safety (no excessive volume or intensity)
   - Feasibility (time, equipment, constraints)
   - Relevance to the user's goal
3. Choose the best strategy among A, B, and C.
4. Explain your reasoning briefly.

IMPORTANT:
- At the end of your answer, write on a separate line:
  BEST_STRATEGY: A
  or B or C.
- Answer in English.
    """

    response = llm.invoke(prompt)
    text = response.content

    # Extract BEST_STRATEGY letter (very simple parsing)
    best_label = "A"
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("BEST_STRATEGY"):
            # e.g. "BEST_STRATEGY: B"
            parts = line.split(":")
            if len(parts) >= 2:
                candidate = parts[1].strip().upper()
                if candidate in {"A", "B", "C"}:
                    best_label = candidate
            break

    reasoning = text
    return best_label, reasoning


#  STEP 3: GENERATE DETAILED PLAN (TRAINING + NUTRITION)
def generate_detailed_plan(
    profile: Dict[str, Any],
    strategies_markdown: str,
    best_label: str,
) -> Dict[str, str]:
    """
    Generate a detailed training + nutrition plan based on the selected strategy.
    Returns a dict with keys 'nutrition' and 'training'.
    """
    llm = get_llm()
    profile_text = format_user_profile(profile)

    prompt = f"""
You are now asked to build a concrete plan.

User profile:
{profile_text}

Here are the 3 candidate strategies (markdown):
{strategies_markdown}

The evaluator has selected Strategy {best_label} as the best one.

Your task:
1. Focus ONLY on Strategy {best_label}.
2. Generate:
   - A detailed nutrition plan (as text)
   - A detailed training plan (as text)
3. The plan must be realistic, progressive, and safe for the user.
4. Do NOT give medical advice, only general fitness and nutrition guidance.
5. Answer in English.

Format your answer in markdown with two main sections:
## Nutrition Plan
...
## Training Plan
...
    """

    response = llm.invoke(prompt)
    content = response.content

    # We'll keep it as markdown and split roughly into two parts
    nutrition_part = ""
    training_part = ""

    current = None
    lines = []
    for line in content.splitlines():
        if line.strip().lower().startswith("## nutrition"):
            current = "nutrition"
            continue
        elif line.strip().lower().startswith("## training"):
            current = "training"
            continue

        if current == "nutrition":
            nutrition_part += line + "\n"
        elif current == "training":
            training_part += line + "\n"
        else:
            # before first heading, ignore or append to nothing
            continue

    return {
        "nutrition": nutrition_part.strip(),
        "training": training_part.strip(),
    }


#  STEP 4: SELF-CORRECTION
def self_correct_plan(
    profile: Dict[str, Any],
    raw_plan: Dict[str, str],
) -> Dict[str, str]:
    """
    Ask the LLM to self-critique and improve the plan.
    Returns a new dict (same structure) with an improved version.
    """
    llm = get_llm()
    profile_text = format_user_profile(profile)

    prompt = f"""
You are an expert coach reviewing a fitness and nutrition plan.

User profile:
{profile_text}

Here is the current Nutrition Plan:
{raw_plan['nutrition']}

Here is the current Training Plan:
{raw_plan['training']}

Your tasks:
1. Critically review the plans:
   - Are they safe for the user?
   - Are they realistic with the user's constraints?
   - Are they coherent with the user's main goal?
2. Fix any issue you detect (too intense, not progressive, unrealistic, etc.).
3. Produce an improved, safer and more coherent version of BOTH plans.
4. Do NOT provide medical advice, only general fitness and nutrition guidance.
5. Answer in English.

Format your answer in markdown with:
## Improved Nutrition Plan
...
## Improved Training Plan
...
Also add a short section at the end:
## Reviewer Notes
(brief bullet points).
    """

    response = llm.invoke(prompt)
    content = response.content

    improved_nutrition = ""
    improved_training = ""
    reviewer_notes = ""

    current = None
    for line in content.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("## improved nutrition"):
            current = "nutrition"
            continue
        elif stripped.startswith("## improved training"):
            current = "training"
            continue
        elif stripped.startswith("## reviewer"):
            current = "notes"
            continue

        if current == "nutrition":
            improved_nutrition += line + "\n"
        elif current == "training":
            improved_training += line + "\n"
        elif current == "notes":
            reviewer_notes += line + "\n"

    return {
        "nutrition": improved_nutrition.strip(),
        "training": improved_training.strip(),
        "notes": reviewer_notes.strip(),
    }


#  STREAMLIT APP
def main():
    st.set_page_config(
        page_title="Intelligent Fitness & Nutrition Coach",
        page_icon="🏋️‍♀️",
        layout="wide",
    )

    st.title("🏋️‍♀️ Intelligent Fitness & Nutrition Coach")

    # ---------------- SIDEBAR : MODE SELECTION ----------------
    st.sidebar.header("Mode")
    mode = st.sidebar.radio(
        "Select mode",
        ["Demo mode (offline)", "Live mode (OpenAI API)"],
        index=0 if not has_openai_key() else 1
    )

    if mode == "Live mode (OpenAI API)" and not has_openai_key():
        st.sidebar.error("OPENAI_API_KEY missing or expired. Please use Demo mode.")
        mode = "Demo mode (offline)"

    st.write(
        "This app uses advanced LLM reasoning (Tree-of-Thoughts, ReAct-style steps, "
        "and Self-Correction) to build a personalized training and nutrition plan."
    )

    # ---------------- SESSION STATE INIT ----------------
    for key in [
        "user_profile",
        "strategies_markdown",
        "best_label",
        "evaluation_text",
        "raw_plan",
        "final_plan",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    tab_profile, tab_plan, tab_summary = st.tabs(
        ["User Profile", "Generated Plan", "Summary / Export"]
    )

    # ================= TAB 1: USER PROFILE =================
    with tab_profile:
        st.header("User Profile")

        with st.form("user_profile_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", min_value=16, max_value=90, value=25)
                gender = st.selectbox("Gender", ["Not specified", "Male", "Female"])

            with col2:
                height_cm = st.number_input("Height (cm)", min_value=130, max_value=220, value=175)
                weight_kg = st.number_input("Weight (kg)", min_value=40.0, max_value=200.0, value=70.0)

            with col3:
                training_level = st.selectbox(
                    "Training level",
                    ["Beginner", "Intermediate", "Advanced"],
                )

            goal = st.selectbox(
                "Main goal",
                ["Weight loss", "Muscle gain", "Body recomposition", "Cutting", "General health"],
            )

            diet_prefs = st.text_area(
                "Dietary preferences / restrictions",
                "e.g. vegetarian, halal, lactose-free, allergies...",
            )

            hours_per_week = st.slider(
                "Available training time per week (hours)",
                min_value=1,
                max_value=14,
                value=4,
            )

            equipment = st.text_area(
                "Available equipment",
                "e.g. none, resistance bands, dumbbells, full gym...",
            )

            constraints = st.text_area(
                "Injuries / constraints",
                "e.g. knee pain, lower back issues, no jumping...",
            )

            submitted = st.form_submit_button("Generate my plan 💪")

        # ================= FORM SUBMISSION =================
        if submitted:
            user_profile = {
                "age": age,
                "gender": gender,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "training_level": training_level,
                "goal": goal,
                "diet_prefs": diet_prefs,
                "hours_per_week": hours_per_week,
                "equipment": equipment,
                "constraints": constraints,
            }
            st.session_state.user_profile = user_profile

            # ---------- DEMO MODE ----------
            if mode == "Demo mode (offline)":
                demo_data = load_demo_data()

                if not demo_data:
                    st.error("Demo data not found or empty (demo_data.json).")
                else:
                    chosen = demo_data[0]  # simple default demo

                    st.session_state.strategies_markdown = chosen.get("strategy")
                    st.session_state.evaluation_text = chosen.get("evaluation")
                    st.session_state.best_label = "A"
                    st.session_state.raw_plan = {
                        "nutrition": chosen["final_plan"]["nutrition"],
                        "training": chosen["final_plan"]["training"],
                    }
                    st.session_state.final_plan = chosen["final_plan"]

                    st.success(f"Demo plan loaded: {chosen.get('name', 'Demo scenario')}")

            # ---------- LIVE MODE ----------
            else:
                with st.spinner("Generating global strategy..."):
                    strategies_md = generate_strategies(user_profile)
                st.session_state.strategies_markdown = strategies_md

                with st.spinner("Reviewing strategy..."):
                    best_label, eval_text = evaluate_strategies(user_profile, strategies_md)
                st.session_state.best_label = best_label
                st.session_state.evaluation_text = eval_text

                with st.spinner("Building training and nutrition plan..."):
                    raw_plan = generate_detailed_plan(user_profile, strategies_md, best_label)
                st.session_state.raw_plan = raw_plan

                with st.spinner("Self-correcting the plan..."):
                    final_plan = self_correct_plan(user_profile, raw_plan)
                st.session_state.final_plan = final_plan

                st.success("Plan generated! Check the other tabs.")

    # ================= TAB 2: GENERATED PLAN =================
    with tab_plan:
        st.header("Generated Plan")

        if st.session_state.final_plan is None:
            st.info("Please generate a plan first.")
        else:
            st.subheader("Global Strategy")
            st.markdown(st.session_state.strategies_markdown)

            st.subheader("Evaluation")
            st.markdown(st.session_state.evaluation_text)

            st.subheader("Final Nutrition Plan")
            st.markdown(st.session_state.final_plan["nutrition"])

            st.subheader("Final Training Plan")
            st.markdown(st.session_state.final_plan["training"])

            if st.session_state.final_plan.get("notes"):
                st.subheader("Notes")
                st.markdown(st.session_state.final_plan["notes"])

    # ================= TAB 3: SUMMARY / EXPORT =================
    with tab_summary:
        st.header("Summary / Export")

        if st.session_state.final_plan is None:
            st.info("No plan to export yet.")
        else:
            summary_md = (
                "## User Profile\n"
                + format_user_profile(st.session_state.user_profile)
                + "\n\n## Global Strategy\n"
                + st.session_state.strategies_markdown
                + "\n\n## Final Nutrition Plan\n"
                + st.session_state.final_plan["nutrition"]
                + "\n\n## Final Training Plan\n"
                + st.session_state.final_plan["training"]
            )

            st.markdown(summary_md)

            st.download_button(
                label="Download plan as .txt",
                data=summary_md,
                file_name="fitness_nutrition_plan.txt",
                mime="text/plain",
            )



if __name__ == "__main__":
    main()
