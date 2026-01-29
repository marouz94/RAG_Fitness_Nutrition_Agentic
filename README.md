# Intelligent Fitness & Nutrition Coach Agent

## Project Overview

This application implements an intelligent agent capable of generating **personalized fitness and nutrition plans** based on a user’s profile.  
The system is built with **Streamlit** and relies on advanced LLM reasoning techniques to go beyond a simple conversational chatbot.

Users provide personal information (age, height, weight), fitness goals, training level, dietary preferences, constraints (injuries, time, equipment), and the agent generates:
- a coherent global strategy,
- a detailed training plan,
- a personalized nutrition plan,
- with the possibility to refine the plan dynamically.

The application is designed to be both **interactive** and **explainable**, making the reasoning behind recommendations explicit.

---

## Reasoning Approach

The agent relies on several advanced reasoning techniques:

- **Tree of Thoughts (ToT)**  
  Used to internally explore multiple possible strategies before selecting the most appropriate global approach for the user.

- **ReAct (Reason + Act)**  
  The agent follows structured reasoning steps to analyze the user profile, apply domain logic, and generate a consistent plan.

- **Self-Correction**  
  The generated plans are reviewed and refined by the agent itself to improve coherence, safety, and realism.

---

## Application Modes

The application provides **two distinct execution modes**, selectable from the sidebar.

### 1. Demo Mode (Offline)

Demo mode allows the application to run **without any external API calls**.  
In this mode, the app loads **precomputed scenarios** from a local dataset (`demo_data.json`).

Characteristics:
- Fully offline
- No OpenAI API key required
- Instant results
- Ideal for demonstrations, testing, or when the API key is unavailable

Each demo scenario contains:
- a predefined user profile,
- a generated global strategy,
- an evaluation of the strategy,
- a final nutrition and training plan.

This mode ensures the application remains functional and demonstrable in all conditions.

---

### 2. Live Mode (OpenAI API)

Live mode uses the **OpenAI API** to generate results dynamically.

Characteristics:
- Requires a valid `OPENAI_API_KEY`
- Plans are generated in real time
- Adapts fully to the user’s specific inputs
- Demonstrates real agent reasoning and LLM capabilities

In this mode, the agent:
1. Analyzes the user profile
2. Generates a global strategy
3. Evaluates and refines the strategy
4. Produces a detailed training and nutrition plan

---

## How to Run the Application

### 1. Install dependencies
```bash
pip install -r requirements.txt
