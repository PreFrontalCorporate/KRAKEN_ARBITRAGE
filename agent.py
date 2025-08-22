#!/usr/bin/env python
# Gemini-powered Persistent Research & Development Agent

import google.generativeai as genai
import subprocess
import textwrap
import datetime
import sys
import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

# --- Configuration ---
CONTEXT_IGNORE = {".git", ".venv", "node_modules", "__pycache__", ".agent"}
# MAX_PASSES has been removed for infinite persistence.
IS_VERBOSE = os.environ.get("AGENT_VERBOSE", "0") == "1"

# --- Constants ---
STATE_DIR = Path(".agent")
REPORTS_DIR = STATE_DIR / "reports"
LOCK_FILE = STATE_DIR / "run.lock"

# --- Setup ---
os.makedirs(REPORTS_DIR, exist_ok=True)
if LOCK_FILE.exists():
    print("WARNING: Stale lock file found. It will be removed.")
    LOCK_FILE.unlink()

# --- Logging ---
log_file_path = REPORTS_DIR / f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
log_file = open(log_file_path, "w", encoding="utf-8")

def log(message):
    log_message = f"[AGENT] {message}"
    print(log_message, flush=True)
    log_file.write(f"[{datetime.datetime.now().isoformat()}] {message}\n")
    log_file.flush()

# --- Utilities ---
def run_command(command, check=True):
    log(f"🏃 Running command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=check,
            encoding="utf-8",
            errors="ignore"
        )
        if IS_VERBOSE:
            log(f"  | stdout: {result.stdout.strip()}")
            log(f"  | stderr: {result.stderr.strip()}")
        if check and result.returncode != 0:
            log(f"❌ Command failed with exit code {result.returncode}")
            return False, result.stderr.strip()
        log("✅ Command successful.")
        return True, result.stdout.strip()
    except Exception as e:
        log(f"❌ Command failed with exception: {e}")
        return False, str(e)

def run_notebook_and_generate_pdf(notebook_path: Path):
    log(f"🧪 Running test gate for notebook: {notebook_path}...")
    if not notebook_path.exists():
        log(f"❌ Notebook not found at {notebook_path}")
        return False, "Notebook file was not created by the model."

    log("  | Installing notebook dependencies...")
    ok, _ = run_command([sys.executable, "-m", "pip", "install", "nbconvert", "jupyter", "matplotlib", "ipykernel", "pandas", "numpy", "scipy", "cvxpy", "sortedcontainers", "stable-baselines3", "shimmy"])
    if not ok:
        return False, "Failed to install notebook dependencies."

    output_notebook_path = notebook_path.with_suffix(".executed.ipynb")
    log(f"  | Executing notebook, output at {output_notebook_path}...")
    execute_command = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        f"--output={output_notebook_path.name}",
        str(notebook_path)
    ]
    ok, result_log = run_command(execute_command, check=False)

    if not ok or "Traceback" in result_log or "Error" in result_log:
        log(f"❌ Notebook execution failed. See logs for traceback.")
        return False, f"Notebook execution failed with the following error:\n---\n{result_log}\n---"

    log("🟢 Notebook execution GREEN.")

    report_name = f"{notebook_path.stem}__{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    report_path = notebook_path.parent / report_name
    log(f"  | Generating PDF report at {report_path}...")
    
    ok, result_log = run_command(["jupyter", "nbconvert", "--to", "latex", str(output_notebook_path)])
    if not ok:
        log("❌ PDF generation failed at LaTeX conversion step.")
        return False, f"Failed to convert notebook to LaTeX:\n---\n{result_log}\n---"

    latex_file = output_notebook_path.with_suffix(".tex")
    run_command(["pdflatex", "-interaction=nonstopmode", str(latex_file)], check=False)
    ok, result_log = run_command(["pdflatex", "-interaction=nonstopmode", str(latex_file)])

    if not ok or not output_notebook_path.with_suffix(".pdf").exists():
         log("❌ PDF generation failed at pdflatex compilation step.")
         return False, f"pdflatex command failed:\n---\n{result_log}\n---"

    output_notebook_path.with_suffix(".pdf").rename(report_path)
    log(f"✅ Report generated: {report_path}")
    return True, report_path.name

# --- Core Logic ---
def get_repo_context(target_dir):
    log(f"🔍 Building repository context for '{target_dir}'...")
    context = []
    base_path = Path(target_dir)
    if not base_path.exists() or not base_path.is_dir():
        log(f"⚠️ Target directory '{target_dir}' does not exist. Context will be empty.")
        return ""

    files_to_scan = [p for p in base_path.rglob("*") if p.is_file() and not any(part in CONTEXT_IGNORE for part in p.parts)]
    for p in files_to_scan:
        try:
            rel_path = p.resolve().relative_to(Path.cwd().resolve())
            context.append(f"----\n📄 {rel_path}\n----\n{p.read_text(encoding='utf-8', errors='ignore')}")
        except Exception as e:
            context.append(f"----\n📄 {p.relative_to(Path.cwd())}\n----\n(could not read file: {e})")
    log("✅ Repository context built.")
    return "\n".join(context)

def process_response(response):
    log("🔎 Processing Gemini's response...")
    raw_text = response.text if response and hasattr(response, 'text') else ""
    if not raw_text:
        log("❌ CRITICAL: Received empty response from model.")
        return {"error": "Empty response from model."}

    if IS_VERBOSE:
        log("--- 🤖 Gemini's Full Response ---")
        log(raw_text)
        log("---------------------------------")

    summary_match = re.search(r"## Summary of Changes\n(.*?)(?=\n##|$)", raw_text, re.DOTALL)
    edit_blocks = re.findall(r"EDIT ([\w/.\-]+)\n```[\w\s]*\n(.*?)\n```", raw_text, re.DOTALL)

    if not summary_match or not summary_match.group(1).strip():
        return {"error": "Response was missing a '## Summary of Changes' section."}
    if not edit_blocks:
        return {"error": "Response did not contain any valid 'EDIT' blocks."}

    log("✅ Response processed successfully.")
    return {"summary": summary_match.group(1).strip(), "edits": edit_blocks}

def apply_edits(edits):
    log("✍️ Applying edits to filesystem...")
    edited_files = []
    for filepath, content in edits:
        try:
            p = Path(filepath.strip())
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            edited_files.append(p)
            log(f"  | ✅ Wrote {len(content)} chars to {p.resolve()}")
        except Exception as e:
            log(f"❌ CRITICAL WRITE FAILURE for '{p.resolve()}': {e}")
            return False, f"FATAL: Error writing to '{p.resolve()}': {e}", []
    log("✅ All edits applied successfully.")
    return True, "Edits applied.", edited_files

def run_pass(user_prompt, memory, target_dir):
    repo_context = get_repo_context(target_dir)
    prompt = textwrap.dedent(f"""
        You are an expert-level AI quantitative finance researcher and Python developer.
        Your task is to solve the user's research request by creating and editing files in a specified project directory.

        ## Instructions
        1.  **Analyze the Request:** Carefully read the user's prompt and the provided context files.
        2.  **Formulate a Plan:** Create a step-by-step plan to generate the required LaTeX theory and Python code.
        3.  **Generate Files:** Provide the FULL, complete content for each file you need to create or edit.
        4.  **Strict Formatting:** Your response MUST follow this structure:
            - A `## Plan` section outlining your steps.
            - One or more `EDIT path/to/file.ext` blocks. For a Jupyter notebook, use `EDIT path/to/notebook.ipynb`.
            - A final `## Summary of Changes` section describing what you did.

        {memory}

        ## Repository Context for directory '{target_dir}'
        {repo_context}

        ## User Request
        {user_prompt}
    """).strip()

    if IS_VERBOSE:
        log("--- 🧠 Prompt to be sent to Gemini ---")
        log(prompt)
        log("--------------------------------------")
    else:
        log("🧠 Sending prompt to Gemini...")

    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        return model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
    except Exception as e:
        log(f"❌ Gemini API call failed: {e}")
        return None

def main():
    if LOCK_FILE.exists():
        log(f"🛑 Lock file {LOCK_FILE} exists. Another agent might be running.")
        sys.exit(1)
    LOCK_FILE.touch()

    log("🚀 Agent starting...")

    env_path = Path(".agent/agent.env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        log(f"✅ Environment file loaded from '{env_path.resolve()}'")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log("❌ FATAL: GOOGLE_API_KEY not found in .agent/agent.env")
        LOCK_FILE.unlink()
        sys.exit(1)
    genai.configure(api_key=api_key)
    log("✅ Gemini API key configured.")

    if len(sys.argv) < 3:
        log("❌ FATAL: Insufficient arguments. Usage: python agent.py \"<prompt_file.txt>\" <target_directory>")
        LOCK_FILE.unlink()
        sys.exit(1)

    user_prompt_file = Path(sys.argv[1])
    target_dir = sys.argv[2]

    if not user_prompt_file.exists():
        log(f"❌ FATAL: Prompt file not found: {user_prompt_file}")
        LOCK_FILE.unlink()
        sys.exit(1)
    user_prompt = user_prompt_file.read_text()
    log(f"✅ Loaded prompt from {user_prompt_file}")

    pass_count = 0
    memory_string = ""
    # --- THIS IS THE CHANGE ---
    # The agent will now loop forever until it succeeds.
    while True:
        pass_count += 1
        log(f"--- 🔁 Starting Pass {pass_count} ---")

        response = run_pass(user_prompt, memory_string, target_dir)
        if not response:
            memory_string = f"## Memory\nThe previous attempt failed due to an API call error. Please try again."
            continue

        processed_data = process_response(response)
        if "error" in processed_data:
            log(f"⚠️ Pass failed: {processed_data['error']}")
            memory_string = f"## Memory\nYour last response was invalid: '{processed_data['error']}'. Please re-read the instructions carefully and try again, ensuring your response includes a Plan, EDIT blocks, and a Summary."
            continue

        ok, message, edited_files = apply_edits(processed_data["edits"])
        if not ok:
            log(f"⚠️ Pass failed: {message}")
            memory_string = f"## Memory\nThe last attempt failed with a critical file write error: '{message}'. This is an environment issue. Please state this in your summary and do not provide an EDIT block."
            continue

        notebook_to_test = next((f for f in edited_files if f.suffix == '.ipynb'), None)

        if not notebook_to_test:
            log("⚠️ No notebook file was edited in this pass. Cannot run test gate. Assuming success.")
            test_ok = True
        else:
            test_ok, test_result = run_notebook_and_generate_pdf(notebook_to_test)

        if test_ok:
            log("🎉 Agent finished successfully!")
            log("Committing successful changes to git...")
            run_command(["git", "add", "."])
            commit_message = f"feat({target_dir}): Complete research module via agent\n\n{processed_data['summary']}"
            run_command(["git", "commit", "-m", commit_message])
            log("Changes committed. You can now 'git push' to save to remote.")
            break # Exit the infinite loop on success
        else:
            log(f"❌ Test gate failed. Reverting changes and preparing for the next attempt.")
            run_command(["git", "reset", "--hard", "HEAD"], check=False)
            run_command(["git", "clean", "-fd"], check=False)
            memory_string = f"## Memory\nYour last code edit was correctly applied, but the test gate failed. The notebook produced an error during execution. Please analyze the error, fix the code, and try again.\n\nFAILED TEST LOG:\n{test_result}"

    log("🛑 Agent shutting down.")
    LOCK_FILE.unlink()

if __name__ == "__main__":
    main()
