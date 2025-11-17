import os
import re
import json
import base64
import stat
import shutil
import asyncio
import logging
import sys
from typing import List, Optional
from datetime import datetime

import httpx
import git
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ------------------------- Settings -------------------------
class Settings(BaseSettings):
    GEMINI_API_KEY: str = Field("", env="GEMINI_API_KEY")
    GITHUB_TOKEN: str = Field("", env="GITHUB_TOKEN")
    GITHUB_USERNAME: str = Field("", env="GITHUB_USERNAME")
    STUDENT_SECRET: str = Field("", env="STUDENT_SECRET")
    LOG_FILE_PATH: str = Field("logs/app.log", env="LOG_FILE_PATH")
    MAX_CONCURRENT_TASKS: int = Field(2, env="MAX_CONCURRENT_TASKS")
    KEEP_ALIVE_INTERVAL_SECONDS: int = Field(30, env="KEEP_ALIVE_INTERVAL_SECONDS")
    GITHUB_API_BASE: str = Field("https://api.github.com", env="GITHUB_API_BASE")
    GITHUB_PAGES_BASE: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
if not settings.GITHUB_PAGES_BASE:
    settings.GITHUB_PAGES_BASE = f"https://{settings.GITHUB_USERNAME}.github.io"

# ------------------------- Logging -------------------------
os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)
logger = logging.getLogger("task_receiver")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(settings.LOG_FILE_PATH, mode="a", encoding="utf-8")
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(fmt)
file_handler.setFormatter(fmt)
logger.handlers = []
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.propagate = False

def flush_logs():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        for h in logger.handlers:
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass

# ------------------------- Models -------------------------
class Attachment(BaseModel):
    name: str
    url: str

class TaskRequest(BaseModel):
    task: str
    email: str
    round: int
    brief: str
    evaluation_url: str
    nonce: str
    secret: str
    attachments: List[Attachment] = []

# ------------------------- App & Globals -------------------------
app = FastAPI(title="Automated Task Receiver & Processor")
background_tasks_list: List[asyncio.Task] = []
task_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
last_received_task: Optional[dict] = None
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# ------------------------- Utility -------------------------
def verify_secret(secret_from_request: str) -> bool:
    return secret_from_request == settings.STUDENT_SECRET

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def remove_local_path(path: str):
    if not os.path.exists(path):
        return
    def onerror(func, path_arg, exc_info):
        try:
            os.chmod(path_arg, stat.S_IWUSR)
            func(path_arg)
        except Exception as exc:
            logger.exception(f"Failed in rmtree on {path_arg}: {exc}")
            raise
    logger.info(f"[CLEANUP] Removing local directory: {path}")
    shutil.rmtree(path, onerror=onerror)
    flush_logs()

# ------------------------- Attachment helpers -------------------------
def is_image_data_uri(data_uri: str) -> bool:
    if not data_uri or not data_uri.startswith("data:"):
        return False
    return re.search(r"data:image/[^;]+;base64,", data_uri, re.IGNORECASE) is not None

def data_uri_to_gemini_part(data_uri: str) -> Optional[dict]:
    if not data_uri or not data_uri.startswith("data:"):
        return None
    match = re.search(r"data:(?P<mime_type>[^;]+);base64,(?P<base64_data>.*)", data_uri, re.IGNORECASE)
    if not match:
        return None
    mime_type = match.group('mime_type')
    base64_data = match.group('base64_data')
    if not mime_type.startswith("image/"):
        return None
    return {"inlineData": {"data": base64_data, "mimeType": mime_type}}

async def attachment_to_gemini_part(attachment_url: str) -> Optional[dict]:
    if not attachment_url:
        return None
    if attachment_url.startswith("data:"):
        return data_uri_to_gemini_part(attachment_url)
    if attachment_url.startswith(("http://", "https://")):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(attachment_url)
                resp.raise_for_status()
                mime = resp.headers.get("Content-Type", "")
                if not mime.startswith("image/"):
                    logger.info(f"[ATTACHMENT] Skipping non-image MIME: {mime}")
                    return None
                b64 = base64.b64encode(resp.content).decode("utf-8")
                return {"inlineData": {"data": b64, "mimeType": mime}}
        except Exception as e:
            logger.warning(f"[ATTACHMENT] Failed to fetch/encode attachment {attachment_url}: {e}")
            return None
    return None

# ------------------------- Filesystem Save Helpers -------------------------
async def save_generated_files_locally(task_dir: str, files: dict) -> None:
    """Save generated files to the specified task directory."""
    logger.info(f"[LOCAL_SAVE] Saving generated files to: {task_dir}")
    for filename, content in files.items():
        file_path = os.path.join(task_dir, filename)
        
        # Create subdirectories if needed (e.g., .github/workflows/)
        file_dir = os.path.dirname(file_path)
        if file_dir and file_dir != task_dir:
            safe_makedirs(file_dir)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"   -> Saved: {filename} (bytes: {len(content)})")
        except Exception as e:
            logger.exception(f"Failed to save generated file {filename}: {e}")
            raise
    flush_logs()

async def save_attachments_locally(task_dir: str, attachments: List[Attachment]) -> List[str]:
    """Save attachments to task directory - works for ALL rounds"""
    saved_files = []
    if not attachments:
        logger.info("[ATTACHMENTS] No attachments to save")
        return saved_files
        
    logger.info(f"[ATTACHMENTS] Processing {len(attachments)} attachments for {task_dir}")
    async with httpx.AsyncClient(timeout=30) as client:
        for attachment in attachments:
            filename = attachment.name
            url = attachment.url
            file_bytes = None
            if not filename or not url:
                logger.warning(f"Skipping invalid attachment entry: {filename}")
                continue
            try:
                if url.startswith("data:"):
                    m = re.search(r"base64,(.*)", url, re.IGNORECASE | re.DOTALL)
                    if m:
                        file_bytes = base64.b64decode(m.group(1))
                elif url.startswith(("http://", "https://")):
                    resp = await client.get(url)
                    resp.raise_for_status()
                    file_bytes = resp.content
                if file_bytes is None:
                    logger.warning(f"No content for attachment: {filename}")
                    continue
                file_path = os.path.join(task_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(file_bytes)
                logger.info(f"   -> Saved Attachment: {filename} (bytes: {len(file_bytes)})")
                saved_files.append(filename)
            except Exception as e:
                logger.exception(f"Failed to save attachment {filename}: {e}")
    flush_logs()
    return saved_files

# ------------------------- GitHub helpers -------------------------
async def setup_local_repo(local_path: str, repo_name: str, repo_url_auth: str, repo_url_http: str, round_index: int) -> git.Repo:
    github_token = settings.GITHUB_TOKEN
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    async with httpx.AsyncClient(timeout=45) as client:
        try:
            if round_index == 1:
                logger.info(f"[GIT] R1: Creating remote repo '{repo_name}'")
                payload = {"name": repo_name, "private": False, "auto_init": True}
                resp = await client.post(f"{settings.GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
                resp.raise_for_status()
                repo = git.Repo.init(local_path)
                repo.create_remote('origin', repo_url_auth)
                logger.info("[GIT] Local repo initialized")
            else:
                logger.info(f"[GIT] R{round_index}: Cloning {repo_url_http}")
                repo = git.Repo.clone_from(repo_url_auth, local_path)
                logger.info("[GIT] Cloned repo")
            flush_logs()
            return repo
        except httpx.HTTPStatusError as e:
            logger.exception(f"GitHub API error: {getattr(e.response, 'text', '')}")
            raise
        except git.GitCommandError as e:
            logger.exception(f"Git command error: {e}")
            raise

async def commit_and_publish(repo: git.Repo, task_id: str, round_index: int, repo_name: str) -> dict:
    github_username = settings.GITHUB_USERNAME
    github_token = settings.GITHUB_TOKEN
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    repo_url_http = f"https://github.com/{github_username}/{repo_name}"
    async with httpx.AsyncClient(timeout=45) as client:
        try:
            repo.git.add(A=True)
            commit_message = f"Task {task_id} - Round {round_index}: automated update"
            repo.index.commit(commit_message)
            commit_sha = repo.head.object.hexsha
            logger.info(f"[GIT] Committed: {commit_sha}")
            repo.git.branch('-M', 'main')
            repo.git.push('--set-upstream', 'origin', 'main', force=True)
            logger.info("[GIT] Pushed to origin/main")

            # Configure GitHub Pages with retries
            pages_api_url = f"{settings.GITHUB_API_BASE}/repos/{github_username}/{repo_name}/pages"
            pages_payload = {"source": {"branch": "main", "path": "/"}}
            pages_max_retries = 5
            pages_base_delay = 3
            for attempt in range(pages_max_retries):
                try:
                    pages_response = await client.get(pages_api_url, headers=headers)
                    is_configured = (pages_response.status_code == 200)
                    if is_configured:
                        await client.put(pages_api_url, json=pages_payload, headers=headers)
                    else:
                        await client.post(pages_api_url, json=pages_payload, headers=headers)
                    logger.info("[GIT] Pages configured")
                    break
                except httpx.HTTPStatusError as e:
                    text = getattr(e.response, "text", "")
                    if e.response.status_code == 422 and "main branch must exist" in text and attempt < pages_max_retries - 1:
                        delay = pages_base_delay * (2 ** attempt)
                        logger.warning(f"[GIT] Pages timing issue, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue
                    logger.exception(f"[GIT] Pages configuration failed: {text}")
                    raise

            await asyncio.sleep(5)
            pages_url = f"{settings.GITHUB_PAGES_BASE}/{repo_name}/"
            flush_logs()
            return {"repo_url": repo_url_http, "commit_sha": commit_sha, "pages_url": pages_url}
        except git.GitCommandError as e:
            logger.exception("Git operation failed during deployment.")
            raise
        except httpx.HTTPStatusError as e:
            logger.exception("GitHub API error during deployment.")
            raise

# ------------------------- Gemini / LLM helpers -------------------------
async def call_gemini_api(contents: list, system_prompt: str, response_schema: dict, max_retries: int = 3, timeout: int = 60) -> dict:
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }
    base_delay = 1
    for attempt in range(max_retries):
        try:
            if not settings.GEMINI_API_KEY:
                raise Exception("GEMINI_API_KEY not configured.")
            url = f"{GEMINI_API_URL}?key={settings.GEMINI_API_KEY}"
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                resp.raise_for_status()
                result = resp.json()
                candidates = result.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidates in LLM response")
                content_parts = candidates[0].get("content", {}).get("parts", [])
                if not content_parts:
                    raise ValueError("No content parts in candidate")
                json_text = content_parts[0].get("text")
                return json.loads(json_text)
        except httpx.HTTPStatusError as e:
            logger.warning(f"[GEMINI] HTTP error attempt {attempt+1}: {e}")
        except (httpx.RequestError, KeyError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[GEMINI] Processing error attempt {attempt+1}: {e}")
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.info(f"[GEMINI] Retrying in {delay}s...")
            await asyncio.sleep(delay)
    raise Exception("LLM generation failed after retries")

# ------------------------- Round 2 surgical update -------------------------
async def call_llm_round2_surgical_update(task_id: str, brief: str, existing_files: dict) -> dict:
    """
    Perform surgical update on existing files.
    existing_files: dict with keys like 'index.html', 'README.md', 'LICENSE', etc.
    """
    system_prompt = (
        "You are an expert full-stack engineer performing PRECISE, SURGICAL updates. "
        "CRITICAL RULES:\n"
        "1. Apply ONLY the specific changes requested in the brief\n"
        "2. PRESERVE all existing functionality, scripts, event handlers, and core logic\n"
        "3. If the brief mentions adding features, ADD them WITHOUT removing existing features\n"
        "4. Return ALL files that exist, even if unchanged\n"
        "5. For files not mentioned in the brief, return them EXACTLY as provided\n"
        "6. Ensure all generated files are complete and functional\n\n"
        "Return a JSON object with keys matching the file names provided."
    )
    
    # Build the file list for the schema
    file_properties = {}
    for filename in existing_files.keys():
        file_properties[filename] = {"type": "STRING"}
    
    response_schema = {
        "type": "OBJECT",
        "properties": file_properties,
        "required": list(existing_files.keys())
    }
    
    # Build prompt with all existing files
    prompt_parts = [f"SURGICAL UPDATE REQUEST:\n\nBrief: {brief}\n\n"]
    prompt_parts.append("EXISTING FILES:\n")
    for filename, content in existing_files.items():
        prompt_parts.append(f"--- {filename} ---\n{content}\n--- END {filename} ---\n\n")
    
    prompt_parts.append(
        "Instructions:\n"
        "1. Apply the requested changes carefully\n"
        "2. Preserve ALL existing functionality\n"
        "3. Return complete, functional files - not diffs or snippets\n"
        "4. For files not affected by the brief, return them unchanged\n"
    )
    
    prompt = "".join(prompt_parts)
    contents = [{"parts": [{"text": prompt}]}]

    try:
        result = await call_gemini_api(
            contents=contents, 
            system_prompt=system_prompt, 
            response_schema=response_schema, 
            max_retries=4, 
            timeout=120
        )
    except Exception as e:
        logger.exception(f"[ROUND2] LLM call failed: {e}")
        # Fallback: return existing files unchanged
        logger.warning("[ROUND2] Returning existing files due to LLM failure")
        return existing_files.copy()

    # Validate and sanitize results
    for filename in existing_files.keys():
        new_content = (result.get(filename) or "").strip()
        
        # Basic validation - ensure content isn't empty or suspiciously small
        if not new_content or len(new_content) < 50:
            logger.warning(f"[SAFE] LLM returned invalid/empty {filename} — reverting to existing.")
            result[filename] = existing_files[filename]
            continue
        
        # For HTML files, check for basic structure
        if filename.endswith('.html'):
            if not re.search(r'<html|<!DOCTYPE', new_content, re.IGNORECASE):
                logger.warning(f"[SAFE] {filename} missing HTML structure — reverting.")
                result[filename] = existing_files[filename]
                continue
            
            # Check for dramatic size reduction (less than 40% of original)
            orig_len = len(existing_files[filename])
            new_len = len(new_content)
            if orig_len > 500 and new_len < int(orig_len * 0.4):
                logger.warning(f"[SAFE] {filename} suspiciously small ({new_len} vs {orig_len}) — reverting.")
                result[filename] = existing_files[filename]
    
    return result

# ------------------------- Notifier -------------------------
async def notify_evaluation_server(evaluation_url: str, email: str, task_id: str, round_index: int, nonce: str, repo_url: str, commit_sha: str, pages_url: str) -> bool:
    payload = {
        "email": email,
        "task": task_id,
        "round": round_index,
        "nonce": nonce,
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "pages_url": pages_url
    }
    max_retries = 3
    base_delay = 1
    logger.info(f"[NOTIFY] Notifying evaluation server at {evaluation_url}")
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(evaluation_url, json=payload)
                resp.raise_for_status()
                logger.info(f"[NOTIFY] Notification succeeded: {resp.status_code}")
                flush_logs()
                return True
        except httpx.HTTPStatusError as e:
            logger.warning(f"[NOTIFY] HTTP error attempt {attempt+1}: {e}")
        except httpx.RequestError as e:
            logger.warning(f"[NOTIFY] Request error attempt {attempt+1}: {e}")
        if attempt < max_retries - 1:
            await asyncio.sleep(base_delay * (2 ** attempt))
    logger.error("[NOTIFY] Failed to notify evaluation server after retries.")
    flush_logs()
    return False

# ------------------------- Main orchestration -------------------------
async def generate_files_and_deploy(task_data: TaskRequest):
    acquired = False
    try:
        await task_semaphore.acquire()
        acquired = True
        logger.info(f"[PROCESS START] Task: {task_data.task} Round: {task_data.round}")
        flush_logs()

        task_id = task_data.task
        email = task_data.email
        round_index = task_data.round
        brief = task_data.brief
        evaluation_url = task_data.evaluation_url
        nonce = task_data.nonce
        attachments = task_data.attachments or []

        repo_name = task_id.replace(" ", "-").lower()
        github_username = settings.GITHUB_USERNAME
        github_token = settings.GITHUB_TOKEN
        repo_url_auth = f"https://{github_username}:{github_token}@github.com/{github_username}/{repo_name}.git"
        repo_url_http = f"https://github.com/{github_username}/{repo_name}"

        base_dir = os.path.join(os.getcwd(), "generated_tasks")
        local_path = os.path.join(base_dir, task_id)

        # Cleanup local path
        if os.path.exists(local_path):
            try:
                remove_local_path(local_path)
            except Exception as e:
                logger.exception(f"Cleanup failed for {local_path}: {e}")
                raise
        safe_makedirs(local_path)

        # Setup repo (init or clone)
        repo = await setup_local_repo(local_path, repo_name, repo_url_auth, repo_url_http, round_index)

        # Build attachment descriptions
        attachment_descriptions = ""
        if attachments:
            attachment_descriptions = "\n\nATTACHMENTS PROVIDED:\n"
            for att in attachments:
                attachment_descriptions += f"- {att.name}\n"
            attachment_descriptions += (
                "\nCRITICAL: You MUST create these files with appropriate content. "
                "Reference them using their exact names (e.g., <img src='filename.png'>). "
                "Generate realistic content for each file type:\n"
                "- .txt files: Generate relevant text content\n"
                "- .json files: Generate valid JSON with appropriate structure\n"
                "- .csv files: Generate CSV data with headers and rows\n"
                "- .svg files: Generate valid SVG markup\n"
                "- .md files: Generate markdown content\n"
            )

        # --- Round 1: Full generation ---
        if round_index == 1:
            logger.info("[WORKFLOW] Round 1: full generation")

            # Prepare image parts for LLM
            image_parts = []
            for attachment in attachments:
                part = await attachment_to_gemini_part(attachment.url)
                if part:
                    image_parts.append(part)

            enriched_brief = f"{brief}{attachment_descriptions}".strip()

            system_prompt = (
                "You are an expert full-stack engineer. Create a complete, production-ready application.\n\n"
                "CRITICAL REQUIREMENTS:\n"
                "1. Generate ALL files mentioned in the attachments list - create realistic content for each\n"
                "2. For data files (.json, .csv, .txt), generate appropriate realistic data\n"
                "3. For SVG files, create valid SVG markup with the described imagery\n"
                "4. index.html: Self-contained, responsive HTML using Tailwind CSS via CDN\n"
                "5. Reference attachments exactly as: <img src='filename'> or fetch('filename')\n"
                "6. README.md: Professional documentation with setup, usage, and features\n"
                "7. LICENSE: Complete MIT license text with current year\n"
                "8. For tasks requiring CI/CD, create .github/workflows/ci.yml with appropriate steps\n"
                "9. For Python tasks, create execute.py and requirements.txt if needed\n\n"
                "Return a JSON object with keys for EVERY file needed (index.html, README.md, LICENSE, "
                "and ALL attachment files mentioned, plus any CI/CD files if applicable)."
            )

            # Build schema dynamically based on attachments
            file_properties = {
                "index.html": {"type": "STRING"},
                "README.md": {"type": "STRING"},
                "LICENSE": {"type": "STRING"},
            }
            
            # Add attachment files to schema
            for att in attachments:
                file_properties[att.name] = {"type": "STRING"}
            
            # Check if task requires CI/CD (contains "analyze", "ci", "workflow", etc.)
            task_lower = task_id.lower()
            brief_lower = brief.lower()
            if any(keyword in task_lower or keyword in brief_lower for keyword in ['analyze', 'ci', 'workflow', 'github actions', 'execute.py']):
                file_properties[".github/workflows/ci.yml"] = {"type": "STRING"}
                if 'python' in brief_lower or 'execute.py' in brief_lower:
                    file_properties["execute.py"] = {"type": "STRING"}
                    file_properties["requirements.txt"] = {"type": "STRING"}

            response_schema = {
                "type": "OBJECT",
                "properties": file_properties,
                "required": ["index.html", "README.md", "LICENSE"],
            }

            contents = []
            if image_parts:
                contents.append({"parts": image_parts + [{"text": enriched_brief}]})
            else:
                contents.append({"parts": [{"text": enriched_brief}]})

            generated = await call_gemini_api(
                contents=contents,
                system_prompt=system_prompt,
                response_schema=response_schema,
                max_retries=4,
                timeout=120,
            )
            
            # Ensure all required files exist
            for att in attachments:
                if att.name not in generated or not generated[att.name].strip():
                    logger.warning(f"[ROUND1] LLM didn't generate {att.name}, will save from attachment")

        # --- Round 2+: Surgical Update ---
        else:
            logger.info(f"[WORKFLOW] Round {round_index}: surgical update")
            
            # Read ALL existing files from repo
            existing_files = {}
            
            # Walk the entire repo to find all files
            for root, dirs, files in os.walk(local_path):
                # Skip .git directory
                if '.git' in root:
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, local_path)
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            existing_files[relative_path] = content
                            logger.info(f"[WORKFLOW] Read existing {relative_path} ({len(content)} bytes)")
                    except Exception as e:
                        logger.warning(f"[WORKFLOW] Could not read {relative_path}: {e}")

            if not existing_files:
                logger.error("[WORKFLOW] No existing files found in Round 2+! This is unexpected.")
                # Fallback to treating as Round 1
                raise Exception("No existing files found for Round 2+ update")

            brief_with_attachments = f"{brief}{attachment_descriptions}".strip()
            
            generated = await call_llm_round2_surgical_update(
                task_id=task_id,
                brief=brief_with_attachments,
                existing_files=existing_files
            )

        # Save generated files (this will create subdirectories as needed)
        await save_generated_files_locally(local_path, generated)

        # ALWAYS save attachments (for ALL rounds)
        if attachments:
            saved_attachment_files = await save_attachments_locally(local_path, attachments)
            logger.info(f"[ATTACHMENTS] Saved {len(saved_attachment_files)} attachment files")

        # Commit and publish
        deployment_info = await commit_and_publish(repo, task_id, round_index, repo_name)

        # Notify evaluation server
        await notify_evaluation_server(
            evaluation_url=evaluation_url,
            email=email,
            task_id=task_id,
            round_index=round_index,
            nonce=nonce,
            repo_url=deployment_info["repo_url"],
            commit_sha=deployment_info["commit_sha"],
            pages_url=deployment_info["pages_url"],
        )

        logger.info(f"[DEPLOYMENT] Success. Repo: {deployment_info['repo_url']} Pages: {deployment_info['pages_url']}")

    except Exception as exc:
        logger.exception(f"[CRITICAL FAILURE] Task {getattr(task_data, 'task', 'unknown')} failed: {exc}")
    finally:
        if acquired:
            task_semaphore.release()
        flush_logs()
        logger.info(
            f"[PROCESS END] Task: {getattr(task_data, 'task', 'unknown')} Round: {getattr(task_data, 'round', 'unknown')}"
        )


# ------------------------- Endpoint handlers -------------------------
def _task_done_callback(task: asyncio.Task):
    try:
        exc = task.exception()
        if exc:
            logger.error(f"[BACKGROUND TASK] Task finished with exception: {exc}")
            logger.exception(exc)
        else:
            logger.info("[BACKGROUND TASK] Task finished successfully.")
    except asyncio.CancelledError:
        logger.warning("[BACKGROUND TASK] Task was cancelled.")
    finally:
        flush_logs()

@app.post("/ready", status_code=200)
async def receive_task(task_data: TaskRequest, request: Request):
    global last_received_task, background_tasks_list
    if not verify_secret(task_data.secret):
        logger.warning(f"Unauthorized attempt for task {task_data.task} from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(status_code=401, detail="Unauthorized: Secret mismatch")

    last_received_task = {
        "task": task_data.task,
        "email": task_data.email,
        "round": task_data.round,
        "brief": (task_data.brief[:250] + "...") if len(task_data.brief) > 250 else task_data.brief,
        "time": datetime.utcnow().isoformat() + "Z"
    }

    bg_task = asyncio.create_task(generate_files_and_deploy(task_data))
    bg_task.add_done_callback(_task_done_callback)
    background_tasks_list.append(bg_task)

    logger.info(f"Received task {task_data.task}. Background processing started.")
    flush_logs()

    return JSONResponse(status_code=200, content={"status": "ready", "message": f"Task {task_data.task} received and processing started."})

@app.get("/")
async def root():
    return {"message": "Task Receiver Service running. POST /ready to submit."}

@app.get("/status")
async def get_status():
    global last_received_task, background_tasks_list
    if last_received_task:
        background_tasks_list[:] = [t for t in background_tasks_list if not t.done()]
        return {"last_received_task": last_received_task, "running_background_tasks": len(background_tasks_list)}
    return {"message": "Awaiting first task submission to /ready"}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.get("/logs")
async def get_logs(lines: int = Query(200, ge=1, le=5000)):
    path = settings.LOG_FILE_PATH
    if not os.path.exists(path):
        return PlainTextResponse("Log file not found.", status_code=404)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            buffer = bytearray()
            block_size = 1024
            blocks = 0
            while file_size > 0 and len(buffer) < lines * 2000 and blocks < 1024:
                read_size = min(block_size, file_size)
                f.seek(file_size - read_size)
                buffer.extend(f.read(read_size))
                file_size -= read_size
                blocks += 1
            text = buffer.decode(errors="ignore").splitlines()
            last_lines = "\n".join(text[-lines:])
            return PlainTextResponse(last_lines)
    except Exception as e:
        logger.exception(f"Error reading log file: {e}")
        return PlainTextResponse(f"Error reading log file: {e}", status_code=500)

# ------------------------- Startup / Shutdown -------------------------
@app.on_event("startup")
async def startup_event():
    async def keep_alive():
        while True:
            try:
                logger.info("[KEEPALIVE] Service heartbeat")
                flush_logs()
            except Exception:
                pass
            await asyncio.sleep(settings.KEEP_ALIVE_INTERVAL_SECONDS)
    asyncio.create_task(keep_alive())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[SHUTDOWN] Waiting for background tasks to finish (graceful shutdown)...")
    for t in background_tasks_list:
        if not t.done():
            try:
                t.cancel()
            except Exception:
                pass
    await asyncio.sleep(0.5)
    flush_logs()
