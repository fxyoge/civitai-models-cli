import os
import sys
import json
import httpx
import shutil
import tempfile
import typer
import time
import yaml
import html2text as html2text_module
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
from tqdm import tqdm
from rich.console import Console
from .helpers import feedback_message, get_model_folder, create_table
from .details import get_model_details

__all__ = ["download_model_cli"]

console = Console()

MAX_RETRIES = 3
TIMEOUT = 30  # seconds
MAX_CONCURRENT_DOWNLOADS = 10

_h2t = html2text_module.HTML2Text()
_h2t.ignore_links = True


class _LiteralDumper(yaml.Dumper):
    pass


def _literal_str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralDumper.add_representer(str, _literal_str_representer)


def write_metadata_yml(
    model_path: str, model_id: int, model_details: Dict[str, Any]
) -> Optional[str]:
    """Write a {stem}.yml alongside the model file. Skips if file already exists."""
    yml_path = Path(model_path).with_suffix(".yml")
    if yml_path.exists():
        return str(yml_path)

    description_html = model_details.get("description", "") or ""
    if description_html:
        raw = _h2t.handle(description_html)
        description = "\n".join(line.rstrip() for line in raw.splitlines()).strip()
    else:
        description = ""

    versions = model_details.get("versions", [])
    base_model = versions[0].get("base_model", "") if versions else model_details.get("base_model", "")

    trained_words = model_details.get("trainedWords", "None")
    if isinstance(trained_words, list):
        trigger_words = trained_words
    elif trained_words and trained_words != "None":
        trigger_words = [trained_words]
    else:
        trigger_words = []

    metadata: Dict[str, Any] = {
        "name": model_details.get("name", ""),
        "base_model": base_model,
        "source_url": f"https://civitai.com/models/{model_id}",
        "tags": model_details.get("tags", []),
    }
    if description:
        metadata["description"] = description
    if trigger_words:
        metadata["trigger_words"] = trigger_words

    try:
        with open(yml_path, "w", encoding="utf-8") as f:
            yaml.dump(metadata, f, Dumper=_LiteralDumper, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return str(yml_path)
    except Exception:
        return None


def select_version(
    model_name: str, versions: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    feedback_message(
        f"Please select a version to download for model {model_name}.", "info"
    )

    versions_table = create_table(
        "Available Versions",
        [
            ("Version ID", "bright_yellow"),
            ("Version Name", "cyan"),
            ("Base Model", "blue"),
        ],
    )

    for version in versions:
        versions_table.add_row(
            str(version["id"]), version["name"], version["base_model"]
        )

    console.print(versions_table)
    selected_version_id = typer.prompt("Enter the version ID to download:")

    for version in versions:
        if str(version["id"]) == selected_version_id:
            return version

    feedback_message(
        f"Version {selected_version_id} is not available for model {model_name}.",
        "error",
    )
    return None


def check_for_upgrade(
    versions: dict, model_path: str, selected_version: Dict[str, Any], json_mode: bool = False
) -> bool:
    if json_mode:
        return False
    latest_version = versions[0].get("file")
    if latest_version != selected_version["file"]:
        feedback_message(
            f"A newer version '{latest_version}' is available.", "info"
        )
        return typer.confirm("Do you want to upgrade?", default=True)
    return False


def download_model(
    MODELS_DIR: str,
    CIVITAI_DOWNLOAD: str,
    CIVITAI_TOKEN: str,
    TYPES,
    model_id: int,
    model_details: Dict[str, Any],
    select: bool = False,
    json_mode: bool = False,
    version_id: Optional[int] = None,
) -> Optional[str]:
    model_name = model_details.get("name", f"Model_{model_id}")
    model_type = model_details.get("type", "unknown")
    model_meta = model_details.get("metadata", {})
    versions = model_details.get("versions", [])

    if not versions and not model_details.get("parent_id"):
        if not json_mode:
            feedback_message(f"No versions available for model {model_name}.", "warning")
        return None

    if model_details.get("parent_id"):
        # Already a specific version (looked up via version ID)
        selected_version = {
            "id": model_id,
            "name": model_details.get("name", ""),
            "base_model": model_details.get("base_model", ""),
            "download_url": model_details.get("download_url", ""),
            "images": model_details["images"][0].get("url", "") if model_details.get("images") else "",
            "file": model_meta.get("file", ""),
        }
    elif version_id is not None:
        # model_id@version_id syntax: find the matching version
        matched = next((v for v in versions if v["id"] == version_id), None)
        if not matched:
            if not json_mode:
                feedback_message(
                    f"Version {version_id} not found for model {model_name}. Available versions:",
                    "error",
                )
                versions_table = create_table(
                    "Available Versions",
                    [("Version ID", "bright_yellow"), ("Version Name", "cyan"), ("Base Model", "blue")],
                )
                for v in versions:
                    versions_table.add_row(str(v["id"]), v["name"], v["base_model"])
                console.print(versions_table)
            return None
        selected_version = matched
    elif len(versions) > 1 and not select:
        # Multiple versions, no version pinned — require explicit selection
        if not json_mode:
            feedback_message(
                f"Model {model_name} has {len(versions)} versions. Specify one with {model_id}@<version_id>:",
                "error",
            )
            versions_table = create_table(
                "Available Versions",
                [("Version ID", "bright_yellow"), ("Version Name", "cyan"), ("Base Model", "blue")],
            )
            for v in versions:
                versions_table.add_row(str(v["id"]), v["name"], v["base_model"])
            console.print(versions_table)
        else:
            raise ValueError(
                f"Model {model_name} has multiple versions. Use {model_id}@<version_id>. "
                f"Available: {[{'id': v['id'], 'name': v['name']} for v in versions]}"
            )
        return None
    elif select:
        if model_details.get("parent_id"):
            if not json_mode:
                feedback_message(
                    f"Model {model_name} is a variant of {model_details['parent_name']} // Model ID: {model_details['parent_id']} \r Needs to be a parent model",
                    "warning",
                )
            return None
        selected_version = select_version(model_name, versions)
    else:
        # Single version, no selection needed
        selected_version = versions[0]

    if not selected_version:
        if not json_mode:
            feedback_message(f"A version is not available for model {model_name}.", "error")
        return None

    model_folder = get_model_folder(MODELS_DIR, model_type, TYPES)
    model_path = os.path.join(
        model_folder,
        selected_version.get("base_model", ""),
        selected_version.get("file"),
    )

    if os.path.exists(model_path):
        if not check_for_upgrade(versions, model_path, selected_version, json_mode=json_mode):
            if not json_mode:
                feedback_message(
                    f"Model {model_name} already exists at {model_path}. Skipping download.",
                    "warning",
                )
            return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return download_file(
        f"{CIVITAI_DOWNLOAD}/{selected_version['id']}?token={CIVITAI_TOKEN}",
        model_path,
        model_name,
        json_mode=json_mode,
    )


def download_file(url: str, path: str, desc: str, json_mode: bool = False) -> Optional[str]:
    temp_file = None
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0, read=None)) as client:
            for attempt in range(MAX_RETRIES):
                try:
                    with client.stream(
                        "GET", url, follow_redirects=True, timeout=TIMEOUT
                    ) as response:
                        response.raise_for_status()
                        total_size = int(response.headers.get("content-length", 0))
                        temp_file = tempfile.NamedTemporaryFile(
                            delete=False, dir=os.path.dirname(path)
                        )
                        if json_mode:
                            buffer = bytearray()
                            for chunk in response.iter_bytes(chunk_size=131072):
                                if chunk:
                                    buffer.extend(chunk)
                                if len(buffer) >= 1048576:  # 1MB
                                    temp_file.write(buffer)
                                    buffer.clear()
                            temp_file.write(buffer)
                        else:
                            with tqdm(
                                total=total_size,
                                unit="B",
                                unit_scale=True,
                                desc=f"Downloading {desc}",
                                colour="yellow",
                            ) as progress_bar:
                                buffer = bytearray()
                                for chunk in response.iter_bytes(chunk_size=131072):
                                    if chunk:
                                        buffer.extend(chunk)
                                        progress_bar.update(len(chunk))
                                    if len(buffer) >= 1048576:  # 1MB
                                        temp_file.write(buffer)
                                        buffer.clear()
                                temp_file.write(buffer)
                        temp_file.close()
                        shutil.move(temp_file.name, path)
                        return path
                except (httpx.RequestError, httpx.TimeoutException) as e:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = 2**attempt
                        if not json_mode:
                            feedback_message(
                                f"Download failed. Retrying in {wait_time} seconds...",
                                "warning",
                            )
                        time.sleep(wait_time)
                    else:
                        raise
    except Exception as e:
        if not json_mode:
            feedback_message(f"Failed to download the file: {e}", "error")
        else:
            raise
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    return None


def download_multiple_models(
    identifiers: List[str], select: bool, json_mode: bool = False, no_metadata: bool = False, **kwargs
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    results = []
    for identifier in identifiers[:MAX_CONCURRENT_DOWNLOADS]:
        result = download_single_model(identifier, select, json_mode=json_mode, no_metadata=no_metadata, **kwargs)
        results.append(result)
    return results


def download_single_model(
    identifier: str, select: bool, json_mode: bool = False, no_metadata: bool = False, **kwargs
) -> Tuple[str, Optional[str], Optional[str]]:
    try:
        version_id = None
        if "@" in identifier:
            model_part, version_part = identifier.split("@", 1)
            model_id = int(model_part)
            version_id = int(version_part)
        else:
            model_id = int(identifier)
        model_details = get_model_details(
            kwargs.get("CIVITAI_MODELS"),
            kwargs.get("CIVITAI_VERSIONS"),
            model_id,
        )
        types = kwargs.get("TYPES")
        if model_details:
            try:
                model_path = download_model(
                    kwargs.get("MODELS_DIR"),
                    kwargs.get("CIVITAI_DOWNLOAD"),
                    kwargs.get("CIVITAI_TOKEN"),
                    types,
                    model_id,
                    model_details,
                    select,
                    json_mode=json_mode,
                    version_id=version_id,
                )
            except Exception as e:
                if json_mode:
                    return identifier, None, str(e), None
                feedback_message(f"Failed to download the model {identifier}: {e}", "error")
                return identifier, None, str(e), None
            if model_path:
                metadata_path = None
                if not no_metadata:
                    metadata_path = write_metadata_yml(model_path, model_id, model_details)
                if not json_mode:
                    feedback_message(
                        f"Model {identifier} downloaded successfully at: {model_path}",
                        "info",
                    )
                return identifier, model_path, None, metadata_path
            else:
                if model_path is not None and not json_mode:
                    feedback_message(
                        f"Failed to download the model {identifier}.", "error"
                    )
                return identifier, None, f"Failed to download model {identifier}", None
        else:
            if not json_mode:
                feedback_message(f"No model found with ID: {identifier}.", "error")
            return identifier, None, f"No model found with ID: {identifier}", None
    except ValueError:
        if not json_mode:
            feedback_message(
                f"Invalid model ID: {identifier}. Please enter a valid number.", "error"
            )
        return identifier, None, f"Invalid model ID: {identifier}", None
    except Exception as e:
        if not json_mode:
            feedback_message(f"Error processing model {identifier}: {e}", "error")
        return identifier, None, str(e), None


def download_model_cli(identifiers: List[str], select: bool = False, **kwargs) -> None:
    json_mode = kwargs.pop("json_mode", False)
    no_metadata = kwargs.pop("no_metadata", False)
    if not identifiers:
        if json_mode:
            print(json.dumps({"status": "error", "message": "No model identifiers provided."}))
        else:
            feedback_message("No model identifiers provided.", "error")
        return
    results = download_multiple_models(identifiers, select, json_mode=json_mode, no_metadata=no_metadata, **kwargs)
    if json_mode:
        output = []
        for identifier, path, error, metadata_path in results:
            if path:
                output.append({
                    "status": "ok",
                    "model_id": identifier,
                    "path": path,
                    "size_bytes": os.path.getsize(path),
                    "model_name": os.path.basename(path),
                    "metadata_path": metadata_path,
                })
            else:
                output.append({
                    "status": "error",
                    "model_id": identifier,
                    "message": error or f"Failed to download model {identifier}",
                })
        print(json.dumps(output if len(output) > 1 else output[0] if output else {}))
