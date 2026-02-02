# from https://github.com/fastapi/fastapi-cli/blob/main/src/fastapi_cli/discover.py

import importlib
import sys
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Union

from .._exceptions import CLIError
from ..worker import AgentServer

logger = getLogger(__name__)


def get_default_path() -> Path:
    potential_paths = ("main.py", "app.py", "agent.py", "app/main.py", "app/app.py", "app/agent.py")

    for full_path in potential_paths:
        path = Path(full_path)
        if path.is_file():
            return path

    raise CLIError("Could not find a default file to run, please provide an explicit path")


@dataclass
class ModuleData:
    module_import_str: str
    extra_sys_path: Path
    module_paths: list[Path]


def get_module_data_from_path(path: Path) -> ModuleData:
    use_path = path.resolve()
    module_path = use_path
    if use_path.is_file() and use_path.stem == "__init__":
        module_path = use_path.parent
    module_paths = [module_path]
    extra_sys_path = module_path.parent
    for parent in module_path.parents:
        init_path = parent / "__init__.py"
        if init_path.is_file():
            module_paths.insert(0, parent)
            extra_sys_path = parent.parent
        else:
            break

    module_str = ".".join(p.stem for p in module_paths)
    return ModuleData(
        module_import_str=module_str,
        extra_sys_path=extra_sys_path.resolve(),
        module_paths=module_paths,
    )


def get_app_name(*, mod_data: ModuleData) -> str:
    try:
        mod = importlib.import_module(mod_data.module_import_str)
    except (ImportError, ValueError) as e:
        logger.error(f"Import error: {e}")
        logger.warning("Ensure all the package directories have an [blue]__init__.py[/blue] file")
        raise

    object_names = dir(mod)
    object_names_set = set(object_names)

    for preferred_name in ["app", "server", "agent"]:
        if preferred_name in object_names_set:
            obj = getattr(mod, preferred_name)
            if isinstance(obj, AgentServer):
                return preferred_name
    for name in object_names:
        obj = getattr(mod, name)
        if isinstance(obj, AgentServer):
            return name
    raise CLIError("Could not find AgentServer in module, try to define the `server` variable")


@dataclass
class ImportData:
    app_name: str
    module_data: ModuleData
    import_string: str


def get_import_data(*, path: Union[Path, None] = None) -> ImportData:
    if not path:
        path = get_default_path()

    if not path.exists():
        raise CLIError(f"Path does not exist {path}")

    mod_data = get_module_data_from_path(path)
    sys.path.insert(0, str(mod_data.extra_sys_path))
    use_app_name = get_app_name(mod_data=mod_data)

    import_string = f"{mod_data.module_import_str}:{use_app_name}"

    return ImportData(app_name=use_app_name, module_data=mod_data, import_string=import_string)
