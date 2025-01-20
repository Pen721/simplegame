import importlib
import inspect
import sys
from typing import Type, List, TypeVar
import logging
from games.games import ResourceGame
from policies.policies import Policy, SingleStatePolicy, AllPastStatePolicy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')

class ModuleLoader:
    """A class to handle dynamic loading and reloading of modules and their subclasses."""
    def __init__(self, base_package: str):
        self.base_package = base_package
        self.cached_modules = set()

    def reload_modules(self, module_paths: List[str]) -> None:
        for module_path in module_paths:
            try:
                if module_path in sys.modules:
                    logger.info(f"Reloading module: {module_path}")
                    importlib.reload(sys.modules[module_path])
                    self.cached_modules.add(module_path)
            except Exception as e:
                logger.error(f"Error reloading module {module_path}: {str(e)}")

    def load_subclasses(self, 
                       module_path: str, 
                       parent_class: Type[T],
                       exclude_classes: List[Type[T]] = None) -> List[Type[T]]:
        exclude_classes = exclude_classes or []
        try:
            module = importlib.import_module(module_path)
            subclasses = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, parent_class) and 
                    obj not in [parent_class] + exclude_classes):
                    logger.info(f"Found valid subclass: {name}")
                    subclasses.append(obj)
            return subclasses
        except ImportError as e:
            logger.error(f"Error importing module {module_path}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error while loading subclasses: {str(e)}")
            return []

def initialize_game_system():
    """Initialize the game system by loading all games and policies."""
    game_loader = ModuleLoader('games')
    policy_loader = ModuleLoader('policies')
    
    modules_to_reload = [
        'games.ResourceGame',
        'policies.Policy',
        'games.games',
        'policies.policies',
        'utils.utils'
    ]
    
    game_loader.reload_modules(modules_to_reload)
    games = game_loader.load_subclasses(
        'games.games',
        ResourceGame
    )
    
    policies = policy_loader.load_subclasses(
        'policies.policies',
        Policy,
        exclude_classes=[]
    )
    
    return games, policies