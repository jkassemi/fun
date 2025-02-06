import inspect
import sys
from pathlib import Path
import importlib

def jump(symbol: str, context_lines: int = 4):
    """Jump to symbol definition and show surrounding context"""
    try:
        # Try to import the module containing the symbol
        module_name = symbol.split('.')[0]
        module = importlib.import_module(module_name)
        
        # Get the object
        obj = module
        for part in symbol.split('.')[1:]:
            obj = getattr(obj, part)
            
        # Get source
        source = inspect.getsource(obj)
        file_path = inspect.getfile(obj)
        line_no = inspect.getsourcelines(obj)[1]
        
        print(f"\nDefinition found in {file_path}:line {line_no}")
        print(f"\nSource with {context_lines} lines of context:")
        print(source)
        
    except Exception as e:
        print(f"Error finding symbol {symbol}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore.py <symbol>")
        sys.exit(1)
        
    jump(sys.argv[1])
