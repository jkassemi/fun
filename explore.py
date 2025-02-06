import inspect
import sys
from pathlib import Path
import importlib
from typing import Optional, Tuple

def get_symbol_info(symbol: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Get source info for a symbol, returns (source, file_path, line_no)"""
    try:
        # Try to import the module containing the symbol
        parts = symbol.split('.')
        module_name = parts[0]
        
        # Special case for textual - try textual.widgets for widget classes
        if module_name == 'textual' and len(parts) > 1:
            try:
                module = importlib.import_module(f'textual.widgets')
                obj = getattr(module, parts[1])
            except (ImportError, AttributeError):
                module = importlib.import_module(module_name)
                obj = module
                
        else:
            module = importlib.import_module(module_name)
            obj = module
        
        # Get the object
        for part in parts[1:]:
            obj = getattr(obj, part)
            
        # Get source info
        source = inspect.getsource(obj)
        file_path = inspect.getfile(obj)
        line_no = inspect.getsourcelines(obj)[1]
        
        return source, file_path, line_no
        
    except Exception as e:
        print(f"Error finding symbol {symbol}: {str(e)}")
        return None, None, None

def jump(symbol: str, context_lines: int = 4):
    """Jump to symbol definition and show surrounding context"""
    source, file_path, line_no = get_symbol_info(symbol)
    
    if source:
        print(f"\nDefinition found in {file_path}:line {line_no}")
        print(f"\nSource with {context_lines} lines of context:")
        print(source)
    else:
        print(f"Could not find source for {symbol}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore.py <symbol>")
        print("\nExample symbols to try:")
        print("  textual.app.App")
        print("  textual.widgets.TextArea")
        sys.exit(1)
        
    jump(sys.argv[1])
