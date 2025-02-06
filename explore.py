import jedi
import sys
from pathlib import Path

def get_symbol_info(symbol: str) -> None:
    """Find and display symbol definition"""
    try:
        # Create a Jedi Script with the current environment
        script = jedi.Script(path=str(Path.cwd()))
        
        # Find references to the symbol
        refs = script.get_references(line=1, column=1, path=None)
        
        if not refs:
            print(f"No references found for {symbol}")
            return
            
        # Get definition
        definition = refs[0].get_definition()
        if definition:
            print(f"\nDefinition found in {definition.module_path}:line {definition.line}")
            print("\nSource:")
            print(definition.get_line_code())
            
            # Show a few lines of context
            with open(definition.module_path) as f:
                lines = f.readlines()
                start = max(0, definition.line - 3)
                end = min(len(lines), definition.line + 3)
                print("\nContext:")
                for i in range(start, end):
                    print(f"{i+1}: {lines[i].rstrip()}")
        
    except Exception as e:
        print(f"Error finding symbol {symbol}: {str(e)}")

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
