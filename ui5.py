"""
A UI for exploring geometric transformation concepts.
Visualize and manipulate embedding spaces and attention patterns.
"""

import os
import mlx.core as mx
from mlx_lm import load, generate
import asyncio
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Header,
    Footer,
    Static,
    DataTable,
    TextArea,
    ProgressBar,
    Log,
    Button,
)
from textual import events
import numpy as np
from typing import List, Dict, Tuple


class TokenTable(DataTable):
    """A table showing token information and predictions"""
    
    def __init__(self, id: str):
        super().__init__(id=id)
        self.add_columns("Position", "Token", "ID", "Predictions")
        
    def update_tokens(self, tokens: List[Tuple[str, int]], predictions: List[List[Tuple[str, float]]] = None):
        """Update table with tokens and their predictions"""
        self.clear()
        if not predictions:
            predictions = [[]] * len(tokens)
            
        for pos, ((token, token_id), preds) in enumerate(zip(tokens, predictions)):
            pred_str = " | ".join(f"{t}:{p:.2f}" for t,p in preds[:5]) if preds else ""
            self.add_row(str(pos), token, str(token_id), pred_str)


class TokenExplorer(App):
    """Interactive token exploration interface"""

    BINDINGS = [
        ("ctrl+r", "reload_app", "Reload"),
    ]

    CSS = """
    Button {
        margin: 1;
    }
    .table-header {
        background: $accent;
        color: $text;
        text-align: center;
    }
    
    TextArea {
        height: 2fr;
        border: solid green;
        margin: 1;
    }

    #horizontal {
        height: 4;
    }

    #inactivity-progress {
        width: 100%;
        height: 1;
        margin: 1;
        align: right middle;
    }
    
    TopTokenAnalysisView, BottomTokenAnalysisView {
        height: 1fr;
        border: solid blue;
    }
    
    #top-token-table, #bottom-token-table {
        height: 1fr;
        margin: 1;
    }
    
    Log {
        height: 10;
        border: solid red;
        margin: 1;
    }
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self._inactivity_timer = None
        self._progress_timer = None

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Horizontal(id="horizontal"):
            yield Button("Load Model", id="load-model")
            yield Button("Generate", id="generate", disabled=True)
            yield ProgressBar(
                total=100, show_percentage=False, id="inactivity-progress"
            )
        yield TextArea()
        yield Static("Top Predictions", classes="table-header")
        yield TokenTable(id="top-tokens")
        yield Static("Bottom Predictions", classes="table-header")
        yield TokenTable(id="bottom-tokens")
        yield Log()
        yield Footer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        log = self.query_one(Log)
        if event.button.id == "load-model":
            try:
                log.write_line("Starting model load...")
                log.write_line(f"Button handler environment:")
                log.write_line(f"HF_TOKEN: {os.environ.get('HF_TOKEN')}")
                log.write_line(f"PWD: {os.environ.get('PWD')}")

                # Run model loading in executor to keep UI responsive
                checkpoint = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
                (
                    self.model,
                    self.tokenizer,
                ) = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: load(path_or_hf_repo=checkpoint)
                )
                log.write_line("Model loaded successfully!")
                log.write_line(f"Model info: {self.model.__class__.__name__}")
                log.write_line(f"Tokenizer info: {self.tokenizer.__class__.__name__}")
                self.query_one("#generate").disabled = False
            except Exception as e:
                log.write_line(f"Error loading model: {type(e).__name__}: {str(e)}")
                log.write_line(f"Error details: {str(e)}")

        elif event.button.id == "generate":
            if self.model and self.tokenizer:
                text_area = self.query_one(TextArea)
                log.write_line("Generating...")
                response = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=self.tokenizer.encode(text_area.text),
                    max_tokens=100,
                    verbose=True,
                )
                text_area.text = response
                log.write_line("Generation complete!")

    def on_mount(self) -> None:
        """Set up initial state when app starts."""
        # Set up analysis views
        top_analysis = self.query_one(TokenAnalysisView)
        bottom_analysis = self.query_one(TokenAnalysisView)
        text_area = self.query_one(TextArea)

        # Set initial text
        text_area.text = "The quick"

        # Sample tokens - just the ones from our text
        tokens = [("The", 464), ("quick", 4789)]

        # Mock top predictions for just our two tokens
        top_predictions = [
            [("is", 0.95), ("was", 0.85), ("and", 0.80), ("has", 0.78), ("will", 0.75)],
            [
                ("brown", 0.90),
                ("red", 0.85),
                ("lazy", 0.80),
                ("small", 0.78),
                ("big", 0.75),
            ],
        ]

        # Mock bottom predictions for just our two tokens
        bottom_predictions = [
            [("xyz", 0.05), ("123", 0.04), ("@#$", 0.03), ("...", 0.02), ("???", 0.01)],
            [
                ("9876", 0.06),
                ("qwer", 0.05),
                ("asdf", 0.04),
                ("zxcv", 0.03),
                ("jklm", 0.02),
            ],
        ]

        top_analysis.update_current_tokens(tokens, predictions=top_predictions)
        bottom_analysis.update_current_tokens(tokens, predictions=bottom_predictions)

    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text changes"""
        # Cancel existing timers if any
        if self._inactivity_timer:
            self._inactivity_timer.cancel()
        if self._progress_timer:
            self._progress_timer.cancel()

        # Reset progress bar
        progress = self.query_one(ProgressBar)
        progress.update(progress=0)

        # Create new timers
        self._inactivity_timer = asyncio.create_task(self._handle_inactivity())
        self._progress_timer = asyncio.create_task(self._update_progress())

        # Get the TextArea widget and its current content
        text_area = self.query_one(TextArea)
        current_text = text_area.text

    async def _update_progress(self) -> None:
        """Update progress bar during inactivity period"""
        progress = self.query_one(ProgressBar)
        for i in range(30):  # 30 steps over 3 seconds
            await asyncio.sleep(0.1)  # 100ms per step
            progress.update(progress=((i + 1) * 100) // 30)

    async def _handle_inactivity(self) -> None:
        """Called when text area has been inactive for 3 seconds"""
        await asyncio.sleep(3.0)

        try:
            progress = self.query_one(ProgressBar)
            progress.update(progress=100)

            # Get current text
            text_area = self.query_one(TextArea)
            current_text = text_area.text

            if self.model is None or self.tokenizer is None:
                log = self.query_one(Log)
                log.write_line("Please load model first")
                return

            # Flow: text -> ids -> model -> logits -> probs
            # Each step preserves meaning while changing shape
            input_ids = self.tokenizer.encode(current_text)
            input_tokens = [(self.tokenizer.decode([id]), id) for id in input_ids]
            
            model_logits = self.model(mx.array([input_ids]))
            next_token_logits = model_logits[-1, -1]  # Last position only
            next_token_probs = mx.softmax(next_token_logits, axis=-1)

            # Get top 5 predictions for next token
            predictions = []
            # Get indices of top 5 probabilities
            top_indices = mx.argmax(next_token_probs, axis=-1)[:5]
            # Convert to (token, prob) pairs
            next_token_preds = [(self.tokenizer.decode([idx]), float(next_token_probs[idx])) 
                               for idx in top_indices]
            predictions = [next_token_preds]  # Single position predictions

            # Update token tables with predictions
            top_table = self.query_one("#top-tokens", TokenTable)
            bottom_table = self.query_one("#bottom-tokens", TokenTable)
            
            top_table.update_tokens(input_tokens, predictions=predictions)
            bottom_table.update_tokens(input_tokens, predictions=[])  # Empty for bottom table
        except Exception as e:
            try:
                import traceback
                log = self.query_one(Log)
                log.write_line(f"Error in _handle_inactivity: {type(e).__name__}: {str(e)}")
                log.write_line("Traceback:")
                for line in traceback.format_exc().split('\n'):
                    log.write_line(line)
            except Exception as inner_e:
                print(f"Failed to log error: {inner_e}")
                print(traceback.format_exc())


if __name__ == "__main__":
    import sys
    import os

    print("\nMain process environment:")
    print("HF_TOKEN:", os.environ.get("HF_TOKEN"))
    print("PWD:", os.environ.get("PWD"))

    print("\nTrying model load in main process...")
    try:
        model, tokenizer = load("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        print("Main process model load succeeded!")
        print("Model info:", model.__class__.__name__)
        print("Tokenizer info:", tokenizer.__class__.__name__)
    except Exception as e:
        print("Main process model load failed:", str(e))

    app = TokenExplorer()

    # Allow --test flag for automated testing
    if "--test" in sys.argv:
        print("Running in test mode")

        # Create test app and debug widget tree
        test_app = TokenExplorer()
        print("\nApp created")

        # Try composing
        widgets = test_app.compose()
        print("\nCompose returned:", list(widgets))

        # Debug widget tree
        print("\nWidget tree before mount:")
        print(test_app.tree)

        # Try mounting
        test_app.post_message(events.Mount())
        print("\nWidget tree after mount:")
        print(test_app.tree)

        # Debug query
        try:
            text_area = test_app.query_one(TextArea)
            print("\nFound TextArea:", text_area)
        except Exception as e:
            print(f"\nError finding TextArea: {type(e).__name__}: {str(e)}")
    else:
        app.run()


# ideas when training:
#
# 1. you could probably find xml tags to define how to perceive the associated content useful.
#           def <1>my_function</1>:
#               pass
#
#    think of this like an extension of the `ai?`, 'ai', 'ai!' marking concept. leads to more
#    effective context discovery.
#
# 2.
