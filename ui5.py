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


class TopTokenAnalysisView(Static):
    """Shows top token predictions"""

    def compose(self) -> ComposeResult:
        """Create tables for token analysis"""
        yield Static("Top Predictions", classes="table-header")
        yield DataTable(id="top-token-table")

    def on_mount(self) -> None:
        """Initialize the data tables"""
        token_table = self.query_one("#top-token-table", DataTable)
        token_table.add_columns(
            "Position",
            "prev(n=1)|p",
            "Token",
            "ID",
            "Is Locked",
            "top(n=1)|v",
            "top(n=1)|p",
            "top(n=2)|v",
            "top(n=2)|p",
            "top(n=3)|v",
            "top(n=3)|p",
            "top(n=4)|v",
            "top(n=4)|p",
            "top(n=5)|v",
            "top(n=5)|p",
        )

    def update_current_tokens(
        self,
        tokens: List[Tuple[str, int]],
        locked_positions: set = None,
        predictions: List[List[Tuple[str, float]]] = None,
    ):
        """Update the current tokens display"""
        if locked_positions is None:
            locked_positions = set()
        if predictions is None:
            predictions = [[("", 0.0)] * 5] * len(tokens)  # Default empty predictions

        try:
            log = self.query_one(Log)
            log.write_line("Top table starting token update")
        except Exception as e:
            print(f"Error in TopTokenAnalysisView.update_current_tokens: {type(e).__name__}: {str(e)}")

        table = self.query_one("#top-token-table", DataTable)
        table.clear()

        for pos, ((token, token_id), token_predictions) in enumerate(
            zip(tokens, predictions)
        ):
            is_locked = "🔒" if pos in locked_positions else ""

            # Pad predictions to ensure 5 entries
            while len(token_predictions) < 5:
                token_predictions.append(("", 0.0))

            # Flatten predictions into alternating token/probability values
            pred_values = []
            for pred_token, pred_prob in token_predictions[:5]:
                pred_values.extend([pred_token, f"{pred_prob:.4f}"])

            # Calculate prev token probability (mock for now)
            prev_prob = "0.00" if pos == 0 else "0.85"
            table.add_row(
                str(pos), prev_prob, token, str(token_id), is_locked, *pred_values
            )


class BottomTokenAnalysisView(Static):
    """Shows bottom token predictions"""

    def compose(self) -> ComposeResult:
        """Create tables for token analysis"""
        yield Static("Bottom Predictions", classes="table-header")
        yield DataTable(id="bottom-token-table")

    def on_mount(self) -> None:
        """Initialize the data tables"""
        token_table = self.query_one("#bottom-token-table", DataTable)
        token_table.add_columns(
            "Position",
            "prev(n=1)|p",
            "Token",
            "ID",
            "Is Locked",
            "bott(n=1)|v",
            "bott(n=1)|p",
            "bott(n=2)|v",
            "bott(n=2)|p",
            "bott(n=3)|v",
            "bott(n=3)|p",
            "bott(n=4)|v",
            "bott(n=4)|p",
            "bott(n=5)|v",
            "bott(n=5)|p",
        )

    def update_current_tokens(
        self,
        tokens: List[Tuple[str, int]],
        locked_positions: set = None,
        predictions: List[List[Tuple[str, float]]] = None,
    ):
        """Update the current tokens display"""
        if locked_positions is None:
            locked_positions = set()
        if predictions is None:
            predictions = [[("", 0.0)] * 5] * len(tokens)  # Default empty predictions

        try:
            log = self.query_one(Log)
            log.write_line("Bottom table starting token update")
        except Exception as e:
            print(f"Error in BottomTokenAnalysisView.update_current_tokens: {type(e).__name__}: {str(e)}")

        table = self.query_one("#bottom-token-table", DataTable)
        table.clear()

        for pos, ((token, token_id), token_predictions) in enumerate(
            zip(tokens, predictions)
        ):
            is_locked = "🔒" if pos in locked_positions else ""

            # Pad predictions to ensure 5 entries
            while len(token_predictions) < 5:
                token_predictions.append(("", 0.0))

            # Flatten predictions into alternating token/probability values
            pred_values = []
            for pred_token, pred_prob in token_predictions[:5]:
                pred_values.extend([pred_token, f"{pred_prob:.4f}"])

            # Calculate prev token probability (mock for now)
            prev_prob = "0.00" if pos == 0 else "0.85"
            table.add_row(
                str(pos), prev_prob, token, str(token_id), is_locked, *pred_values
            )


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
        yield TopTokenAnalysisView()
        yield BottomTokenAnalysisView()
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
        top_analysis = self.query_one(TopTokenAnalysisView)
        bottom_analysis = self.query_one(BottomTokenAnalysisView)
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

            # Tokenize with MLX tokenizer
            token_ids = self.tokenizer.encode(current_text)
            tokens = [(self.tokenizer.decode([tid]), tid) for tid in token_ids]

            log = self.query_one(Log)
            log.write_line(f"token_ids shape: {mx.array([token_ids]).shape}")

            # Get model predictions for next token
            model_output = self.model(mx.array([token_ids]))
            log.write_line(f"model_output shape: {model_output.shape}")

            # Get logits for last position only
            last_position_logits = model_output[-1, -1]  # Last layer, last position
            log.write_line(f"last position logits shape: {last_position_logits.shape}")

            # Get probabilities for next token
            next_token_probs = mx.softmax(last_position_logits, axis=-1)
            log.write_line(f"next token probs shape: {next_token_probs.shape}")

            # Get top 5 predictions for next token
            predictions = []
            # Get indices of top 5 probabilities
            top_indices = mx.argmax(next_token_probs, axis=-1)[:5]
            # Convert to (token, prob) pairs
            next_token_preds = [(self.tokenizer.decode([idx]), float(next_token_probs[idx])) 
                               for idx in top_indices]
            predictions = [next_token_preds]  # Single position predictions
                # Get indices of top 5 probabilities
                top_indices = mx.argmax(pos_probs, axis=-1)[:5]
                # Convert to (token, prob) pairs
                pos_preds = [
                    (self.tokenizer.decode([idx]), float(pos_probs[idx]))
                    for idx in top_indices
                ]
                predictions.append(pos_preds)

            top_analysis = self.query_one(TopTokenAnalysisView)
            bottom_analysis = self.query_one(BottomTokenAnalysisView)

            # Update views with real predictions
            top_analysis.update_current_tokens(tokens, predictions=predictions)
            bottom_analysis.update_current_tokens(tokens, predictions=predictions)
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
