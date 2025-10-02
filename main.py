#!/usr/bin/env python3
"""
Main entry point for IntelliStock Pro: AI-Powered Stock Prediction & Analytics Platform.

This script provides a command-line interface to train models, run predictions,
perform analysis, and run tests.

Developer: Himanshu Salunke
GitHub: https://github.com/HimanshuSalunke
LinkedIn: https://www.linkedin.com/in/himanshuksalunke/
"""

import argparse
import sys
from pathlib import Path

# Add the 'src' directory to the Python path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from scripts.training import run_training_pipeline as train_main
from scripts.prediction import interactive_demo as predict_demo
from scripts.analysis import interactive_analysis_tool as analysis_main
from scripts.testing import main as test_main

def main():
    """Parses command-line arguments and executes the corresponding command."""
    parser = argparse.ArgumentParser(
        description="IntelliStock Pro: AI-Powered Stock Prediction & Analytics Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --stocks RELIANCE SBIN --models lstm xgboost
  python main.py predict
  python main.py analyze --stock RELIANCE
  python main.py test
"""
    )

    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Train Command ---
    parser_train = subparsers.add_parser('train', help='Train machine learning models.')
    parser_train.add_argument(
        '--stocks',
        nargs='+',
        metavar='SYMBOL',
        help='One or more stock symbols to train on. (default: all available)'
    )
    parser_train.add_argument(
        '--models',
        nargs='+',
        default=['lstm', 'random_forest', 'xgboost'],
        help='Specific models to train. (default: lstm, random_forest, xgboost)'
    )
    # Set the function to be called for the 'train' command
    parser_train.set_defaults(func=train_main)

    # --- Predict Command ---
    parser_predict = subparsers.add_parser('predict', help='Run an interactive prediction demo.')
    # Set the function to be called for the 'predict' command
    parser_predict.set_defaults(func=predict_demo)

    # --- Analyze Command ---
    parser_analyze = subparsers.add_parser('analyze', help='Perform model analysis.')
    # Example of a command-specific argument
    parser_analyze.add_argument(
        '--stock',
        metavar='SYMBOL',
        help='A specific stock symbol to analyze.'
    )
    # Set the function to be called for the 'analyze' command
    parser_analyze.set_defaults(func=analysis_main)

    # --- Test Command ---
    parser_test = subparsers.add_parser('test', help='Run tests on the models.')
    # Set the function to be called for the 'test' command
    parser_test.set_defaults(func=test_main)

    args = parser.parse_args()

    # --- Execute the corresponding function ---
    print(f"Executing command: {args.command}...")

    # The `func` attribute was set by `set_defaults` for the chosen subparser.
    # We pass the entire 'args' object so the function can access all its arguments.
    # Note: You will need to update the called functions (e.g., train_main) to accept an 'args' parameter.
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)