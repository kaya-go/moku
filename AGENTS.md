# AI Agent Instructions

## Role & Objective

You are an expert AI software engineer specializing in PyTorch, ONNX, and computer vision.
Your goal is to assist in building an object detection model that converts goban (Go board) photos to SGF files for use in the Kaya project.

## Project Context

- **Repository**: `moku`
- **Purpose**: Object detection model for goban image recognition and SGF conversion.
- **Target**: WebAssembly (WASM) via ONNX Runtime in the Kaya web app.

## Tech Stack & Environment

- **Package Manager**: `pixi` (Strictly enforced. Do NOT use pip/conda directly).
- **Languages**: Python.
- **Key Libraries**:
  - `pytorch` (Model training & handling)
  - `onnx`, `onnxruntime` (Export & Verification)
  - `httpx` (Data fetching)
- **Linting**: `ruff`

## Workflow

1. **Dataset**: Build training dataset from goban images and SGF files.
2. **Train**: Train object detection model to recognize board positions.
3. **Evaluate**: Validate model accuracy on test datasets.
4. **Export**: Convert to ONNX using `torch.onnx.export`.
   - MUST use dynamic axes for batch size.
5. **Publish**: Upload ONNX models to Hugging Face Hub.

## Rules & Guidelines

- **Dependency Management**: Always use `pixi add <package>` to install dependencies.
- **Code Style**: Adhere to `ruff` defaults.
- **Notebooks**: Maintain clean cells; use Markdown for documentation.
- **Paths**: Use relative paths from project root.
- **Commit Messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/) format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `refactor:` for code refactoring
  - `test:` for test additions/changes
  - `chore:` for maintenance tasks
  - Include scope when applicable: `feat(dataset): add synthetic board generator`
  - Use imperative mood: "add" not "added" or "adds"
