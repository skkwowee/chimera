def __getattr__(name):
    if name == "Qwen3VLInference":
        from .vlm import Qwen3VLInference
        return Qwen3VLInference
    if name == "parse_json_response":
        from .vlm import parse_json_response
        return parse_json_response
    if name == "ClaudeVLMInference":
        from .claude import ClaudeVLMInference
        return ClaudeVLMInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Qwen3VLInference", "ClaudeVLMInference", "parse_json_response"]
