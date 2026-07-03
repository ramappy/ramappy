from collections.abc import Mapping

from pydantic import BaseModel, ValidationError


class PipelineError(Exception):
    def __init__(self, error: ValidationError, input_data=None, model: type[BaseModel] | None = None):
        self.raw_errors = error.errors()
        self.input_data = input_data or {}
        self.model = model
        self.message = self._format_errors()
        super().__init__(self.message)

    def _format_errors(self) -> str:
        formatted_errors = []
        for err in self.raw_errors:
            loc = err.get("loc", [])
            loc_str = " -> ".join(str(part) for part in loc)
            msg = err.get("msg", "Invalid value")
            error_type = err.get("type", "")
            expected = self._extract_expected(error_type, err)
            found = self._extract_found(loc)

            formatted = f"• `{loc_str}`: {msg}{expected}{found}"
            formatted_errors.append(formatted)
        schema_str = self._get_schema_str()
        return "Validation error(s):\n" + "\n".join(formatted_errors) + schema_str

    def _extract_expected(self, error_type: str, err: Mapping) -> str:
        ctx = err.get("ctx", {})
        if error_type.startswith("type_error") and "expected" in ctx:
            return f" (expected type: {ctx['expected']})"
        if error_type == "value_error.const" and "expected" in ctx:
            return f" (expected value: {ctx['expected']})"
        if "literal_error" in error_type and "expected" in ctx:
            return f" (expected one of: {ctx['expected']})"
        return ""

    def _extract_found(self, loc) -> str:
        try:
            value = self.input_data
            for key in loc:
                value = value[key] if isinstance(value, dict) else value[int(key)]
            return f" (found: {value!r})"
        except Exception:
            return ""

    def _get_schema_str(self) -> str:
        if self.model and issubclass(self.model, BaseModel):
            schema = self.model.schema()
            # Show only properties and their types/allowed values
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            lines = ["\n\nAvailable/expected fields:"]
            for field, details in props.items():
                line = f"- `{field}`"
                # desc = details.get("title", "")
                # if desc:
                #     desc = f"{desc}: "
                desc = details.get("description", "")
                typ = details.get("type", None)
                enum = details.get("enum")
                desc += f"type: {typ}" if typ else ""
                if enum:
                    desc += f", allowed: {enum}"
                if field in required:
                    desc += " (required)"
                if desc:
                    line += f": {desc}"
                lines.append(line)
            return "\n".join(lines)
        return ""
