; Syntax highlighting for Kedi (tree-sitter-kedi)
;
; Editor consumers: VS Code, Neovim (nvim-treesitter), Helix, Zed,
; tree-sitter highlight CLI. Captures use the standard tree-sitter
; capture names so existing themes pick them up without extra mapping.

; ----------------------------------------------------------------
; Comments
; ----------------------------------------------------------------

(line_comment) @comment.line
(block_comment) @comment.block
(procedure_def
  body: (block
    . (block_comment) @variable))

; ----------------------------------------------------------------
; Operators / punctuation
; ----------------------------------------------------------------

(template_block_stmt
  ">>" @keyword)

["@" "~" ">"] @punctuation.special
["=" ":" "(" ")" "[" "]" "<" "," "|" "->" "*"] @punctuation.delimiter
["```"] @punctuation.special
"`" @punctuation.special

; ----------------------------------------------------------------
; Procedure & validation block keywords
; ----------------------------------------------------------------

(validation_keyword) @keyword
"import" @operator
"export" @operator
"auto" @keyword
"optimize" @keyword
"model" @keyword
"effort" @keyword
"system" @keyword
"mcp" @keyword
"settings" @keyword
"profile" @keyword
"use" @keyword
"case" @keyword
"data" @keyword
"test_data" @keyword
"metric" @keyword

; ----------------------------------------------------------------
; Procedure and type-definition names
; ----------------------------------------------------------------

(procedure_def name: (identifier) @function)
(type_def name: (identifier) @type.definition)
(module_import module: (identifier) @namespace)
(module_export_name name: (identifier) @variable)
(assign_target name: (identifier) @variable)
(param name: (identifier) @variable.parameter)
(type_field name: (identifier) @property)

; ----------------------------------------------------------------
; Template segments
; ----------------------------------------------------------------

; Segment captures apply anywhere (template_line, template_block, returns, …).
(input_segment name: (identifier) @variable)
(system_angle_segment) @variable
(call_segment name: (identifier) @function.call)
(output_segment
  name: (identifier) @variable.builtin)
(text_segment) @string

; ----------------------------------------------------------------
; Type expressions
; ----------------------------------------------------------------

(type_ref name: (identifier) @type)
(type_apply name: (identifier) @type)

; ----------------------------------------------------------------
; Validation suite names (after `@test:` / `@eval:`)
; ----------------------------------------------------------------

(validation_block procedure: (identifier) @function)
(test_case name: (identifier) @label)
(eval_data name: (identifier) @label)
(eval_test_data name: (identifier) @label)
(eval_metric name: (identifier) @label)
(eval_metric dataset: (identifier) @variable)

(optimize_directive name: (identifier) @label)
(profile_directive name: (identifier) @label)
(use_directive name: (identifier) @label)
(use_tool_name name: (identifier) @function.call)
(use_tool_backtick name: (identifier) @function.call)
(mcp_field name: (identifier) @property)
(settings_field name: (identifier) @property)

(model_directive
  value: (model_plain_value) @string)
(model_directive
  value: (inline_python_expr) @string.special)

(effort_directive
  value: (effort_plain_value) @string)
(effort_directive
  value: (inline_python_expr) @string.special)

(mcp_field
  value: (mcp_plain_value) @string)
(mcp_field
  value: (inline_python_expr) @string.special)

(settings_field
  value: (settings_plain_value) @string)
(settings_field
  value: (inline_python_expr) @string.special)

; ----------------------------------------------------------------
; Python source embedded in Kedi (these regions get a Python
; injection — see queries/injections.scm — but we also give them a
; fallback "code" style so they remain readable when no injection
; parser is available).
; ----------------------------------------------------------------

(python_code) @string.special
(python_inline_body) @string.special
