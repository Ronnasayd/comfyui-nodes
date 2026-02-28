---
description: Comprehensive guide to generate code with Copilot (CODE MODE ONLY)
applyTo: "**/*"
---

<context>
This instruction file applies exclusively to code generation or code modification tasks.
Code review, auditing, analysis, or diff inspection are handled by a separate review prompt.
Do not perform review-only activities unless explicitly requested.
</context>

<rules>
Generate production-ready code aligned with the project's existing architecture, constraints, and conventions.
Ensure the code follows recognized industry best practices and is optimized for performance, security, and scalability.
Ensure the code is clean, well-structured, readable, and easy to maintain.
Strictly follow coding conventions, naming standards, and the style guidelines of the project or chosen language.
Avoid obsolete or deprecated methods, APIs, or libraries; always prefer current, well-supported alternatives.
Include explanatory comments for complex logic, architectural decisions, known limitations, and non-trivial code sections.
Integrate new implementations compatibly with the existing codebase, following established patterns and boundaries.
Avoid changes that break backward compatibility unless explicitly required; in such cases, provide clear documentation and migration instructions.
Use clear, descriptive, and consistent names for functions, methods, classes, and modules, avoiding ambiguity.
Implement robust error handling and input validation at all relevant points.
Ensure the code is testable, including automated tests, examples, or validation instructions whenever applicable.
For front-end code, ensure responsiveness, accessibility, and cross-browser/device compatibility (WCAG, ARIA).
For back-end code, prioritize security (injection protection, authN/authZ, encryption), scalability, and efficient data access.
If the request is ambiguous, incomplete, or contradictory, ask for clarification before generating code.
Avoid slang, informal language, or excessive abbreviations in comments, documentation, and log messages.
Adopt defensive programming, modularization, reuse, and separation of concerns.
Include clear and objective documentation when introducing significant changes, new components, endpoints, or public interfaces.
Consider internationalization/localization requirements when applicable.
Prefer native language features or well-established libraries over custom implementations unless technically justified.
</rules>

<avoid>
Avoid confusing, generic, or abbreviated names for identifiers (e.g., `x`, `temp1`, `doStuff`).
Do not write complex or non-trivial code without appropriate comments.
Avoid code duplication; prefer reuse through functions, modules, or abstractions.
Do not ignore error handling or assume success in critical operations (I/O, external services).
Do not use magic values without explanation or named constants/enums.
Avoid mixing multiple responsibilities in a single function, class, or module.
Do not implement code without basic tests, validations, or usage examples when feasible.
Do not ignore performance in critical paths (inefficient loops, heavy queries, blocking calls).
Never trust user input without proper validation or protection against vulnerabilities (XSS, CSRF).
Do not leave dead code, commented-out code, unused functions, or obsolete snippets.
Do not reimplement functionality already provided by stable libraries without justification.
Do not disregard project conventions, style guides, or team standards.
Avoid excessive nested conditionals; prefer clearer control-flow alternatives or patterns.
Never import modules outside the top level unless explicitly necessary.
Do not catch overly broad exceptions without justification; prefer specific exceptions.
Do not access protected or private members outside their intended scope.
Avoid unused variables, arguments, or parameters.
Do not use overly permissive typing (e.g., `any`) without a clear reason.
Avoid circular dependencies and excessive coupling; respect dependency inversion.
Do not use outdated security practices or cryptographic algorithms.
Do not ignore localization requirements in multi-language projects.
Avoid environment inconsistencies (hardcoded configs, absolute paths).
</avoid>

<update_documentation>
When documenting modifications after an implementation, use the MCP tool `my_generate_docs_update`,
passing the correct `rootProject` and a command describing the changes (e.g., `git diff`).
Follow the tool’s instructions precisely.
</update_documentation>

<add_techs>
When introducing new technologies, frameworks, or libraries, generate documentation with usage references,
best practices, and useful links using a research tool or MCP Context7.
Save this documentation in `docs/techs/<technology-name>.md`.
</add_techs>

<task_master>
When implementing a new task via MCP Task Master:

- [ ] Analyze the task scope in MCP Task Master
- [ ] Update the task status to "In Progress"
- [ ] Execute one subtask at a time:
  - [ ] Update subtask status to "In Progress"
  - [ ] Create a detailed action plan
  - [ ] Present the plan and wait for user confirmation
  - [ ] Generate a preview of the code and wait for approval
  - [ ] Implement according to the approved plan
  - [ ] Wait for user approval and mark subtask as "Completed"
- [ ] Close the task branch
      </task_master>

<style_guides>
Always follow the official style guide of the language or framework in use.
If no project-specific guide exists, follow widely accepted community standards.

References:
golang: https://google.github.io/styleguide/go/index.html
html: https://google.github.io/styleguide/htmlcssguide.html
css: https://google.github.io/styleguide/htmlcssguide.html
python: https://google.github.io/styleguide/pyguide.html
typescript: https://google.github.io/styleguide/tsguide.html
javascript: https://google.github.io/styleguide/jsguide.html
markdown: https://google.github.io/styleguide/docguide/style.html
</style_guides>

<!-- THE RULES BELOW APPLY ONLY WHEN CREATING OR EDITING CODE FILES -->
<!-- THESE RULES TAKE PRECEDENCE OVER ALL OTHERS -->

<show_action_plan>
For any significant code addition, deletion, or modification:

- Explain what will be done and why
- Wait for explicit user confirmation before proceeding
- Adjust the plan if the user provides feedback
  </show_action_plan>

<preview_code>
For any significant code addition, deletion, or modification:

- Show a preview of the key parts of the code to be generated
- Explain important lines or decisions
- Wait for explicit user confirmation before applying changes
  </preview_code>
