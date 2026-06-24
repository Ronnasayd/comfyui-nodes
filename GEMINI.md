# GEMINI.md - Project Context

## Project Overview

**my_custom_nodes** is a collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), designed to extend its functionality with specialized image and video processing capabilities. The project was initialized using the [ComfyUI Cookiecutter Template](https://github.com/Comfy-Org/cookiecutter-comfy-extension).

### Key Features & Nodes

- **Image Processing**:
    - `AspectRatioCrop`: Crops images to match a target aspect ratio.
    - `PixelatedBorderNode`: Adds stylized pixelated borders to images.
- **Video Segmentation & Temporal Consistency**:
    - `VideoSegmentPrepare` & `VideoSegmentSave`: Infrastructure for processing long videos in smaller segments.
    - `LatentPrependCache`, `LatentExtendFrames`, `ConditioningExtendFrames`: Advanced nodes for maintaining temporal consistency between video segments using latent caching.
- **Debugging**:
    - `LatentShapeDebug` & `ConditioningShapeDebug`: Utility nodes for inspecting tensor shapes within workflows.

### Architecture

- **Entry Point**: The root `__init__.py` exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` required by ComfyUI.
- **Source Code**: Implementation logic resides in `src/my_custom_nodes/`.
    - `nodes.py`: Central registration point for all nodes.
    - `video_segment_extender.py`: Contains complex logic for video segment handling and caching.
- **Tests**: Comprehensive test suite in `tests/` using Pytest and Torch.

---

## Building and Running

### Development Setup

1. **Clone into ComfyUI**:
    ```bash
    cd ComfyUI/custom_nodes
    git clone <repository_url>
    ```
2. **Install Dependencies**:
    ```bash
    pip install -e .[dev]
    ```
3. **Install Pre-commit Hooks**:
    ```bash
    pre-commit install
    ```

### Running Tests

Execute the full test suite using Pytest:

```bash
pytest tests/
```

### Linting and Type Checking

The project uses `ruff` for linting and `mypy` for static type checking:

```bash
# Linting
ruff check .

# Type Checking
mypy .
```

---

## Development Conventions

### Coding Style

- **Linter**: `ruff` is configured with a line length of 140 characters and specific safety checks (e.g., banning `exec`/`eval`).
- **Type Safety**: `mypy` is used in `strict` mode to ensure high type safety across the codebase (except for tests where some rules are relaxed).
- **Naming**: Follows standard Python (PEP 8) and ComfyUI naming conventions (PascalCase for Node Classes).

### Testing Practices

- **Framework**: `pytest`.
- **Standards**: Every new node or significant logic change should be accompanied by unit tests in `tests/test_my_custom_nodes.py`.
- **CI/CD**: GitHub Actions (`build-pipeline.yml`) automatically run linting and tests on every Pull Request to `main` or `master`.

### ComfyUI Integration

- All exported nodes must be added to `NODE_CLASS_MAPPINGS` in `src/my_custom_nodes/nodes.py`.
- Metadata such as `RETURN_TYPES`, `FUNCTION`, and `CATEGORY` must be explicitly defined in each node class.
- The project follows the `tool.comfy` configuration in `pyproject.toml` for registry publishing.

@.github/instructions/code.instructions.md

@.github/instructions/orchestration.instructions.md

@.github/instructions/agent.instructions.md

<!-- INIT AUTO-CONTEXT -->

---
description: Agent behavior rules.
applyTo: "**/*"
---

## Environments

- JS/TS: use `yarn`, not `npm`, unless project says `npm`.
- Python: use `pip` + `venv`.
- Multi-language versions: use `asdf` when fit.

## Documentation

Search context in this order:

- `docs/SUMMARY.md`, `README.md`, `GEMINI.md`, `CLAUDE.md`, `AGENTS.md`
- `docs/architecture.md`, `docs/setup.md`, `docs/usage.md`, `docs/modules/**`, `docs/contribution.md`, `docs/faq.md`
- `docs/adr/**`, `docs/techs/**`, `docs/misc/**`

When entering module, check for `CONTEXT.md` inside module dir. File holds module-only context.

Load only minimum sections needed for task domain.

## Always Use Interactive Question Tools

For every user question, use interactive question tool. No exceptions for context, type, or intent.

Use this for clarifications, options, confirmations, preference checks, all user interactions.

- **VS Code (GitHub Copilot)**: Use `vscode_askQuestions`
- **Other environments**: Use equivalent interactive question tools available in your context
- **Fallback**: if no interactive tools exist, use labeled options (A, B, C... Z)

If interactive tool exists, never ask plain-text question.
---
description: Rules for code generation and modification tasks.
applyTo: "**/*.ts, **/*.js, **/*.py, **/*.java, **/*.go, **/*.css, **/*.cpp, **/*.c, **/*.vue, **/*.jsx, **/*.tsx"
---

# Code Standards

## Language

All code, comments, variable names, and documentation must be written in **English**.

```ts
// ✅ Good
const userAge = 25;
// ❌ Bad
const idadeUsuario = 25;
```

---

## Naming Conventions

### camelCase — variables, methods, functions

```ts
// ✅ Good
const maxRetries = 3;
function fetchUserData(userId: string) {}
const handleSubmit = () => {};

// ❌ Bad
const MaxRetries = 3;
function FetchUserData(UserId: string) {}
```

### PascalCase — classes and interfaces

```ts
// ✅ Good
class UserRepository {}
interface PaymentGateway {}

// ❌ Bad
class userRepository {}
interface payment_gateway {}
```

---

## Methods and Functions Must Start with a Verb

```ts
// ✅ Good
function getUserById(id: string): User {}
function validateEmail(email: string): boolean {}
function sendWelcomeEmail(user: User): void {}

// ❌ Bad
function userData(id: string): User {}
function emailCheck(email: string): boolean {}
```

---

## No Magic Numbers — use named constants

```ts
// ✅ Good
const MAX_LOGIN_ATTEMPTS = 5;
if (loginAttempts >= MAX_LOGIN_ATTEMPTS) {
  lockAccount();
}

// ❌ Bad
if (loginAttempts >= 5) {
  lockAccount();
}
```

---

## Avoid Nesting More Than 2 if/else Levels

Deeply nested conditionals harm readability. Use early returns or extract functions.

```ts
// ✅ Good
function processOrder(order: Order): void {
  if (!order) return;
  if (!order.isPaid) return;
  shipOrder(order);
}

// ❌ Bad
function processOrder(order: Order): void {
  if (order) {
    if (order.isPaid) {
      if (order.items.length > 0) {
        shipOrder(order);
      }
    }
  }
}
```

---

## Avoid More Than 3 Parameters

When more arguments are needed, use an object parameter.

```ts
// ✅ Good
function createUser({ name, email, role }: CreateUserParams): User {}

// ❌ Bad
function createUser(
  name: string,
  email: string,
  role: string,
  age: number
): User {}
```

---

## Avoid switch/case — prefer object maps or polymorphism

```ts
// ✅ Good
const statusLabels: Record<OrderStatus, string> = {
  pending: "Pending",
  paid: "Paid",
  shipped: "Shipped"
};
const label = statusLabels[status];

// ❌ Bad
switch (status) {
  case "pending":
    return "Pending";
  case "paid":
    return "Paid";
  case "shipped":
    return "Shipped";
}
```

---

## Prefer `const` and `let` — never use `var`

```ts
// ✅ Good
const BASE_URL = "https://api.example.com";
let retryCount = 0;

// ❌ Bad
var BASE_URL = "https://api.example.com";
var retryCount = 0;
```

---

## Keep Methods and Functions Below 30 Lines

If a function exceeds 30 lines, extract sub-responsibilities into smaller functions.

```ts
// ✅ Good
function processPayment(payment: Payment): PaymentResult {
  validatePayment(payment);
  const charged = chargeCard(payment);
  return buildPaymentResult(charged);
}

function validatePayment(payment: Payment): void {
  if (!payment.amount || payment.amount <= 0) throw new Error("Invalid amount");
  if (!payment.cardToken) throw new Error("Missing card token");
}

// ❌ Bad — everything crammed into one long function
function processPayment(payment: Payment): PaymentResult {
  if (!payment.amount || payment.amount <= 0) throw new Error("Invalid amount");
  if (!payment.cardToken) throw new Error("Missing card token");
  // ... 25 more lines of mixed logic
}
```

---

## Rules

- **Integrate new code compatibly with the existing codebase; follow established patterns and boundaries.**

  ```ts
  // ❌ BAD: Inventing a new fetch wrapper when the codebase already uses apiClient
  const res = await fetch("/api/users");

  // ✅ GOOD: Reuse the existing abstraction
  const res = await apiClient.get("/users");
  ```

- **Avoid breaking backward compatibility unless required; document migration steps if you do.**

  ```ts
  // ❌ BAD: Renaming a public function used in 30 places without notice
  export function fetchUser(id: string) { ... }  // was getUser()

  // ✅ GOOD: Keep the old name as an alias and note the deprecation
  /** @deprecated Use fetchUser instead */
  export const getUser = fetchUser;
  ```

- **Include automated tests, examples, or validation instructions whenever technically feasible.**

  ```ts
  // ❌ BAD: Shipping a new formatCurrency() utility with no tests
  export function formatCurrency(amount: number) { ... }

  // ✅ GOOD: Add at least a basic test alongside
  it('formats USD correctly', () => {
    expect(formatCurrency(1000)).toBe('$1,000.00');
  });
  ```

- **Include clear documentation when introducing significant new components, endpoints, or public interfaces.**

  ```ts
  // ❌ BAD: Exporting a new hook with no JSDoc
  export function usePayment() { ... }

  // ✅ GOOD: Document purpose, params, and return value
  /**
   * Handles payment submission and tracks status.
   * @returns { submit, status, error }
   */
  export function usePayment() { ... }
  ```

## Anti-Patterns

- **Do not use confusing, generic, or abbreviated identifiers.**

  ```ts
  // ❌ BAD
  const x = users.filter((u) => u.a > 0);

  // ✅ GOOD
  const activeUsers = users.filter((user) => user.age > 0);
  ```

- **Do not duplicate code; prefer reuse through functions, modules, or abstractions.**

  ```ts
  // ❌ BAD: Same validation logic copy-pasted in three components
  if (!email.includes("@")) throw new Error("Invalid email");

  // ✅ GOOD: Extract once
  import { validateEmail } from "@/utils/validation";
  validateEmail(email);
  ```

- **Do not use magic values; use named constants or enums.**

  ```ts
  // ❌ BAD
  if (user.role === 3) { ... }

  // ✅ GOOD
  const UserRole = { Admin: 3 } as const;
  if (user.role === UserRole.Admin) { ... }
  ```

- **Do not mix multiple responsibilities in a single function, class, or module.**

  ```ts
  // ❌ BAD: One function fetches data, formats it, and sends an email
  async function processOrder(id: string) {
    const order = await db.find(id);
    const formatted = formatOrder(order);
    await mailer.send(formatted);
  }

  // ✅ GOOD: Split into focused units
  const order = await fetchOrder(id);
  const formatted = formatOrder(order);
  await sendOrderConfirmation(formatted);
  ```

- **Do not ignore performance in critical paths (inefficient loops, heavy queries, blocking calls).**

  ```ts
  // ❌ BAD: N+1 query inside a loop
  for (const user of users) {
    user.orders = await db.orders.findMany({ where: { userId: user.id } });
  }

  // ✅ GOOD: Batch fetch
  const orders = await db.orders.findMany({
    where: { userId: { in: userIds } }
  });
  ```

- **Do not leave dead code, commented-out code, or unused functions.**

  ```ts
  // ❌ BAD
  // const oldHandler = () => { ... };
  export function unusedHelper() { ... }

  // ✅ GOOD: Delete unused code; use version control to recover it if needed
  ```

- **Do not catch overly broad exceptions without justification; prefer specific exception types.**

  ```ts
  // ❌ BAD
  try { ... } catch (e) { console.log(e); }

  // ✅ GOOD
  try { ... } catch (e) {
    if (e instanceof NetworkError) handleNetworkError(e);
    else throw e;
  }
  ```

- **Avoid circular dependencies and excessive coupling; respect dependency inversion.**

  ```ts
  // ❌ BAD: services/user.ts imports from services/order.ts which imports services/user.ts
  import { getUser } from "./user"; // inside order.ts — circular

  // ✅ GOOD: Extract shared types/interfaces to a shared layer both can import
  import type { UserId } from "@/types/user";
  ```

- **Avoid hardcoded configs or absolute paths; do not create environment inconsistencies.**

  ```ts
  // ❌ BAD
  const API_URL = "http://localhost:3000/api";

  // ✅ GOOD
  const API_URL = process.env.NEXT_PUBLIC_API_URL;
  ```

- **Avoid outdated cryptographic algorithms or security practices.**

  ```ts
  // ❌ BAD: MD5 for password hashing
  const hash = crypto.createHash("md5").update(password).digest("hex");

  // ✅ GOOD: Use bcrypt or argon2
  const hash = await bcrypt.hash(password, 12);
  ```
---
description: How to use the mcp-manager tool
applyTo: "**/*"
---

The **mcp-manager** is a Model Context Protocol (MCP) manager that allows access to tools from different servers. Here is the practical guide:

## 1️⃣ **Discover Available Tools**

```bash
# List all MCP servers
mcp_mcp-manager_list_servers

# Search for tools by keyword
mcp_mcp-manager_search_tools
  query: "your search term"
  max_results: 10

# Get tools from a specific server
mcp_mcp-manager_get_tools_by_server
  server: "taskmaster-ai"  # or: exa, csv-editor, burp, etc.
```

## 2️⃣ **Understand a Tool's Structure**

```bash
# View the complete schema (parameters, types, etc.)
mcp_mcp-manager_get_tool_schema
  server: "taskmaster-ai"
  tool_name: "get_tasks"
```

## 3️⃣ **Execute a Tool**

```bash
# General syntax
mcp_mcp-manager_call_tool
  server: "server-name"
  tool_name: "tool-name"
  arguments: { /* tool parameters */ }
```

### Practical Examples:

**Example 1: Get tasks from TaskMaster**

```bash
mcp_mcp-manager_call_tool
  server: "taskmaster-ai"
  tool_name: "get_tasks"
  arguments: { "status": "not-started" }
```

**Example 2: Search the web (Exa)**

```bash
mcp_mcp-manager_call_tool
  server: "exa"
  tool_name: "web_search_exa"
  arguments: { "query": "TypeScript best practices" }
```

**Example 3: Update task status**

```bash
mcp_mcp-manager_call_tool
  server: "taskmaster-ai"
  tool_name: "set_task_status"
  arguments: {
    "task_id": "task-123",
    "status": "completed"
  }
```

## 💡 **Typical Workflow**

1. **Discover** → `list_servers` → `get_tools_by_server`
2. **Understand** → `get_tool_schema`
3. **Execute** → `call_tool` with the correct arguments
4. **Interpret results** → use the tool output

---

**Want to execute something specific?** Tell me which server/tool you want to use, and I'll call it for you! 🚀
---
description: Test Standards
applyTo: "**/*.test.ts,**/*.test.js"
---

# Test Standards

## Cover Code with Tests

Every non-trivial function, service, and component must have corresponding tests. Aim for high coverage on business logic.

---

## Keep Tests Independent

Each test must be able to run in isolation. Tests must not depend on the order of execution or shared mutable state.

```ts
// ✅ Good — each test sets up its own data
it("should return the user by id", async () => {
  const user = await createUser({ name: "Alice" });
  const result = await getUserById(user.id);
  expect(result.name).toBe("Alice");
});

// ❌ Bad — relies on a user created by a previous test
it("should return the user by id", async () => {
  const result = await getUserById(globalUserId); // depends on external state
  expect(result.name).toBe("Alice");
});
```

---

## Use Given/When/Then or Arrange/Act/Assert

Structure each test with a clear setup, action, and assertion phase. Use comments to separate phases when it improves clarity.

```ts
// ✅ Good — Arrange / Act / Assert
it("should apply a 10% discount for premium users", () => {
  // Arrange
  const user = buildUser({ isPremium: true });
  const order = buildOrder({ total: 100 });

  // Act
  const discounted = applyDiscount(order, user);

  // Assert
  expect(discounted.total).toBe(90);
});

// ✅ Good — Given / When / Then (comment style)
it("given a premium user, when discount is applied, then total is reduced by 10%", () => {
  const user = buildUser({ isPremium: true });
  const order = buildOrder({ total: 100 });
  const result = applyDiscount(order, user);
  expect(result.total).toBe(90);
});
```

---

## Mock the Date When Behavior Depends on It

Never rely on the real system clock in tests. Always control time explicitly.

```ts
// ✅ Good
it("should mark the subscription as expired", () => {
  jest.useFakeTimers().setSystemTime(new Date("2026-01-01"));
  const subscription = buildSubscription({ expiresAt: new Date("2025-12-31") });
  expect(isExpired(subscription)).toBe(true);
  jest.useRealTimers();
});

// ❌ Bad — result depends on when the test runs
it("should mark the subscription as expired", () => {
  const subscription = buildSubscription({ expiresAt: new Date("2020-01-01") });
  expect(isExpired(subscription)).toBe(true); // may fail in any year before 2020
});
```

---

## Keep Test Cases Under 100 Lines

If a test exceeds 100 lines, extract helpers or split into smaller focused tests.

---

## Make Tests Clear and Objective

Test names should describe the scenario and expected outcome. Avoid generic names.

```ts
// ✅ Good
it("should throw an error when the email is already registered");
it("should return an empty array when no orders exist for the user");
it("should format the date as DD/MM/YYYY");

// ❌ Bad
it("test email");
it("works correctly");
it("order test 2");
```

---

## Use `beforeEach` for Similar Scenarios

When multiple tests share the same setup, consolidate it in `beforeEach`.

```ts
// ✅ Good
describe("UserService", () => {
  let userService: UserService;

  beforeEach(() => {
    userService = new UserService(mockRepository);
  });

  it("should create a user", async () => {
    const user = await userService.create({
      name: "Alice",
      email: "alice@example.com"
    });
    expect(user.id).toBeDefined();
  });

  it("should throw when email is already taken", async () => {
    await expect(
      userService.create({ name: "Bob", email: "existing@example.com" })
    ).rejects.toThrow("Email already in use");
  });
});
```

---

## Use `afterEach` to Clean Up Resources

Use `afterEach` to close database connections, reset mocks, or clear side effects after each test.

```ts
// ✅ Good
describe("OrderRepository", () => {
  let db: Database;

  beforeEach(async () => {
    db = await connectTestDatabase();
  });

  afterEach(async () => {
    await db.clear();
    await db.close();
  });

  it("should persist a new order", async () => {
    const repo = new OrderRepository(db);
    const order = await repo.save(buildOrder());
    expect(order.id).toBeDefined();
  });
});

// ❌ Bad — leaving connections open between tests causes leaks and flaky tests
```
<!-- END AUTO-CONTEXT -->
