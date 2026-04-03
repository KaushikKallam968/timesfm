# Project Configuration

## Coding Preferences
- JavaScript-first (~95% JS, TypeScript only for config/infra)
- Functional components, hooks-based React
- Meaningful variable names, keep functions under 20 lines
- Always use try-catch with async/await
- Never swallow errors silently
- No ESLint/Prettier — follow project conventions
- YAGNI — don't over-engineer or add speculative abstractions

## Git
- Conventional commits format (feat:, fix:, chore:, refactor:)
- Never push directly to main without asking
- Descriptive commit messages focused on "why"

## Workflow
- Read code before proposing changes
- Don't create documentation files unless asked
- Don't add docstrings, comments, or type annotations to code you didn't change
- Don't add error handling for scenarios that can't happen
- Three similar lines > premature abstraction

## Skill Routing

When the user's request matches a pattern below, follow that workflow:

### Brainstorming (before any creative/feature work)
Before building features, creating components, or modifying behavior:
1. Explore what the user actually wants (not what they literally said)
2. Ask 2-3 clarifying questions about scope, constraints, and preferences
3. Propose 2-3 approaches with trade-offs
4. Get alignment before writing code

### Writing Plans (before multi-step implementation)
When you have requirements for a multi-step task:
1. Break into discrete phases with clear deliverables
2. Identify dependencies between phases
3. Note risks and unknowns
4. Present for user approval before executing

### Executing Plans
When executing an approved plan:
1. Work through phases sequentially
2. Verify each phase before moving to next
3. If you need to deviate from the plan, explain why and get approval
4. Commit atomically per logical unit of work

### Test-Driven Development
When implementing features or fixing bugs:
1. Write a failing test first that captures the expected behavior
2. Implement the minimum code to make it pass
3. Refactor if needed while keeping tests green
4. Never claim "tests pass" without running them

### Verification Before Completion
Before claiming work is done:
1. Run the actual test/build/lint commands
2. Verify the output shows success
3. If verification fails, fix and re-verify
4. Evidence before assertions — always

### Systematic Debugging
When encountering bugs or unexpected behavior:
1. Reproduce the issue first
2. Form a hypothesis about root cause
3. Test the hypothesis with the smallest possible change
4. Don't shotgun-fix multiple things at once
5. Iron rule: no fix without understanding root cause

### Code Review (requesting)
Before merging or when asked to review:
1. Check diff against the base branch
2. Look for: security issues, logic errors, missing edge cases, API contract violations
3. Verify tests cover the changes
4. Flag anything that needs discussion vs. things that are fine

### Finishing a Branch
When implementation is complete and tests pass:
1. Review the full diff one more time
2. Present options: merge to main, create PR, or needs more work
3. Don't push without explicit approval
